import math

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import vmap
from jaxtyping import Array, ArrayLike, PyTree

from .._custom_types import (
    AbstractSpaceTimeTimeLevyArea,
    BoolScalarLike,
    DenseInfo,
    RealScalarLike,
)
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import LangevinTerm, LangevinTuple
from .base import AbstractItoSolver


# UBU evaluates at l = (3 -sqrt(3))/6, at r = (3 + sqrt(3))/6 and at 1,
# so we need 3 versions of each coefficient


class _Coeffs(eqx.Module):
    beta_lr1: Array  # (gamma, 3, *taylor)
    a_lr1: Array  # (gamma, 3, *taylor)
    b_lr1: Array  # (gamma, 3, *taylor)
    a_third: Array  # (gamma, 1, *taylor)
    a_div_h: Array  # (gamma, 1, *taylor)

    @property
    def dtype(self):
        return self.beta_lr1.dtype


class _SolverState(eqx.Module):
    h: RealScalarLike
    taylor_coeffs: _Coeffs
    coeffs: _Coeffs
    rho: Array


# CONCERNING COEFFICIENTS:
# The coefficients used in a step of UBU depend on
# the time increment h, and the parameter gamma.
# Assuming the modelled SDE stays the same (i.e. gamma is fixed),
# then these coefficients must be recomputed each time h changes.
# Furthermore, for very small h, directly computing the coefficients
# via the function below can cause large floating point errors.
# Hence, we pre-compute the Taylor expansion of the UBU coefficients
# around h=0. Then we can compute the UBU coefficients either via
# the Taylor expansion, or via direct computation.
# In short the Taylor coefficients give a Taylor expansion with which
# one can compute the UBU coefficients more precisely for a small h.


def _directly_compute_coeffs(h, gamma) -> _Coeffs:
    # compute the coefficients directly (as opposed to via Taylor expansion)
    assert gamma.ndim in [0, 1]
    original_shape = gamma.shape
    gamma = jnp.expand_dims(gamma, axis=-1)
    alpha = gamma * h
    l = 0.5 - math.sqrt(3) / 6
    r = 0.5 + math.sqrt(3) / 6
    l_r_1 = jnp.array([l, r, 1.0], dtype=jnp.dtype(gamma))
    if gamma.ndim == 1:
        jnp.expand_dims(l_r_1, axis=0)
    alpha_lr1 = alpha * l_r_1
    assert alpha_lr1.shape == original_shape + (
        3,
    ), f"expected {original_shape + (3,)}, got {alpha_lr1.shape}"
    beta_lr1 = jnp.exp(-alpha_lr1)
    a_lr1 = (1.0 - beta_lr1) / gamma
    b_lr1 = (beta_lr1 + alpha_lr1 - 1.0) / (gamma**2 * h)
    a_third = (1.0 - jnp.exp(-alpha / 3)) / gamma
    a_div_h = (1.0 - jnp.exp(-alpha)) / (gamma * h)

    assert a_third.shape == a_div_h.shape == original_shape + (1,)

    return _Coeffs(
        beta_lr1=beta_lr1,
        a_lr1=a_lr1,
        b_lr1=b_lr1,
        a_third=a_third,
        a_div_h=a_div_h,
    )


def _tay_cfs_single(c) -> _Coeffs:
    # c is gamma
    assert c.ndim == 0
    dtype = jnp.dtype(c)
    c2 = jnp.square(c)
    c3 = c2 * c
    c4 = c3 * c
    c5 = c4 * c

    l = 0.5 - math.sqrt(3) / 6
    r = 0.5 + math.sqrt(3) / 6
    lr1 = jnp.expand_dims(jnp.array([l, r, 1.0], dtype=dtype), axis=-1)
    exponents = jnp.expand_dims(jnp.arange(0, 6, dtype=dtype), axis=0)
    lr1_pows = jnp.power(lr1, exponents)
    assert lr1_pows.shape == (3, 6)

    beta = jnp.array([1, -c, c2 / 2, -c3 / 6, c4 / 24, -c5 / 120], dtype=dtype)
    a = jnp.array([0, 1, -c / 2, c2 / 6, -c3 / 24, c4 / 120], dtype=dtype)
    b = jnp.array([0, 1 / 2, -c / 6, c2 / 24, -c3 / 120, c4 / 720], dtype=dtype)

    beta_lr1 = lr1_pows * jnp.expand_dims(beta, axis=0)
    a_lr1 = lr1_pows * jnp.expand_dims(a, axis=0)
    # b needs an extra power of l and r
    b_lr1 = lr1_pows * lr1 * jnp.expand_dims(b, axis=0)
    assert beta_lr1.shape == a_lr1.shape == b_lr1.shape == (3, 6)

    # a_third = (1 - exp(-1/3 * gamma * h))/gamma
    a_third = jnp.array(
        [0, 1 / 3, -c / 18, c2 / 162, -c3 / 1944, c4 / 29160], dtype=dtype
    )
    a_third = jnp.expand_dims(a_third, axis=0)
    a_div_h = jnp.array([1, -c / 2, c2 / 6, -c3 / 24, c4 / 120, -c5 / 720], dtype=dtype)
    a_div_h = jnp.expand_dims(a_div_h, axis=0)
    assert a_third.shape == a_div_h.shape == (1, 6)

    return _Coeffs(
        beta_lr1=beta_lr1,
        a_lr1=a_lr1,
        b_lr1=b_lr1,
        a_third=a_third,
        a_div_h=a_div_h,
    )


def _comp_taylor_coeffs(gamma) -> _Coeffs:
    # When the step-size h is small the coefficients (which depend on h) need
    # to be computed via Taylor expansion to ensure numerical stability.
    # This precomputes the Taylor coefficients (depending on gamma and u), which
    # are then multiplied by powers of h, to get the coefficients of ALIGN.
    if jnp.ndim(gamma) == 0:
        out = _tay_cfs_single(gamma)
    else:
        out = jax.vmap(_tay_cfs_single)(gamma)

    def check_shape(x):
        assert x.shape == gamma.shape + (3, 6) or x.shape == gamma.shape + (1, 6)

    jtu.tree_map(check_shape, out)
    return out


def _eval_taylor(h, tay_cfs: _Coeffs) -> _Coeffs:
    # Multiplies the pre-computed Taylor coefficients by powers of h.
    # jax.debug.print("eval taylor for h = {h}", h=h)
    dtype = tay_cfs.dtype
    h_powers = jnp.power(h, jnp.arange(0, 6, dtype=h.dtype)).astype(dtype)
    return jtu.tree_map(
        lambda tay_leaf: jnp.tensordot(tay_leaf, h_powers, axes=1), tay_cfs
    )


class UBU3(AbstractItoSolver):
    r"""The third order version of the UBU method by Daire O'Kane and James Foster.
    Works for underdamped Langevin SDEs of the form

    $$d x_t = v_t dt$$

    $$d v_t = - gamma v_t dt - u ∇f(x_t) dt + (2gammau)^(1/2) dW_t$$

    where $v$ is the velocity, $f$ is the potential, $gamma$ is the friction, and
    $W$ is a Brownian motion.
    """

    term_structure = LangevinTerm
    interpolation_cls = LocalLinearInterpolation
    taylor_threshold: RealScalarLike = eqx.field(static=True)

    @property
    def minimal_levy_area(self):
        return AbstractSpaceTimeTimeLevyArea

    def __init__(self, taylor_threshold: RealScalarLike = 0.0):
        r"""**Arguments:**

        - `taylor_threshold`: If the product `h*gamma` is less than this, then
        the Taylor expansion will be used to compute the coefficients.
        Otherwise they will be computed directly. When using float32, the
        empirically optimal value is 0.1, and for float64 about 0.01.
        """
        self.taylor_threshold = taylor_threshold

    def order(self, terms):
        return 3

    def strong_order(self, terms):
        return 3.0

    def recompute_coeffs(self, h: RealScalarLike, gamma: Array, tay_cfs: _Coeffs):
        # Used when the step-size h changes and coefficients need to be recomputed
        # Depending on the size of h*gamma choose whether the Taylor expansion or
        # direct computation is more accurate.
        cond = h * gamma < self.taylor_threshold
        if jnp.ndim(gamma) == 0:
            return lax.cond(
                cond,
                lambda h_: _eval_taylor(h_, tay_cfs),
                lambda h_: _directly_compute_coeffs(h_, gamma),
                h,
            )
        else:
            cond = jnp.expand_dims(cond, axis=-1)
            tay_out = _eval_taylor(h, tay_cfs)

            def select_tay_or_direct(dummy):
                fun = lambda gam: _directly_compute_coeffs(h, gam)
                direct_out = vmap(fun)(gamma)

                def _choose(tay_leaf, direct_leaf):
                    assert tay_leaf.ndim == direct_leaf.ndim == cond.ndim
                    return jnp.where(cond, tay_leaf, direct_leaf)

                return jtu.tree_map(_choose, tay_out, direct_out)

            # If all entries of h*gamma are below threshold, only compute tay_out
            # otherwise, compute both tay_out and direct_out and select the
            # correct one for each dimension
            return lax.cond(
                jnp.all(cond), lambda _: tay_out, select_tay_or_direct, None
            )

    def init(
        self,
        terms: LangevinTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: LangevinTuple,
        args: PyTree,
    ) -> _SolverState:
        """Precompute _SolverState which carries the Taylor coefficients and the
        ALIGN coefficients (which can be computed from h and the Taylor coeffs).
        This method is FSAL, so _SolverState also carries the previous evaluation
        of grad_f.
        """
        assert isinstance(terms, LangevinTerm)
        gamma, u, f = terms.args  # f is in fact grad(f)
        assert gamma.shape == u.shape
        h = t1 - t0

        tay_cfs = _comp_taylor_coeffs(gamma)
        coeffs = self.recompute_coeffs(h, gamma, tay_cfs)
        rho = jnp.sqrt(2 * gamma * u)

        x0, v0 = y0
        assert x0.shape == v0.shape
        assert x0.ndim in [0, 1]

        state_out = _SolverState(
            h=h,
            taylor_coeffs=tay_cfs,
            coeffs=coeffs,
            rho=rho,
        )

        return state_out

    def step(
        self,
        terms: LangevinTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: LangevinTuple,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[LangevinTuple, None, DenseInfo, _SolverState, RESULTS]:
        del made_jump, args
        st = solver_state
        h = t1 - t0
        assert isinstance(terms, LangevinTerm)
        gamma, u, f = terms.args

        h_state = st.h
        tay: _Coeffs = st.taylor_coeffs
        cfs = st.coeffs

        # If h changed recompute coefficients
        cond = jnp.isclose(h_state, h)
        cfs: _Coeffs = lax.cond(
            cond, lambda x: x, lambda _: self.recompute_coeffs(h, gamma, tay), cfs
        )

        drift, diffusion = terms.term.terms
        # compute the Brownian increment and space-time Levy area
        levy = diffusion.contr(t0, t1, use_levy=True)
        assert isinstance(levy, AbstractSpaceTimeTimeLevyArea)
        assert (
            levy.H is not None and levy.K is not None
        ), "The Brownian motion must have `levy_area=diffrax.SpaceTimeTimeLevyArea`"
        w: ArrayLike = levy.W
        hh: ArrayLike = levy.H
        kk: ArrayLike = levy.K

        x0, v0 = y0
        assert x0.shape == v0.shape
        assert x0.ndim in [0, 1]
        assert gamma.shape in [(), x0.shape]
        assert cfs.beta_lr1.shape == gamma.shape + (3,)
        assert cfs.a_third.shape == gamma.shape + (1,)

        def _l(coeff):
            return coeff[..., 0]

        def _r(coeff):
            return coeff[..., 1]

        def _one(coeff):
            return coeff[..., 2]

        beta = cfs.beta_lr1
        a = cfs.a_lr1
        b = cfs.b_lr1
        a_third = cfs.a_third[..., 0]
        a_div_h = cfs.a_div_h[..., 0]

        rho_w_k = st.rho * (w - 12 * kk)
        uh = u * h
        v_tilde = v0 + st.rho * (hh + 6 * kk)

        x1 = x0 + _l(a) * v_tilde + _l(b) * rho_w_k
        f1uh = f(x1) * uh

        x2 = x0 + _r(a) * v_tilde + _r(b) * rho_w_k - a_third * f1uh
        f2uh = f(x2) * uh

        x_out = (
            x0
            + _one(a) * v_tilde
            + _one(b) * rho_w_k
            - 0.5 * (_r(a) * f1uh + _l(a) * f2uh)
        )

        v_out_tilde = (
            _one(beta) * v_tilde
            - 0.5 * (_r(beta) * f1uh + _l(beta) * f2uh)
            + a_div_h * rho_w_k
        )
        v_out = v_out_tilde - st.rho * (hh - 6 * kk)

        y1 = (x_out, v_out)
        assert v_out.dtype == x_out.dtype == x0.dtype, (
            f"dtypes don't match. x0: {x0.dtype},"
            f" v_out: {v_out.dtype}, x_out: {x_out.dtype}"
        )
        assert x_out.shape == v_out.shape == x0.shape, (
            f"Shapes don't match. x0: {x0.shape},"
            f" v_out: {v_out.shape}, x_out: {x_out.shape}"
        )

        # TODO: compute error estimate

        dense_info = dict(y0=y0, y1=y1)
        st = _SolverState(
            h=h,
            taylor_coeffs=tay,
            coeffs=cfs,
            rho=st.rho,
        )
        return y1, None, dense_info, st, RESULTS.successful

    def func(
        self,
        terms: LangevinTerm,
        t0: RealScalarLike,
        y0: LangevinTuple,
        args: PyTree,
    ):
        return terms.vf(t0, y0, args)
