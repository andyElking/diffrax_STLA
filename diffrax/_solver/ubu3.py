import math
from typing import Callable

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
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
from .._term import LangevinTerm, LangevinTuple, LangevinX
from .base import AbstractItoSolver


# UBU evaluates at l = (3 -sqrt(3))/6, at r = (3 + sqrt(3))/6 and at 1,
# so we need 3 versions of each coefficient


class _Coeffs(eqx.Module):
    beta_lr1: PyTree[Array]  # (gamma, 3, *taylor)
    a_lr1: PyTree[Array]  # (gamma, 3, *taylor)
    b_lr1: PyTree[Array]  # (gamma, 3, *taylor)
    a_third: PyTree[Array]  # (gamma, 1, *taylor)
    a_div_h: PyTree[Array]  # (gamma, 1, *taylor)

    @property
    def dtype(self):
        return jtu.tree_leaves(self.beta_lr1)[0].dtype


_coeffs_structure = jtu.tree_structure(
    _Coeffs(
        beta_lr1=jnp.array(0.0),
        a_lr1=jnp.array(0.0),
        b_lr1=jnp.array(0.0),
        a_third=jnp.array(0.0),
        a_div_h=jnp.array(0.0),
    )
)


class _SolverState(eqx.Module):
    h: RealScalarLike
    taylor_coeffs: PyTree[_Coeffs, "LangevinX"]
    coeffs: _Coeffs
    rho: PyTree[Array]
    prev_f: LangevinX


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


def _directly_compute_coeffs_leaf(h, c) -> _Coeffs:
    # compute the coefficients directly (as opposed to via Taylor expansion)
    assert c.ndim in [0, 1]
    original_shape = c.shape
    c = jnp.expand_dims(c, axis=-1)
    alpha = c * h
    l = 0.5 - math.sqrt(3) / 6
    r = 0.5 + math.sqrt(3) / 6
    l_r_1 = jnp.array([l, r, 1.0], dtype=jnp.dtype(c))
    if c.ndim == 1:
        jnp.expand_dims(l_r_1, axis=0)
    alpha_lr1 = alpha * l_r_1
    assert alpha_lr1.shape == original_shape + (
        3,
    ), f"expected {original_shape + (3,)}, got {alpha_lr1.shape}"
    beta_lr1 = jnp.exp(-alpha_lr1)
    a_lr1 = (1.0 - beta_lr1) / c
    b_lr1 = (beta_lr1 + alpha_lr1 - 1.0) / (c**2 * h)
    a_third = (1.0 - jnp.exp(-alpha / 3)) / c
    a_div_h = (1.0 - jnp.exp(-alpha)) / (c * h)

    assert a_third.shape == a_div_h.shape == original_shape + (1,)

    return _Coeffs(
        beta_lr1=beta_lr1,
        a_lr1=a_lr1,
        b_lr1=b_lr1,
        a_third=a_third,
        a_div_h=a_div_h,
    )


def _tay_cfs_single(c: Array) -> _Coeffs:
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


def _comp_taylor_coeffs_leaf(c) -> _Coeffs:
    # When the step-size h is small the coefficients (which depend on h) need
    # to be computed via Taylor expansion to ensure numerical stability.
    # This precomputes the Taylor coefficients (depending on gamma and u), which
    # are then multiplied by powers of h, to get the coefficients of ALIGN.
    if jnp.ndim(c) == 0:
        out = _tay_cfs_single(c)
    else:
        out = jax.vmap(_tay_cfs_single)(c)

    def check_shape(x):
        assert x.shape == c.shape + (3, 6) or x.shape == c.shape + (1, 6)

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

    def recompute_coeffs(
        self, h, gamma: LangevinX, tay_cfs: PyTree[_Coeffs]
    ) -> _Coeffs:
        def recompute_coeffs_leaf(c: ArrayLike, _tay_cfs: _Coeffs):
            # Used when the step-size h changes and coefficients need to be recomputed
            # Depending on the size of h*gamma choose whether the Taylor expansion or
            # direct computation is more accurate.
            cond = h * c < self.taylor_threshold
            if jnp.ndim(c) == 0:
                return lax.cond(
                    cond,
                    lambda h_: _eval_taylor(h_, _tay_cfs),
                    lambda h_: _directly_compute_coeffs_leaf(h_, c),
                    h,
                )
            else:
                cond = jnp.expand_dims(cond, axis=-1)
                tay_out = _eval_taylor(h, _tay_cfs)

                def select_tay_or_direct(dummy):
                    fun = lambda _c: _directly_compute_coeffs_leaf(h, _c)
                    direct_out = vmap(fun)(c)

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

        tree_with_cfs = jtu.tree_map(recompute_coeffs_leaf, gamma, tay_cfs)
        outer = jtu.tree_structure(gamma)
        inner = _coeffs_structure
        cfs_with_tree = jtu.tree_transpose(outer, inner, tree_with_cfs)
        assert isinstance(cfs_with_tree, _Coeffs)
        return cfs_with_tree

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

        x0, v0 = y0

        def _check_shapes(_c, _u, _x, _v):
            assert _x.ndim in [0, 1]
            assert _c.shape == _u.shape == _x.shape == _v.shape

        assert jtu.tree_all(jtu.tree_map(_check_shapes, gamma, u, x0, v0))

        h = t1 - t0

        tay_cfs = jtu.tree_map(_comp_taylor_coeffs_leaf, gamma)
        # tay_cfs have the same tree structure as gamma, with each leaf being a _Coeffs
        # and the arrays have an extra trailing dimension of 6

        coeffs = self.recompute_coeffs(h, gamma, tay_cfs)
        rho = jtu.tree_map(lambda c, _u: jnp.sqrt(2 * c * _u), gamma, u)

        def check_coeff_shapes(_x, _bet_lr1, _a_third, _tay):
            assert _bet_lr1.shape == _x.shape + (3,)
            assert _a_third.shape == _x.shape + (1,)
            assert isinstance(_tay, _Coeffs)

        jtu.tree_map(check_coeff_shapes, x0, coeffs.beta_lr1, coeffs.a_third, tay_cfs)

        state_out = _SolverState(
            h=h,
            taylor_coeffs=tay_cfs,
            coeffs=coeffs,
            rho=rho,
            prev_f=f(x0),
        )

        return state_out

    def _compute_step(
        self,
        h: RealScalarLike,
        levy: AbstractSpaceTimeTimeLevyArea,
        x0: LangevinX,
        v0: LangevinX,
        f: Callable[[LangevinX], LangevinX],
        u: LangevinX,
        cfs: _Coeffs,
        st: _SolverState,
    ) -> tuple[LangevinX, LangevinX, LangevinX]:
        w: LangevinX = levy.W
        hh: LangevinX = levy.H
        kk: LangevinX = levy.K

        def _l(coeff):
            return jtu.tree_map(lambda arr: arr[..., 0], coeff)

        def _r(coeff):
            return jtu.tree_map(lambda arr: arr[..., 1], coeff)

        def _one(coeff):
            return jtu.tree_map(lambda arr: arr[..., 2], coeff)

        beta_l = _l(cfs.beta_lr1)
        beta_r = _r(cfs.beta_lr1)
        beta_1 = _one(cfs.beta_lr1)
        a_l = _l(cfs.a_lr1)
        a_r = _r(cfs.a_lr1)
        a_1 = _one(cfs.a_lr1)
        b_l = _l(cfs.b_lr1)
        b_r = _r(cfs.b_lr1)
        b_1 = _one(cfs.b_lr1)
        a_third = _l(cfs.a_third)
        a_div_h = _l(cfs.a_div_h)

        rho_w_k = (st.rho**ω * (w**ω - 12 * kk**ω)).ω
        uh = (u**ω * h).ω
        v_tilde = (v0**ω + st.rho**ω * (hh**ω + 6 * kk**ω)).ω

        x1 = (x0**ω + a_l**ω * v_tilde**ω + b_l**ω * rho_w_k**ω).ω
        f1uh = (f(x1) ** ω * uh**ω).ω

        x2 = (
            x0**ω + a_r**ω * v_tilde**ω + b_r**ω * rho_w_k**ω - a_third**ω * f1uh**ω
        ).ω
        f2uh = (f(x2) ** ω * uh**ω).ω

        x_out = (
            x0**ω
            + a_1**ω * v_tilde**ω
            + b_1**ω * rho_w_k**ω
            - 0.5 * (a_r**ω * f1uh**ω + a_l**ω * f2uh**ω)
        ).ω

        v_out_tilde = (
            beta_1**ω * v_tilde**ω
            - 0.5 * (beta_r**ω * f1uh**ω + beta_l**ω * f2uh**ω)
            + a_div_h**ω * rho_w_k**ω
        ).ω
        v_out = (v_out_tilde**ω - st.rho**ω * (hh**ω - 6 * kk**ω)).ω

        f_fsal = (
            st.prev_f
        )  # this method is not FSAL, but this is for compatibility with the base class
        return x_out, v_out, f_fsal

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
        tay: PyTree[_Coeffs] = st.taylor_coeffs
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

        x0, v0 = y0
        x_out, v_out, f_fsal = self._compute_step(h, levy, x0, v0, f, u, cfs, st)

        def check_shapes_dtypes(_x, _v, _f, _x0):
            assert _x.dtype == _v.dtype == _f.dtype == _x0.dtype, (
                f"dtypes don't match. x0: {x0.dtype},"
                f" v_out: {_v.dtype}, x_out: {_x.dtype}, f_fsal: {_f.dtype}"
            )
            assert _x.shape == _v.shape == _f.shape == _x0.shape, (
                f"Shapes don't match. x0: {x0.shape},"
                f" v_out: {_v.shape}, x_out: {_x.shape}, f_fsal: {_f.shape}"
            )

        jtu.tree_map(check_shapes_dtypes, x_out, v_out, f_fsal, x0)

        y1 = (x_out, v_out)

        # TODO: compute error estimate

        dense_info = dict(y0=y0, y1=y1)
        st = _SolverState(
            h=h,
            taylor_coeffs=tay,
            coeffs=cfs,
            rho=st.rho,
            prev_f=f_fsal,
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
