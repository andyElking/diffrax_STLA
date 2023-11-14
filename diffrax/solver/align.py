from typing import Any, Tuple, Union

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array, vmap
from jax.numpy import sqrt

from ..custom_types import Bool, DenseInfo, LevyVal, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..solution import RESULTS
from ..term import AbstractTerm, LangevinTerm
from .base import AbstractItoSolver


_ErrorEstimate = Tuple[Array, Array]
_SolverState = dict[str, Union[Array, float, Any]]


def _match_shape(c, u):
    if jnp.ndim(c) != 0 and jnp.ndim(u) == 0:
        u = u * jnp.ones_like(c)
    if jnp.ndim(u) != 0 and jnp.ndim(c) == 0:
        c = c * jnp.ones_like(u)
    return c, u


def _directly_compute_coeffs(h, γ, u):
    # compute the coefficients directly (as opposed to via Taylor expansion)
    dtype = jnp.dtype(γ)
    α = γ * h
    β = jnp.exp(-α)
    a1 = (1 - β) / γ
    a2 = u * (1 - β - α) / jnp.square(γ)
    ρ = jnp.sqrt(2 * u / γ)
    ch1 = 6 * ρ * ((1 + β) / α + 2 * (β - 1) / jnp.square(α))
    cw1 = ρ * ((β - 1) / α + 1)

    a3 = u * ((1 + α) * β - 1) / (jnp.square(γ) * h)
    a4 = a2 / h
    ρ2 = ρ * γ
    ch2 = -γ * ch1
    cw2 = ρ2 * (1 - β) / α

    out = {
        "beta": β,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "a4": a4,
        "cw1": cw1,
        "ch1": ch1,
        "cw2": cw2,
        "ch2": ch2,
    }

    return jtu.tree_map(lambda leaf: jnp.asarray(leaf, dtype=dtype), out)


def _tay_cfs_single(c, u):
    # c is γ
    dtype = jnp.dtype(c)
    c2 = jnp.square(c)
    c3 = c2 * c
    c4 = c3 * c
    c5 = c4 * c
    rcu = jnp.sqrt(u * c)

    beta = jnp.array([1, -c, c2 / 2, -c3 / 6, c4 / 24, -c5 / 120], dtype=dtype)
    a1 = jnp.array([0, 1, -c / 2, c2 / 6, -c3 / 24, c4 / 120], dtype=dtype)
    a2 = jnp.array([0, 0, -u / 2, c * u / 6, -c2 * u / 24, c3 * u / 120], dtype=dtype)
    a3 = jnp.array(
        [0, -u / 2, c * u / 3, -c2 * u / 8, c3 * u / 30, -c4 * u / 144], dtype=dtype
    )
    a4 = jnp.array(
        [0, -u / 2, c * u / 6, -c2 * u / 24, c3 * u / 120, -c4 * u / 720], dtype=dtype
    )
    cw1 = sqrt(2) * jnp.array(
        [0, rcu / 2, -c * rcu / 6, c2 * rcu / 24, -c3 * rcu / 120, c4 * rcu / 720],
        dtype=dtype,
    )
    ch1 = sqrt(2) * jnp.array(
        [0, rcu, -c * rcu / 2, 3 * c2 * rcu / 20, -c3 * rcu / 30, c4 * rcu / 168],
        dtype=dtype,
    )
    cw2 = sqrt(2) * jnp.array(
        [
            rcu,
            -c * rcu / 2,
            c2 * rcu / 6,
            -c3 * rcu / 24,
            c4 * rcu / 120,
            -c5 * rcu / 720,
        ],
        dtype=dtype,
    )
    ch2 = -c * ch1
    return {
        "beta": beta,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "a4": a4,
        "cw1": cw1,
        "ch1": ch1,
        "cw2": cw2,
        "ch2": ch2,
    }


def _comp_taylor_coeffs(γ, u):
    # When the step-size h is small the coefficients (which depend on h) need
    # to be computed via Taylor expansion to ensure numerical stability.
    # This precomputes the Taylor coefficients (depending on γ and u), which
    # are then multiplied by powers of h, to get the coefficients of ALIGN.

    if jnp.ndim(γ) == 0 and jnp.ndim(u) == 0:
        return _tay_cfs_single(γ, u)

    γ, u = _match_shape(γ, u)

    return jax.vmap(_tay_cfs_single)(γ, u)


def _eval_taylor(h, tay_cfs):
    # Multiplies the pre-computed Taylor coefficients by powers of h.
    # jax.debug.print("eval taylor for h = {h}", h=h)
    dtype = jnp.dtype(tay_cfs["a1"])
    h_powers = jnp.power(h, jnp.arange(0, 6)).astype(dtype)
    return jtu.tree_map(
        lambda tay_leaf: jnp.tensordot(tay_leaf, h_powers, axes=1), tay_cfs
    )


class ALIGN(AbstractItoSolver):
    """The Adaptive Langevin via Interpolated Gradients and Noise method
    designed by James Foster. Only works for Underdamped Langevin Diffusion
    of the form
    $d x_t = v_t dt$
    $d v_t = - γ v_t dt - u ∇f(x_t) dt + (2γu)^(1/2) dW_t$
    where v is the velocity, f is the potential, γ is the friction, and
    W is a Brownian motion.
    """

    term_structure = LangevinTerm
    interpolation_cls = LocalLinearInterpolation
    taylor_threshold: Scalar = eqx.field(static=True)

    def __init__(self, taylor_threshold: Scalar = 0.0):
        self.taylor_threshold = taylor_threshold

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 2

    def recompute_coeffs(
        self, h: Scalar, γ: Union[Array, Scalar], u: Scalar, tay_cfs: dict[str, Array]
    ):
        # Used when the step-size h changes and coefficients need to be recomputed
        # Depending on the size of h*γ choose whether the Taylor expansion or
        # direct computation is more accurate.
        cond = h * γ < self.taylor_threshold
        if jnp.ndim(γ) == 0 and jnp.ndim(u) == 0:
            return lax.cond(
                cond,
                lambda h_: _eval_taylor(h_, tay_cfs),
                lambda h_: _directly_compute_coeffs(h_, γ, u),
                h,
            )
        else:
            tay_out = _eval_taylor(h, tay_cfs)

            γ, u = _match_shape(γ, u)

            def select_tay_or_direct(dummy):
                fun = lambda gam, _u: _directly_compute_coeffs(h, gam, _u)
                direct_out = vmap(fun)(γ, u)
                return jtu.tree_map(
                    lambda tay_leaf, direct_leaf: jnp.where(
                        cond, tay_leaf, direct_leaf
                    ),
                    tay_out,
                    direct_out,
                )

            # If all entries of h*γ are below threshold, only compute tay_out
            # otherwise, compute both tay_out and direct_out and select the
            # correct one for each dimension
            return lax.cond(
                jnp.all(cond), lambda _: tay_out, select_tay_or_direct, None
            )

    def init(
        self,
        terms: LangevinTerm,
        t0: Scalar,
        t1: Scalar,
        y0: Array,
        args: PyTree,
    ) -> _SolverState:
        """
        Precompute _SolverState which carries the Taylor coefficients and the
        ALIGN coefficients (which can be computed from h and the Taylor coeffs).
        This method is FSAL, so _SolverState also carries the previous evaluation
        of grad_f.

        Args:
            terms:
            t0:
            t1:
            y0:
            args:

        Returns:
            solver_state:
        """
        γ, u, f = terms.args  # f is in fact grad(f)
        h = t1 - t0

        tay_cfs = _comp_taylor_coeffs(γ, u)
        coeffs = self.recompute_coeffs(h, γ, u, tay_cfs)

        x0, v0 = y0
        assert x0.shape == v0.shape
        assert x0.ndim in [0, 1]

        state_out = {"h": h, "taylor_coeffs": tay_cfs, "coeffs": coeffs, "f(x)": f(x0)}

        return state_out

    def step(
        self,
        terms: LangevinTerm,
        t0: Scalar,
        t1: Scalar,
        y0: Tuple[Array, Array],
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[Tuple[Array, Array], _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del made_jump
        st = solver_state
        h = t1 - t0
        γ, u, f = terms.args

        h_state = st["h"]
        tay = st["taylor_coeffs"]
        cfs = st["coeffs"]

        # If h changed recompute coefficients
        cond = jnp.isclose(h_state, h)
        cfs = lax.cond(
            cond, lambda x: x, lambda _: self.recompute_coeffs(h, γ, u, tay), cfs
        )
        st["coeffs"] = cfs
        st["h"] = h
        # jax.debug.print("{h}", h=st['h'])

        drift, diffusion = terms.terms
        # compute the Brownian increment and space-time Levy area
        _, levy = diffusion.contr(t0, t1, use_levy=True)
        assert isinstance(levy, LevyVal)
        assert levy.H is not None, (
            "The diffusion should be a ControlTerm controlled by either a"
            "VirtualBrownianTree or an UnsafeBrownianPath with"
            "`spacetime_levyarea` set to True."
        )
        w = levy.W
        hh = levy.H

        x0, v0 = y0
        assert x0.shape == v0.shape
        assert x0.ndim in [0, 1]

        assert jnp.shape(cfs["a1"]) == jnp.shape(γ) or jnp.shape(
            cfs["a1"]
        ) == jnp.shape(u)
        assert jnp.shape(γ) in [(), x0.shape]

        f0 = st["f(x)"]
        x1 = x0 + cfs["a1"] * v0 + cfs["a2"] * f0 + cfs["cw1"] * w + cfs["ch1"] * hh
        f1 = f(x1)
        assert f1.shape == f0.shape, f"f0: {f0.shape}, f1: {f1.shape}"
        st["f(x)"] = f1
        v1 = (
            cfs["beta"] * v0
            + cfs["a3"] * f0
            + cfs["a4"] * f1
            + cfs["cw2"] * w
            + cfs["ch2"] * hh
        )

        y1 = (x1, v1)
        assert v1.dtype == x1.dtype == x0.dtype
        assert x1.shape == v1.shape == x0.shape

        error_estimate = (
            jnp.zeros_like(x0),
            jnp.sqrt(jnp.sum(jnp.square(cfs["a4"] * (f1 - f0)))),
        )

        dense_info = dict(y0=y0, y1=y1)
        return y1, error_estimate, dense_info, st, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> PyTree:
        return terms.vf(t0, y0, args)
