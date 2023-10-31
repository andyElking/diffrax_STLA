from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any

import jax
import jax.tree_util as jtu
import numpy as np
import jax.numpy as jnp
from jax.numpy import sqrt
import jax.lax as lax
import equinox.internal as eqxi
from equinox.internal import ω
import equinox as eqx
from jax import debug, Array, vmap

from ..custom_types import Bool, DenseInfo, PyTree, Scalar, LevyVal
from ..local_interpolation import LocalLinearInterpolation
from ..solution import RESULTS
from ..term import AbstractTerm, MultiTerm, ODETerm, ControlTerm
from .base import AbstractItoSolver

_ErrorEstimate = Array
_SolverState = dict[str, Array | float | Any]


def match_shape(c, u):
    if jnp.ndim(c) != 0 and jnp.ndim(u) == 0:
        u = u * jnp.ones_like(c)
    if jnp.ndim(u) != 0 and jnp.ndim(c) == 0:
        c = c * jnp.ones_like(u)
    return c, u


def directly_compute_coeffs(h, γ, u):
    # compute the coefficients directly (as opposed to via Taylor expansion)
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
    return {'beta': β,
            'a1': a1,
            'a2': a2,
            'a3': a3,
            'a4': a4,
            'cw1': cw1,
            'ch1': ch1,
            'cw2': cw2,
            'ch2': ch2}

def _tay_cfs_single(c, u):
    # c is γ
    c2 = jnp.square(c)
    c3 = c2 * c
    c4 = c3 * c
    c5 = c4 * c
    rcu = jnp.sqrt(u * c)

    beta = jnp.array([1, -c, c2 / 2, -c3 / 6, c4 / 24, -c5 / 120])
    a1 = jnp.array([0, 1, -c / 2, c2 / 6, -c3 / 24, c4 / 120])
    a2 = jnp.array([0, 0, -u / 2, c * u / 6, -c2 * u / 24, c3 * u / 120])
    a3 = jnp.array([0, -u / 2, c * u / 3, -c2 * u / 8, c3 * u / 30, -c4 * u / 144])
    a4 = jnp.array([0, -u / 2, c * u / 6, -c2 * u / 24, c3 * u / 120, - c4 * u / 720])
    cw1 = sqrt(2) * jnp.array([0, rcu / 2, -c * rcu / 6, c2 * rcu / 24, -c3 * rcu / 120, c4 * rcu / 720])
    ch1 = sqrt(2) * jnp.array([0, rcu, -c * rcu / 2, 3 * c2 * rcu / 20, -c3 * rcu / 30, c4 * rcu / 168])
    cw2 = sqrt(2) * jnp.array(
        [rcu, -c * rcu / 2, c2 * rcu / 6, -c3 * rcu / 24, c4 * rcu / 120, -c5 * rcu / 720])
    ch2 = -c * ch1
    return {'beta': beta,
            'a1': a1,
            'a2': a2,
            'a3': a3,
            'a4': a4,
            'cw1': cw1,
            'ch1': ch1,
            'cw2': cw2,
            'ch2': ch2}


def comp_taylor_coeffs(γ, u):
    # When the step-size h is small the coefficients (which depend on h) need
    # to be computed via Taylor expansion to ensure numerical stability.
    # This precomputes the Taylor coefficients (depending on γ and u), which
    # are then multiplied by powers of h, to get the coefficients of ALIGN.

    if jnp.ndim(γ) == 0 and jnp.ndim(u) == 0:
        return _tay_cfs_single(γ, u)

    γ, u = match_shape(γ, u)

    return jax.vmap(_tay_cfs_single)(γ, u)


def eval_taylor(h, tay_cfs):
    # Multiplies the pre-computed Taylor coefficients by powers of h.
    # jax.debug.print("eval taylor for h = {h}", h=h)
    h_powers = jnp.power(h, jnp.arange(0, 6, dtype=jnp.dtype(h)))
    return jtu.tree_map(lambda tay_leaf: jnp.tensordot(tay_leaf, h_powers, axes=1), tay_cfs)


class ALIGN(AbstractItoSolver):
    """The Adaptive Langevin via Interpolated Gradients and Noise method
    designed by James Foster. Only works for Underdamped Langevin Diffusion
    of the form
    d x_t &= v_t dt
    d v_t &= - γ v_t dt - u ∇f(x_t) dt + (2γu)^(1/2) dW_t
    where v is the velocity, f is the potential, γ is the friction, and
    W is a Brownian motion.
    """

    term_structure = MultiTerm[Tuple[ODETerm, ControlTerm]]
    interpolation_cls = LocalLinearInterpolation
    taylor_threshold: Scalar = eqx.field(static=True)

    def __init__(self, taylor_threshold: Scalar = 0.0):
        self.taylor_threshold = taylor_threshold

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 2

    def recompute_coeffs(self, h: Scalar, γ: Array | Scalar, u: Scalar, tay_cfs: PyTree):
        # Used when the step-size h changes and coefficients need to be recomputed
        # Depending on the size of h*γ choose whether the Taylor expansion or
        # direct computation is more accurate.
        cond = h * γ < self.taylor_threshold
        if jnp.ndim(γ) == 0:
            return lax.cond(cond,
                            lambda h_: eval_taylor(h_, tay_cfs),
                            lambda h_: directly_compute_coeffs(h_, γ, u),
                            h)
        else:
            tay_out = eval_taylor(h, tay_cfs)

            γ, u = match_shape(γ, u)

            def select_tay_or_direct(dummy):
                direct_out = vmap(lambda gam, _u: directly_compute_coeffs(h, gam, _u))(γ, u)
                # debug.print("direct_out = {du}", du=direct_out)
                return jtu.tree_map(lambda tay_leaf, direct_leaf: jnp.where(cond, tay_leaf, direct_leaf),
                                    tay_out,
                                    direct_out)

            # If all entries of h*γ are below threshold, only compute tay_out
            # otherwise, compute both tay_out and direct_out and select the
            # correct one for each dimension
            return lax.cond(jnp.all(cond),
                            lambda _: tay_out,
                            select_tay_or_direct,
                            None)

    def init(
            self,
            terms: MultiTerm[Tuple[ODETerm, ControlTerm]],
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
        """
        γ, u, f = args  # f is in fact grad(f)
        h = t1 - t0

        tay_cfs = comp_taylor_coeffs(γ, u)
        coeffs = self.recompute_coeffs(h, γ, u, tay_cfs)

        assert y0.ndim == 1
        dim = int(y0.shape[0] / 2)
        assert y0.shape[0] == 2 * dim
        x0 = y0[:dim]

        return {'h': h,
                'taylor_coeffs': tay_cfs,
                'coeffs': coeffs,
                'f(x)': f(x0)}

    def step(
            self,
            terms: MultiTerm[Tuple[ODETerm, ControlTerm]],
            t0: Scalar,
            t1: Scalar,
            y0: Array,
            args: PyTree,
            solver_state: _SolverState,
            made_jump: Bool,
    ) -> Tuple[Array, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del made_jump
        st = solver_state
        h = t1 - t0
        γ, u, f = args

        h_state = st['h']
        tay = st['taylor_coeffs']
        cfs = st['coeffs']

        # If h changed recompute coefficients
        cond = jnp.isclose(h_state, h)
        cfs = lax.cond(cond, lambda x: x, lambda _: self.recompute_coeffs(h, γ, u, tay), cfs)
        st['coeffs'] = cfs
        st['h'] = h
        # jax.debug.print("{h}", h=st['h'])

        drift, diffusion = terms.terms
        levy: LevyVal = diffusion.levy_contr(t0, t1)
        w = levy.W
        hh = levy.H

        assert y0.ndim == 1
        dim = int(y0.shape[0] / 2)
        assert y0.shape[0] == 2 * dim
        x0, v0 = y0[:dim], y0[dim:]
        assert jnp.shape(cfs['a1']) == jnp.shape(γ)
        assert jnp.shape(γ) in [(), (dim,)]

        f0 = st['f(x)']
        x1 = x0 + cfs['a1'] * v0 + cfs['a2'] * f0 + cfs['cw1'] * w + cfs['ch1'] * hh
        f1 = f(x1)
        st['f(x)'] = f1
        v1 = cfs['beta'] * v0 + cfs['a3'] * f0 + cfs['a4'] * f1 + cfs['cw2'] * w + cfs['ch2'] * hh

        y1 = jnp.concatenate((x1, v1))

        error_estimate = jnp.sqrt(jnp.sum(jnp.square(cfs['a4'] * (f1 - f0))))

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
