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
from jax import debug, Array

from ..custom_types import Bool, DenseInfo, PyTree, Scalar, LevyVal
from ..local_interpolation import LocalLinearInterpolation
from ..solution import RESULTS
from ..term import AbstractTerm, MultiTerm, ODETerm, ControlTerm
from .base import AbstractItoSolver

_ErrorEstimate = None
_SolverState = dict[str, Array | float | Any]


class ALIGN(AbstractItoSolver):
    """Additive-Noise Stochastic Runge-Kutta method.
    For description see StochasticButcherTableau.
    """

    term_structure = MultiTerm[Tuple[ODETerm, ControlTerm]]
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 2

    @staticmethod
    def taylor_coeffs(args):
        c, u, _ = args
        c2 = jnp.square(c)
        c3 = c2 * c
        c4 = c3 * c
        c5 = c4 * c
        rcu = jnp.sqrt(u * c)

        β = jnp.array([1, -c, c2 / 2, -c3 / 6, c4 / 24, -c5/120])
        a1 = jnp.array([0, 1, -c / 2, c2 / 6, -c3 / 24, c4/120])
        a2 = jnp.array([0, 0, -u / 2, c * u / 6, -c2 * u / 24, c3 * u / 120])
        a3 = jnp.array([0, -u / 2, c * u / 3, -c2 * u / 8, c3 * u / 30, -c4*u/144])
        a4 = jnp.array([0, -u / 2, c * u / 6, -c2 * u / 24, c3 * u / 120, - c4*u/720])
        cw1 = sqrt(2) * jnp.array([0, rcu/2, -c * rcu / 6, c2 * rcu / 24, -c3 * rcu / 120, c4*rcu/720])
        ch1 = sqrt(2) * jnp.array([0, rcu, -c * rcu / 2, 3 * c2 * rcu / 20, -c3 * rcu / 30, c4 * rcu/168])
        cw2 = sqrt(2) * jnp.array([rcu, -c * rcu / 2, c2 * rcu / 6, -c3 * rcu / 24, c4 * rcu / 120, -c5*rcu/720])
        ch2 = -c * ch1
        return {'beta': β,
                'a1': a1,
                'a2': a2,
                'a3': a3,
                'a4': a4,
                'cw1': cw1,
                'ch1': ch1,
                'cw2': cw2,
                'ch2': ch2}

    @staticmethod
    def eval_taylor(h, coeffs):
        # jax.debug.print("eval taylor for h = {h}", h=h)
        h_powers = jnp.power(h, jnp.arange(0, 6, dtype=jnp.dtype(h)))
        return jtu.tree_map(lambda tay_coeffs: jnp.vdot(h_powers, tay_coeffs), coeffs)

    @staticmethod
    def directly_compute_coeffs(h, args):
        # jax.debug.print("direct compute coeffs for h = {h}", h=h)
        γ, u, _ = args  # f is in fact grad(f)
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
        # ch2 = -(6 * ρ2 / α) * (1 + β + 2 * (β-1) / α)
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

    def recompute_coeffs(self, h, args, taylor_coeffs):
        # jax.debug.print("recomputing coeffs for h = {h}", h=h)
        γ, _, _ = args
        return lax.cond(h * γ < 0.05,
                        lambda h_: self.eval_taylor(h_, taylor_coeffs),
                        lambda h_: self.directly_compute_coeffs(h_, args),
                        h)

    def init(
            self,
            terms: MultiTerm[Tuple[ODETerm, ControlTerm]],
            t0: Scalar,
            t1: Scalar,
            y0: Array,
            args: PyTree,
    ) -> _SolverState:
        γ, u, f = args  # f is in fact grad(f)
        h = t1 - t0

        # jax.debug.print("beta: {beta} a1: {a1}, a2: {a2}, ch1: {ch1}, cw1: {cw1}",
        #                 beta=β, a1=a1, a2=a2, ch1=ch1, cw1=cw1)
        # jax.debug.print("a3: {a3}, a4: {a4}, ch2: {ch2}, cw2: {cw2}",
        #                 a3=a3, a4=a4, ch2=ch2, cw2=cw2)

        taylor_coeffs = self.taylor_coeffs(args)
        coeffs = self.recompute_coeffs(h, args, taylor_coeffs)

        assert y0.ndim == 1
        dim = int(y0.shape[0] / 2)
        assert y0.shape[0] == 2 * dim
        x0 = y0[:dim]

        return {'h': h,
                'taylor_coeffs': taylor_coeffs,
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
        cfs = lax.cond(cond, lambda x: x, lambda _: self.recompute_coeffs(h, args, tay), cfs)
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

        f0 = st['f(x)']
        x1 = x0 + cfs['a1'] * v0 + cfs['a2'] * f0 + cfs['cw1'] * w + cfs['ch1'] * hh
        f1 = f(x1)
        st['f(x)'] = f1
        v1 = cfs['beta'] * v0 + cfs['a3'] * f0 + cfs['a4'] * f1 + cfs['cw2'] * w + cfs['ch2'] * hh

        y1 = jnp.concatenate((x1, v1))
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, st, RESULTS.successful

    def func(
            self,
            terms: AbstractTerm,
            t0: Scalar,
            y0: PyTree,
            args: PyTree,
    ) -> PyTree:
        return terms.vf(t0, y0, args)
