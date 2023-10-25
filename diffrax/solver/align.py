from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any

import jax
import jax.tree_util as jtu
import numpy as np
import jax.numpy as jnp
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
        return 1.5

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
        ch2 = -(6 * ρ2 / α) * (1 + β + 2 * (β-1) / α)
        cw2 = ρ2 * (1 - β) / α

        # jax.debug.print("beta: {beta} a1: {a1}, a2: {a2}, ch1: {ch1}, cw1: {cw1}",
        #                 beta=β, a1=a1, a2=a2, ch1=ch1, cw1=cw1)
        # jax.debug.print("a3: {a3}, a4: {a4}, ch2: {ch2}, cw2: {cw2}",
        #                 a3=a3, a4=a4, ch2=ch2, cw2=cw2)

        assert y0.ndim == 1
        dim = int(y0.shape[0] / 2)
        assert y0.shape[0] == 2 * dim
        x0 = y0[:dim]

        return {'h': h,
                'gamma': γ,
                'u': u,
                'beta': β,
                'a1': a1,
                'a2': a2,
                'a3': a3,
                'a4': a4,
                'cw1': cw1,
                'ch1': ch1,
                'cw2': cw2,
                'ch2': ch2,
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
        h = t1 - t0
        γ, u, f = args
        h_s, gamma_s, u_s = solver_state['h'], solver_state['gamma'], solver_state['u']
        cond = jnp.allclose(jnp.array([h, γ, u]), jnp.array([h_s, gamma_s, u_s]))
        state = lax.cond(cond, lambda x: x, lambda _: self.init(terms, t0, t1, y0, args), solver_state)
        # jax.debug.print("{h}", h=state['h'])

        drift, diffusion = terms.terms
        levy: LevyVal = diffusion.levy_contr(t0, t1)
        w = levy.W
        hh = levy.H

        assert y0.ndim == 1
        dim = int(y0.shape[0] / 2)
        assert y0.shape[0] == 2 * dim
        x0, v0 = y0[:dim], y0[dim:]

        f0 = state['f(x)']
        x1 = x0 + state['a1'] * v0 + state['a2'] * f0 + state['cw1'] * w + state['ch1'] * hh
        f1 = f(x1)
        state['f(x)'] = f1
        v1 = state['beta'] * v0 + state['a3'] * f0 + state['a4'] * f1 + state['cw2'] * w + state['ch2'] * hh

        y1 = jnp.concatenate((x1, v1))
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, state, RESULTS.successful

    def func(
            self,
            terms: AbstractTerm,
            t0: Scalar,
            y0: PyTree,
            args: PyTree,
    ) -> PyTree:
        return terms.vf(t0, y0, args)
