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


@dataclass(frozen=True)
class StochasticButcherTableau:
    """A Butcher Tableau for Additive-noise SRK methods.

    Given the SDE
    dX_t = f(t, X_t) dt + σ dW_t

    We construct the SRK as follows:
    y_1 = y_0 + h (Σ_{j=1}^s b_j k_j) + σ * (cw_last * ΔW + ch_last * ΔH)
    k_j = f(t_0 + c_j h, z_j)
    z_j = y_0 + h (Σ_{i=1}^{j-1} a_j_i k_j) + σ * (cw_j * ΔW + ch_j * ΔH)

    where ΔW := W_{t0, t1} is the increment of the Brownian motion and
    ΔH := H_{t0, t1} is its corresponding space-time Levy Area.
    """

    # Only supports explicit SRK so far
    c: np.ndarray
    b: np.ndarray
    a: list[np.ndarray]

    # coefficients for W and H (of shape (len(c)+1,)
    cw: np.ndarray
    ch: np.ndarray
    cw_last: Scalar
    ch_last: Scalar

    def __post_init__(self):
        assert self.c.ndim == 1
        for a_i in self.a:
            assert a_i.ndim == 1
        assert self.b.ndim == 1
        assert self.c.shape[0] == len(self.a)
        assert all(i + 1 == a_i.shape[0] for i, a_i in enumerate(self.a))
        assert self.c.shape[0] + 1 == self.b.shape[0]
        assert self.cw.shape[0] == self.b.shape[0]
        assert self.ch.shape[0] == self.b.shape[0]
        for i, (a_i, c_i) in enumerate(zip(self.a, self.c)):
            assert np.allclose(sum(a_i), c_i)
        assert np.allclose(sum(self.b), 1.0)

        # TODO: add checks for whether the method is FSAL


class ANSR(AbstractItoSolver):
    """Additive-Noise Stochastic Runge-Kutta method.
    For description see StochasticButcherTableau.
    """

    term_structure = MultiTerm[Tuple[ODETerm, ControlTerm]]
    interpolation_cls = LocalLinearInterpolation
    tableau: StochasticButcherTableau

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.5

    def init(
            self,
            terms: MultiTerm[Tuple[ODETerm, ControlTerm]],
            t0: Scalar,
            t1: Scalar,
            y0: PyTree,
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

        ch2 = -6 * ρ / α * (1 + β + 2 * (1 - β) / α)
        cw2 = ρ * (1 - β) / α

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
                'ch2': ch2}

    def step(
            self,
            terms: MultiTerm[Tuple[ODETerm, ControlTerm]],
            t0: Scalar,
            t1: Scalar,
            y0: PyTree,
            args: PyTree,
            solver_state: _SolverState,
            made_jump: Bool,
    ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del made_jump
        h = t1 - t0
        γ, u, f = args
        h_s, gamma_s, u_s = solver_state['h'], solver_state['gamma'], solver_state['u']
        cond = jnp.all(jnp.array([h == h_s, γ == gamma_s, u == u_s]))
        solver_state = lax.cond(cond, self.init(terms, t0, t1, y0, args), solver_state)

        drift, diffusion = terms.terms
        levy: LevyVal = diffusion.levy_contr(t0, t1)
        w = levy.W
        hh = levy.H

        # TODO: add body

        y1 = y0
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, solver_state, RESULTS.successful

    def func(
            self,
            terms: AbstractTerm,
            t0: Scalar,
            y0: PyTree,
            args: PyTree,
    ) -> PyTree:
        return terms.vf(t0, y0, args)
