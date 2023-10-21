from dataclasses import dataclass, field
from typing import Tuple, Optional

import jax
import jax.tree_util as jtu
import numpy as np
import jax.numpy as jnp
import jax.lax as lax
import equinox.internal as eqxi
from equinox.internal import ω

from ..custom_types import Bool, DenseInfo, PyTree, Scalar, LevyVal
from ..local_interpolation import LocalLinearInterpolation
from ..solution import RESULTS
from ..term import AbstractTerm, MultiTerm, ODETerm, ControlTerm
from .base import AbstractItoSolver

_ErrorEstimate = None
_SolverState = None


def linear_combination(a: jax.Array):
    def lin_comb(*k_leaf):
        return jnp.vdot(a, jnp.array([*k_leaf]))

    def fun(ks: list[PyTree]) -> PyTree:
        return jtu.tree_map(lin_comb, *ks)
    return fun


@dataclass(frozen=True)
class StochasticButcherTableau:
    # Only supports explicit SRK so far
    c: jax.Array
    b: jax.Array
    a: list[jax.Array]

    # coefficients for W and H (of shape (len(c)+1,)
    cw: jax.Array
    ch: jax.Array
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

        # FSAL checks TBA


class ANSR(AbstractItoSolver):
    """Additive-Noise Stochastic Runge-Kutta method.
    Takes in Butcher Tableau (ish). Description TBA.
    """

    term_structure = MultiTerm[Tuple[ODETerm, ControlTerm]]
    interpolation_cls = LocalLinearInterpolation
    tableau: StochasticButcherTableau

    def __init__(self, tab):
        self.tableau = tab

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
        return None

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
        del solver_state, made_jump

        h = t1 - t0
        drift, diffusion = terms.terms
        levy: LevyVal = diffusion.levy_contr(t0, t1)
        w = levy.W
        hh = levy.H
        sigma = diffusion.vf(t0, y0, args)

        def stage(ks: list[PyTree], x: tuple[jax.Array, Scalar, Scalar, Scalar]):
            # carry = [k1, k2, ..., k_{j-1}] where ki are already multiplied by h, i.e.
            # ki = drift.vf_prod(t0 + c_i*h, y_i, args, h)
            a_j, c_j, cw_j, ch_j = x
            diffusion_contr = cw_j * w + ch_j * hh
            mult_with_a_j = linear_combination(a_j)

            y_j = (y0 ** ω + (diffusion.prod(sigma, diffusion_contr)) ** ω).ω
            y_j = lax.cond(len(ks) > 0,
                           lambda y_: (y_ ** ω + (mult_with_a_j(ks)) ** ω).ω,
                           lambda y_: y_,
                           y_j)

            k_j = drift.vf_prod(t0 + c_j * h, y_j, args, h)
            ks.append(k_j)
            # note that carry will already contain the whole stack of k_js, so no need for second return value
            return ks, None

        a = [jnp.array([], dtype=self.tableau.a[0].dtype)] + list(self.tableau.a)
        c = jnp.insert(self.tableau.c, 0, 0.0)
        b, cw, ch = self.tableau.b, self.tableau.cw, self.tableau.ch
        ks, _ = lax.scan(stage, [], (a, c ,cw, ch), length=len(b))

        mult_with_b = linear_combination(b)
        diffusion_contr = self.tableau.cw_last * w + self.tableau.ch_last * hh
        y1 = (y0 ** ω + (mult_with_b(ks)) ** ω + (diffusion.prod(sigma, diffusion_contr)) ** ω).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
            self,
            terms: AbstractTerm,
            t0: Scalar,
            y0: PyTree,
            args: PyTree,
    ) -> PyTree:
        return terms.vf(t0, y0, args)
