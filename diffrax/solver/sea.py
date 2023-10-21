from typing import Tuple
import jax.tree_util as jtu
from equinox.internal import ω

from ..custom_types import Bool, DenseInfo, PyTree, Scalar, LevyVal
from ..local_interpolation import LocalLinearInterpolation
from ..solution import RESULTS
from ..term import AbstractTerm, MultiTerm, ODETerm
from .base import AbstractItoSolver

_ErrorEstimate = None
_SolverState = None


class SEA(AbstractItoSolver):
    """Shifted Euler method for SDEs with additive noise.
     It has a local error of O(h^2) compared to
     standard Euler-Maruyama, which has O(h^1.5).
    """

    term_structure = AbstractTerm
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
            self,
            terms: MultiTerm[Tuple[ODETerm, AbstractTerm]],
            t0: Scalar,
            t1: Scalar,
            y0: PyTree,
            args: PyTree,
    ) -> _SolverState:
        return None

    def step(
            self,
            terms: MultiTerm[Tuple[ODETerm, AbstractTerm]],
            t0: Scalar,
            t1: Scalar,
            y0: PyTree,
            args: PyTree,
            solver_state: _SolverState,
            made_jump: Bool,
    ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms.terms
        levy: LevyVal = diffusion.levy_contr(t0, t1)
        w = levy.W
        hh = levy.H
        contr_tilde = 0.5 * w + hh
        sigma = diffusion.vf(t0, y0, args)
        y_tilde = (y0 ** ω + (diffusion.prod(sigma, contr_tilde)) ** ω).ω
        y1 = (y0 ** ω + (drift.vf_prod(t0, y_tilde, args, t1 - t0)) ** ω + (diffusion.prod(sigma, w)) ** ω).ω
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
