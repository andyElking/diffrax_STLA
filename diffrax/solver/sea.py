from typing import Tuple
import jax.tree_util as jtu
from equinox.internal import ω

from ..custom_types import Bool, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..solution import RESULTS
from ..term import AbstractTerm, LevyVal
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
            terms: AbstractTerm,
            t0: Scalar,
            t1: Scalar,
            y0: PyTree,
            args: PyTree,
    ) -> _SolverState:
        return None

    def step(
            self,
            terms: AbstractTerm,
            t0: Scalar,
            t1: Scalar,
            y0: PyTree,
            args: PyTree,
            solver_state: _SolverState,
            made_jump: Bool,
    ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        levy_contr = terms.levy_contr(t0, t1)
        is_levy_val = lambda l: isinstance(l, LevyVal)

        def filt_contr(ctr):
            return 0.5 * ctr.W + ctr.H if isinstance(ctr, LevyVal) else 0
        filtered_contr = jtu.tree_map(filt_contr, levy_contr, is_leaf=is_levy_val)
        y_tilde = (y0 ** ω + (terms.vf_prod(t0, y0, args, filtered_contr)) ** ω).ω

        def w_and_t(ctr):
            return ctr.W if isinstance(ctr, LevyVal) else ctr
        w_and_t_contr = jtu.tree_map(w_and_t, levy_contr, is_leaf=is_levy_val)

        y1 = (y0**ω + (terms.vf_prod(t0, y_tilde, args, w_and_t_contr))**ω).ω
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
