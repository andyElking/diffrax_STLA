from typing import Tuple
import jax.tree_util as jtu
from equinox.internal import ω

from ..custom_types import Bool, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..solution import RESULTS
from ..term import AbstractTerm, STLAMultiTerm
from .base import AbstractItoSolver

_ErrorEstimate = None
_SolverState = None


def _sum(*x):
    return sum(x[1:], x[0])


class ShiftedEuler(AbstractItoSolver):
    """Shifted Euler's method for SDEs. When applied to SDEs with
     additive noise, it has a local error of O(h^2) compared to
     standard Euler-Maruyama, which has O(h^1.5).
    """

    term_structure = AbstractTerm
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.5

    def init(
            self,
            terms: STLAMultiTerm,
            t0: Scalar,
            t1: Scalar,
            y0: PyTree,
            args: PyTree,
    ) -> _SolverState:
        return None

    def step(
            self,
            terms: STLAMultiTerm,
            t0: Scalar,
            t1: Scalar,
            y0: PyTree,
            args: PyTree,
            solver_state: _SolverState,
            made_jump: Bool,
    ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        # control, stla = terms.stla_contr(t0, t1)
        h = t1 - t0
        w_term = terms.stla_term
        t_terms = terms.non_stla_terms
        w, hh = w_term.stla_contr(t0, t1)
        y_tilde1 = (y0**ω + (w_term.vf_prod(t0, y0, args, hh))**ω).ω
        # y_tilde2 = (y_tilde1**ω + ((5/6) * t_terms.vf_prod(t0, y_tilde1, args, h))**ω + (w_term.vf_prod(t0, y0, args, w))**ω).ω
        # y1 = (y0**ω + (t_terms.vf_prod(t0, y_tilde, args, t1 - t0))**ω + (w_term.vf_prod(t0, y0, args, w))**ω).ω
        y1 = y0
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
