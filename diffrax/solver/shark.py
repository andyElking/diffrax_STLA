from typing import Tuple, Optional
import jax.tree_util as jtu
from equinox.internal import ω

from ..custom_types import Bool, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..solution import RESULTS
from ..term import AbstractTerm, STLAMultiTerm
from .base import AbstractItoSolver

_ErrorEstimate = None
_SolverState = None


class ShARK(AbstractItoSolver):
    """Shifted Additive-noise Runge-Kutta method for SDEs.
    When applied to SDEs with additive noise, it converges
    strongly with order 1.5.
    """

    term_structure = AbstractTerm
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.5

    # def error_order(self, terms: PyTree[AbstractTerm]) -> Optional[Scalar]:
    #     return 2

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
        ode_term = terms.non_stla_terms
        bm_inc = w_term.stla_contr(t0, t1)
        w = bm_inc.W
        hh = bm_inc.H
        y_tilde1 = (y0**ω + (w_term.vf_prod(t0, y0, args, hh))**ω).ω
        ode_out_1 = ode_term.vf_prod(t0, y_tilde1, args, h)
        w_term_out = w_term.vf_prod(t0, y0, args, w)
        y_tilde2 = (y_tilde1 ** ω + (5/6) *
                    (ode_out_1 ** ω + w_term_out ** ω)).ω
        ode_out_2 = ode_term.vf_prod(t0, y_tilde2, args, h)
        y1 = (y0**ω + (2/5) * ode_out_1**ω + (3/5) * ode_out_2**ω + w_term_out ** ω).ω
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
