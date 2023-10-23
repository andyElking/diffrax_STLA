from typing import Tuple, Optional
import jax.tree_util as jtu
import jax.numpy as jnp
from equinox.internal import ω

from ..custom_types import Bool, DenseInfo, PyTree, Scalar, LevyVal
from ..local_interpolation import LocalLinearInterpolation
from ..solution import RESULTS
from ..term import AbstractTerm, MultiTerm, ODETerm
from .base import AbstractItoSolver
from .ansr import ANSR, StochasticButcherTableau

# class ShARK(AbstractItoSolver):
#     """Shifted Additive-noise Runge-Kutta method for SDEs.
#     When applied to SDEs with additive noise, it converges
#     strongly with order 1.5.
#     """
#
#     term_structure = AbstractTerm
#     interpolation_cls = LocalLinearInterpolation
#
#     def order(self, terms):
#         return 2
#
#     def strong_order(self, terms):
#         return 1.5
#
#     # def error_order(self, terms: PyTree[AbstractTerm]) -> Optional[Scalar]:
#     #     return 2
#
#     def init(
#             self,
#             terms: MultiTerm[Tuple[ODETerm, AbstractTerm]],
#             t0: Scalar,
#             t1: Scalar,
#             y0: PyTree,
#             args: PyTree,
#     ) -> _SolverState:
#         return None
#
#     def step(
#             self,
#             terms: MultiTerm[Tuple[ODETerm, AbstractTerm]],
#             t0: Scalar,
#             t1: Scalar,
#             y0: PyTree,
#             args: PyTree,
#             solver_state: _SolverState,
#             made_jump: Bool,
#     ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
#         del solver_state, made_jump
#         # control, stla = terms.stla_contr(t0, t1)
#         h = t1 - t0
#         drift, diffusion = terms.terms
#         levy: LevyVal = diffusion.levy_contr(t0, t1)
#         w = levy.W
#         hh = levy.H
#         sigma = diffusion.vf(t0, y0, args)
#         y_tilde1 = (y0**ω + (diffusion.prod(sigma, hh))**ω).ω
#         ode_out_1 = drift.vf_prod(t0, y_tilde1, args, h)
#         w_term_out = diffusion.prod(sigma, w)
#         y_tilde2 = (y_tilde1 ** ω + (5/6) *
#                     (ode_out_1 ** ω + w_term_out ** ω)).ω
#         ode_out_2 = drift.vf_prod(t0, y_tilde2, args, h)
#         y1 = (y0**ω + (2/5) * ode_out_1**ω + (3/5) * ode_out_2**ω + w_term_out ** ω).ω
#         dense_info = dict(y0=y0, y1=y1)
#         return y1, None, dense_info, None, RESULTS.successful
#
#     def func(
#             self,
#             terms: AbstractTerm,
#             t0: Scalar,
#             y0: PyTree,
#             args: PyTree,
#     ) -> PyTree:
#         return terms.vf(t0, y0, args)
tab = StochasticButcherTableau(
    c=jnp.array([5 / 6]),
    b=jnp.array([0.4, 0.6]),
    a=[jnp.array([5 / 6])],
    cw=jnp.array([0.0, 5 / 6]),
    ch=jnp.array([1.0, 0.0]),
    cw_last=1.0,
    ch_last=0.0
)


class ShARK(ANSR):
    tableau = tab
