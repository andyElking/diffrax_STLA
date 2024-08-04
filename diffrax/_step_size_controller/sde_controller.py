# import typing
# from typing import Callable, TYPE_CHECKING, Optional, TypeAlias
# import optax as optx
# import equinox as eqx
# import lineax.internal as lxi
# import jax.numpy as jnp
# import jax.tree_util as jtu
# from jax import lax
# from jaxtyping import PyTree, Bool, Array
# from lineax.internal import complex_to_real_dtype
#
# from .._term import AbstractTerm
# from .._custom_types import RealScalarLike, Args, Y, IntScalarLike, VF
#
# from .adaptive import AbstractAdaptiveStepSizeController, _select_initial_step
# from .._misc import upcast_or_raise
#
# if TYPE_CHECKING:
#     rms_norm = optx.rms_norm
# else:
#     # We can't use `optx.rms_norm` itself as a default attribute value. This is
#     # because it is a callable, and then the doc stack thinks that it is a method.
#     if getattr(typing, "GENERATING_DOCUMENTATION", False):
#
#         class _RmsNorm:
#             def __repr__(self):
#                 return "<function rms_norm>"
#
#         old_rms_norm = optx.rms_norm
#         rms_norm = _RmsNorm()
#     else:
#         rms_norm = optx.rms_norm
#
#
# _State: TypeAlias = tuple[RealScalarLike, list[bool]]
#
#
# class SDEController(AbstractAdaptiveStepSizeController):
#     """A step-size controller that either halves or doubles the step size.
#     Normally it takes a step of size `dt0`, but is it is unsuccessful, it will
#     descend to a lower level and instead take two steps of size `dt0 / 2`. Each
#     of these steps can be further subdivided until the desired accuracy is
#     achieved, or until the maximum depth is reached.
#     """
#
#     rtol: RealScalarLike
#     atol: RealScalarLike
#     norm: Callable[[PyTree], RealScalarLike] = rms_norm
#     depth: int = eqx.field(static=True, default=2)
#
#     def init(
#         self,
#         terms: PyTree[AbstractTerm],
#         t0: RealScalarLike,
#         t1: RealScalarLike,
#         y0: Y,
#         dt0: RealScalarLike,
#         args: Args,
#         func: Callable[[PyTree[AbstractTerm], RealScalarLike, Y, Args], VF],
#         error_order: Optional[RealScalarLike],
#     ) -> tuple[RealScalarLike, _State]:
#         del t1, terms, y0, args, func, error_order
#
#         t1 = self._clip_step_ts(t0, t0 + dt0)
#
#         # The level array is used to keep track of which levels have been visited.
#         # If  we did a single step at level 3, say, then we know that we must later
#         # make another step at that level.
#         level_stack = []
#         return t1, (dt0, level_stack)
#
#     def adapt_step_size(
#         self,
#         t0: RealScalarLike,
#         t1: RealScalarLike,
#         y0: Y,
#         y1_candidate: Y,
#         args: Args,
#         y_error: Y,
#         error_order: RealScalarLike,
#         controller_state: _State,
#     ):
#         del args
#         if y_error is None and y0 is not None:
#             # y0 is not None check is included to handle the edge case that the state
#             # is just a trivial `None` PyTree. In this case `y_error` has the same
#             # PyTree structure and thus overlaps with our special usage of `None` to
#             # indicate a lack of error estimate.
#             raise RuntimeError(
#                 "Cannot use adaptive step sizes with a solver that does not provide "
#                 "error estimates."
#             )
#
#         dt0, level_arr = controller_state
#         # Current level is the index of the last True in level_arr.
#         current_level = jnp.nonzero(level_arr)[0][-1]
#         prev_dt = t1 - t0
#         assert jnp.isclose(prev_dt, dt0 * 2 ** current_level)
#
#     def _clip_step_ts(self, t0: RealScalarLike, t1: RealScalarLike) -> RealScalarLike:
#         if self.step_ts is None:
#             return t1
#
#         step_ts0 = upcast_or_raise(
#             self.step_ts,
#             t0,
#             "`PIDController.step_ts`",
#             "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
#         )
#         step_ts1 = upcast_or_raise(
#             self.step_ts,
#             t1,
#             "`PIDController.step_ts`",
#             "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
#         )
#         # TODO: it should be possible to switch this O(nlogn) for just O(n) by keeping
#         # track of where we were last, and using that as a hint for the next search.
#         t0_index = jnp.searchsorted(step_ts0, t0, side="right")
#         t1_index = jnp.searchsorted(step_ts1, t1, side="right")
#         # This minimum may or may not actually be necessary. The left branch is taken
#         # iff t0_index < t1_index <= len(self.step_ts), so all valid t0_index s must
#         # already satisfy the minimum.
#         # However, that branch is actually executed unconditionally and then where'd,
#         # so we clamp it just to be sure we're not hitting undefined behaviour.
#         t1 = jnp.where(
#             t0_index < t1_index,
#             step_ts1[jnp.minimum(t0_index, len(self.step_ts) - 1)],
#             t1,
#         )
#         return t1
#
#
#
# def adapt(
#     dtmax,
#     t0,
#     t1,
#     error,
#     tol,
#     level_stack: Array[bool],
#     current_level: int
# ):
#     # The current level tells the depth of the binary tree.
#     # The root represents a step of size dmax.
#     assert t1 - t0 == dtmax * 2 ** current_level
#
#     # We can recover our position in the binary tree by looking at the level_stack:
#     # A True means left child, a False means right child.
#
#     # We accept the step if the error is less than the tolerance
#     # or if we are at the lowest level.
#     accept = (error < tol) or (current_level == len(level_stack) - 1)
#
#     if accept:
#         # We move one node in the binary tree to the right.
#         # This means that we find the first ancestor which is a left child,
#         # and we move to its right sibling by changing the value of the level_stack
#         # at that position to False.
#
#         while not level_stack[current_level]:
#             current_level -= 1
#         level_stack[current_level] = False
#
#         # We never go above the root, that stays True.
#         level_stack[0] = True
#
#         # Since we just moved to a right sibling on the current level,
#         # we need to make a step at that level.
#         dt_next = dtmax * 2 ** current_level
#         t0_next = t1
#         t1_next = t1 + dt_next
#
#         return accept, t0_next, t1_next, current_level, level_stack
#
#     else:  # accept == False
#         # If we reject the step, we go one level down the stack,
#         # and set that level to True, to remember we need to make two steps there.
#         current_level += 1
#         level_stack[current_level] = True
#
#         dt_next = dtmax * 2 ** current_level
#         t0_next = t0
#         t1_next = t0_next + dt_next
#         return accept, t0_next, t1_next, current_level, level_stack
