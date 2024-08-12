from collections.abc import Callable
from typing import Optional, TYPE_CHECKING, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Real

from .._custom_types import (
    Args,
    BoolScalarLike,
    IntScalarLike,
    RealScalarLike,
    VF,
    Y,
)
from .._misc import static_select, upcast_or_raise
from .._solution import RESULTS
from .._term import AbstractTerm
from .adaptive import _none_or_array
from .base import AbstractStepSizeController


_ControllerState = TypeVar("_ControllerState")
_Dt0 = TypeVar("_Dt0", None, RealScalarLike, Optional[RealScalarLike])


class JumpStepWrapper(
    AbstractStepSizeController[
        tuple[BoolScalarLike, RealScalarLike, _ControllerState], _Dt0
    ]
):
    """Wraps an existing step controller and adds the ability to specify `step_ts`
    and `jump_ts`. The former are times to which the controller should step and the
    latter are times at which the vector field has a discontinuity (jump)."""

    controller: AbstractStepSizeController[_ControllerState, _Dt0]
    step_ts: Optional[Real[Array, " steps"]] = eqx.field(
        default=None, converter=_none_or_array
    )
    jump_ts: Optional[Real[Array, " jumps"]] = eqx.field(
        default=None, converter=_none_or_array
    )

    def __check_init__(self):
        if self.jump_ts is not None and not jnp.issubdtype(
            self.jump_ts.dtype, jnp.inexact
        ):
            raise ValueError(
                f"jump_ts must be floating point, not {self.jump_ts.dtype}"
            )

    def wrap(self, direction: IntScalarLike):
        step_ts = None if self.step_ts is None else self.step_ts * direction
        jump_ts = None if self.jump_ts is None else self.jump_ts * direction
        return eqx.tree_at(
            lambda s: (s.step_ts, s.jump_ts),
            self,
            (step_ts, jump_ts),
            is_leaf=lambda x: x is None,
        )

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        dt0: _Dt0,
        args: Args,
        func: Callable[[PyTree[AbstractTerm], RealScalarLike, Y, Args], VF],
        error_order: Optional[RealScalarLike],
    ) -> tuple[RealScalarLike, tuple[BoolScalarLike, RealScalarLike, _ControllerState]]:
        t1, inner_state = self.controller.init(
            terms, t0, t1, y0, dt0, args, func, error_order
        )
        dt_proposal = t1 - t0

        t1 = self._clip_step_ts(t0, t1)
        t1, jump_next_step = self._clip_jump_ts(t0, t1)

        state = (jump_next_step, dt_proposal, inner_state)

        return t1, state

    def adapt_step_size(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        y1_candidate: Y,
        args: Args,
        y_error: Optional[Y],
        error_order: RealScalarLike,
        controller_state: tuple[BoolScalarLike, RealScalarLike, _ControllerState],
    ) -> tuple[
        BoolScalarLike,
        RealScalarLike,
        RealScalarLike,
        BoolScalarLike,
        tuple[BoolScalarLike, RealScalarLike, _ControllerState],
        RESULTS,
    ]:
        made_jump, prev_dt, inner_state = controller_state
        eqx.error_if(prev_dt, prev_dt < t1 - t0, "prev_dt must be >= t1-t0")

        (
            keep_step,
            next_t0,
            next_t1,
            _,
            inner_state,
            result,
        ) = self.controller.adapt_step_size(
            t0, t1, y0, y1_candidate, args, y_error, error_order, inner_state
        )

        dt_proposal = next_t1 - next_t0
        dt_proposal = jnp.where(
            keep_step, jnp.maximum(dt_proposal, prev_dt), dt_proposal
        )
        new_prev_dt = dt_proposal

        # If the step was kept and a jump was made, then we need to set
        # `next_t0 = nextafter(nextafter(next_t0))` to ensure that we really skip
        # over the jump and don't evaluate the vector field at the discontinuity.
        if jnp.issubdtype(jnp.result_type(t1), jnp.inexact):
            # Two nextafters. If made_jump then t1 = prevbefore(jump location)
            # so now _t1 = nextafter(jump location)
            # This is important because we don't know whether or not the jump is as a
            # result of a left- or right-discontinuity, so we have to skip the jump
            # location altogether.
            jump_keep = made_jump & keep_step
            next_t0 = static_select(
                jump_keep, eqxi.nextafter(eqxi.nextafter(next_t0)), next_t0
            )

        if TYPE_CHECKING:
            assert isinstance(
                next_t0, RealScalarLike
            ), f"type(next_t0) = {type(next_t0)}"
        next_t1 = next_t0 + dt_proposal

        # Clip the step to the next element of jump_ts or step_ts.
        next_t1 = self._clip_step_ts(next_t0, next_t1)
        next_t1, jump_next_step = self._clip_jump_ts(next_t0, next_t1)

        state = (jump_next_step, new_prev_dt, inner_state)

        return keep_step, next_t0, next_t1, made_jump, state, result

    def _clip_step_ts(self, t0: RealScalarLike, t1: RealScalarLike) -> RealScalarLike:
        if self.step_ts is None:
            return t1

        step_ts0 = upcast_or_raise(
            self.step_ts,
            t0,
            "`PIDController.step_ts`",
            "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
        )
        step_ts1 = upcast_or_raise(
            self.step_ts,
            t1,
            "`PIDController.step_ts`",
            "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
        )
        # TODO: it should be possible to switch this O(nlogn) for just O(n) by keeping
        # track of where we were last, and using that as a hint for the next search.
        t0_index = jnp.searchsorted(step_ts0, t0, side="right")
        t1_index = jnp.searchsorted(step_ts1, t1, side="right")
        # This minimum may or may not actually be necessary. The left branch is taken
        # iff t0_index < t1_index <= len(self.step_ts), so all valid t0_index s must
        # already satisfy the minimum.
        # However, that branch is actually executed unconditionally and then where'd,
        # so we clamp it just to be sure we're not hitting undefined behaviour.
        t1 = jnp.where(
            t0_index < t1_index,
            step_ts1[jnp.minimum(t0_index, len(self.step_ts) - 1)],
            t1,
        )
        return t1

    def _clip_jump_ts(
        self, t0: RealScalarLike, t1: RealScalarLike
    ) -> tuple[RealScalarLike, BoolScalarLike]:
        if self.jump_ts is None:
            return t1, False
        assert jnp.issubdtype(self.jump_ts.dtype, jnp.inexact)
        if not jnp.issubdtype(jnp.result_type(t0), jnp.inexact):
            raise ValueError(
                "`t0`, `t1`, `dt0` must be floating point when specifying `jump_ts`. "
                f"Got {jnp.result_type(t0)}."
            )
        if not jnp.issubdtype(jnp.result_type(t1), jnp.inexact):
            raise ValueError(
                "`t0`, `t1`, `dt0` must be floating point when specifying `jump_ts`. "
                f"Got {jnp.result_type(t1)}."
            )
        jump_ts0 = upcast_or_raise(
            self.jump_ts,
            t0,
            "`PIDController.jump_ts`",
            "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
        )
        jump_ts1 = upcast_or_raise(
            self.jump_ts,
            t1,
            "`PIDController.jump_ts`",
            "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
        )
        t0_index = jnp.searchsorted(jump_ts0, t0, side="right")
        t1_index = jnp.searchsorted(jump_ts1, t1, side="right")
        next_made_jump = t0_index < t1_index
        t1 = jnp.where(
            next_made_jump,
            eqxi.prevbefore(jump_ts1[jnp.minimum(t0_index, len(self.jump_ts) - 1)]),
            t1,
        )
        return t1, next_made_jump


JumpStepWrapper.__init__.__doc__ = r"""**Arguments**:

- `controller`: The controller to wrap.
- `step_ts`: Denotes extra times that must be stepped to.
- `jump_ts`: Denotes extra times that must be stepped to, and at which the vector field
    has a known discontinuity. (This is used to force FSAL solvers so re-evaluate the
    vector field.)

"""
