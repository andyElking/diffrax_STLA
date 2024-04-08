from typing import Callable, Optional, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree, Real

from .._custom_types import Args, BoolScalarLike, IntScalarLike, RealScalarLike, VF, Y
from .._misc import upcast_or_raise
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractStepSizeController


_ControllerState: TypeAlias = None


def _none_or_array(x):
    if x is None:
        return None
    else:
        return jnp.asarray(x)


class SABRController(AbstractStepSizeController[None, Optional[RealScalarLike]]):
    """Step size controller for the CIR process."""

    ctol: RealScalarLike
    dtmax: RealScalarLike
    dtmin: RealScalarLike
    step_ts: Optional[Real[Array, " steps"]] = eqx.field(
        default=None, converter=_none_or_array
    )

    def wrap(self, direction: IntScalarLike) -> "AbstractStepSizeController":
        return self

    def desired_step_size(self, v_max):
        step_size = jnp.log(1 + self.ctol * jnp.exp(-2 * v_max))
        step_size = jnp.nan_to_num(
            step_size, nan=self.dtmin, posinf=self.dtmax, neginf=self.dtmin
        )
        return jnp.clip(step_size, self.dtmin, self.dtmax)

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

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        dt0: Optional[RealScalarLike],
        args: Args,
        func: Callable[[PyTree[AbstractTerm], RealScalarLike, Y, Args], VF],
        error_order: Optional[RealScalarLike],
    ) -> tuple[RealScalarLike, None]:
        del terms, t1, dt0, args, func, error_order
        assert y0.shape == (2,)
        step_size = self.desired_step_size(y0[1])
        t1 = self._clip_step_ts(t0, t0 + step_size)
        return t1, None

    def adapt_step_size(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        y1_candidate: Y,
        args: Args,
        y_error: Optional[Y],
        error_order: RealScalarLike,
        controller_state: _ControllerState,
    ) -> tuple[
        BoolScalarLike,
        RealScalarLike,
        RealScalarLike,
        BoolScalarLike,
        _ControllerState,
        RESULTS,
    ]:
        del args, y_error, error_order, controller_state
        assert y0.shape == (2,)
        v0 = y0[1]
        v1 = y1_candidate[1]
        v_max = jnp.maximum(v0, v1)
        desired = self.desired_step_size(v_max)
        accepted_desired = self.desired_step_size(v1)

        accept = t1 - t0 < 1.1 * desired
        new_t0 = jnp.where(accept, t1, t0)
        new_dt = jnp.where(accept, accepted_desired, desired)
        new_dt = jnp.clip(new_dt, self.dtmin, self.dtmax)
        new_t1 = self._clip_step_ts(new_t0, new_t0 + new_dt)
        return accept, new_t0, new_t1, False, None, RESULTS.successful
