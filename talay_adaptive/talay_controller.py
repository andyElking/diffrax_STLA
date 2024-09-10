from collections.abc import Callable
from typing import Any, Optional, TypeAlias

import jax.numpy as jnp
from diffrax import AbstractStepSizeController
from diffrax._custom_types import (
    Args,
    BoolScalarLike,
    IntScalarLike,
    RealScalarLike,
    VF,
    Y,
)
from diffrax._solution import RESULTS
from diffrax._term import AbstractTerm
from jaxtyping import PyTree
from talay import compute_next_dt, vf_derivatives


_ControllerState: TypeAlias = None


class TalayController(AbstractStepSizeController[_ControllerState, Any]):
    ctol: RealScalarLike
    dtmin: RealScalarLike
    dtmax: RealScalarLike

    def __init__(self, ctol, dtmin, dtmax):
        self.ctol = ctol
        self.dtmin = dtmin
        self.dtmax = dtmax

    def wrap(self, direction: IntScalarLike):
        return self

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        dt0: Any,
        args: Args,
        func: Callable[[PyTree[AbstractTerm], RealScalarLike, Y, Args], VF],
        error_order: Optional[RealScalarLike],
    ) -> tuple[RealScalarLike, _ControllerState]:
        del dt0, func, error_order
        drift, diffusion = terms.terms
        w = diffusion.contr(t0, t1, use_levy=False)
        d = jnp.shape(w)[0]
        f_y, g_y, f_prime_f, g_prime_f, f_prime_g, g_prime_g, _ = vf_derivatives(
            drift, diffusion, t0, y0, args, d
        )
        dt_proposal = compute_next_dt(f_prime_g, g_prime_f, g_prime_g, self.ctol)
        dt_proposal = jnp.clip(dt_proposal, self.dtmin, self.dtmax)
        return dt_proposal, None

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
        del y0, y1_candidate, args, error_order, controller_state
        # Given that this is specially designed to work with the Talay solver,
        # the dt proposal is computed by the solver itself and passed to the
        # controller via the y_error argument.
        dt = jnp.clip(y_error, self.dtmin, self.dtmax)

        return True, t1, t1 + dt, False, None, RESULTS.successful
