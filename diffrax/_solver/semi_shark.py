import abc
from collections.abc import Callable
from typing import Any, ClassVar
from typing_extensions import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jax import lax

from .._custom_types import (
    AbstractSpaceTimeLevyArea,
    Args,
    BoolScalarLike,
    DenseInfo,
    RealScalarLike,
    VF,
    Y,
)
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import _VF, AbstractTerm, MultiTerm, ODETerm, WrapTerm
from .base import AbstractStratonovichSolver


_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None
_FiveScalars: TypeAlias = tuple[
    RealScalarLike, RealScalarLike, RealScalarLike, RealScalarLike, RealScalarLike
]
_TwoScalars: TypeAlias = tuple[RealScalarLike, RealScalarLike]


class SemiLinearTerm(AbstractTerm[_VF, RealScalarLike]):
    f: Callable[[RealScalarLike, Y, Args], Y]
    gamma: RealScalarLike

    def vf(self, t: RealScalarLike, y: Y, args: Args):
        return self.gamma * y + self.f(t, y, args)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> RealScalarLike:
        return t1 - t0

    def prod(self, vf: _VF, control: RealScalarLike) -> Y:
        return vf * control

    def to_ode(self) -> ODETerm:
        return ODETerm(vector_field=self.vf)


_TermStructure: TypeAlias = MultiTerm[
    tuple[SemiLinearTerm, AbstractTerm[Any, AbstractSpaceTimeLevyArea]]
]


def phi_01_direct(x) -> _TwoScalars:
    exp_x = jnp.exp(x)
    expm1_x = jnp.expm1(x)
    phi1x = expm1_x / x
    return exp_x, phi1x


def phi_01_taylor(x) -> _TwoScalars:
    dtype = x.dtype
    x_powers = jnp.power(x, jnp.arange(0, 5, dtype=dtype))
    exp_x = jnp.exp(x)
    phi1_coeffs = jnp.array([1, 1 / 2, 1 / 6, 1 / 24, 1 / 120], dtype=dtype)
    phi1x = jnp.dot(phi1_coeffs, x_powers)

    return exp_x, phi1x


def coeffs_shark_direct(x) -> _FiveScalars:
    # phi0(x) = exp(x)
    exp_x = jnp.exp(x)
    expm1_x = jnp.expm1(x)
    phi1x = expm1_x / x
    phi2x = (expm1_x - x) / (x**2)
    c3_top = (exp_x + 1) * x - 2 * expm1_x
    c3x = 6 * c3_top / (x**2)
    expm1_2x = jnp.expm1(2 * x)
    c4x = -c3_top / (x * expm1_2x)
    c5x = 2 * c3x / expm1_2x
    return exp_x, phi1x, phi2x, c4x, c5x


def coeffs_shark_taylor(x) -> _FiveScalars:
    dtype = x.dtype
    # We use an order 5 Taylor expansion
    x_powers = jnp.power(x, jnp.arange(0, 5, dtype=dtype))

    # Don't need an expansion for phi0 because it doesn't have a singularity at 0
    exp_x = jnp.exp(x)

    phi1_coeffs = jnp.array([1, 1 / 2, 1 / 6, 1 / 24, 1 / 120], dtype=dtype)
    phi2_coeffs = jnp.array([1 / 2, 1 / 6, 1 / 24, 1 / 120, 1 / 720], dtype=dtype)
    # c3_coeffs = jnp.array([0, 1 / 2, 3 / 20, 1 / 30, 1 / 168], dtype=dtype)
    c4_coeffs = jnp.array([0, -1 / 12, 1 / 24, 1 / 720, -1 / 240], dtype=dtype)
    c5_coeffs = jnp.array([1, -1 / 2, -1 / 60, 1 / 20, 1 / 2520], dtype=dtype)

    phi1x = jnp.dot(phi1_coeffs, x_powers)
    phi2x = jnp.dot(phi2_coeffs, x_powers)
    # c3x = jnp.dot(c3_coeffs, x_powers)
    c4x = jnp.dot(c4_coeffs, x_powers)
    c5x = jnp.dot(c5_coeffs, x_powers)

    return exp_x, phi1x, phi2x, c4x, c5x


def coeffs_sea_direct(x) -> _FiveScalars:
    exp_x = jnp.exp(x)
    expm1_x = jnp.expm1(x)
    phi1x = expm1_x / x
    c3x = 6 * ((exp_x + 1) * x - 2 * expm1_x) / (x**2)
    c6x = (expm1_x - x) / (x * (expm1_x))
    c7x = c3x / expm1_x
    return exp_x, phi1x, c3x, c6x, c7x


def coeffs_sea_taylor(x) -> _FiveScalars:
    dtype = x.dtype
    x_powers = jnp.power(x, jnp.arange(0, 5, dtype=dtype))

    exp_x = jnp.exp(x)
    phi1_coeffs = jnp.array([1, 1 / 2, 1 / 6, 1 / 24, 1 / 120], dtype=dtype)
    c3_coeffs = jnp.array([0, 1 / 2, 3 / 20, 1 / 30, 1 / 168], dtype=dtype)
    c6_coeffs = jnp.array([1 / 2, -1 / 12, 0, 1 / 720, 0], dtype=dtype)
    c7_coeffs = jnp.array([1, 0, -1 / 60, 0, 1 / 2520], dtype=dtype)

    phi1x = jnp.dot(phi1_coeffs, x_powers)
    c3x = jnp.dot(c3_coeffs, x_powers)
    c6x = jnp.dot(c6_coeffs, x_powers)
    c7x = jnp.dot(c7_coeffs, x_powers)

    return exp_x, phi1x, c3x, c6x, c7x


def diffusion_vf_check(diffusion, t0, y0, args):
    # check that the vector field of the diffusion term is the identity
    diff_vf = diffusion.vf(t0, y0, args)
    x = jnp.ones_like(y0)
    eqx.error_if(
        y0,
        jnp.any(jnp.matmul(diff_vf, x) != x),
        "Vector field of the diffusion term should be 1.",
    )

    # check that the vector field of the diffusion term does not depend on y or t
    ones_like_y0 = jtu.tree_map(jnp.ones_like, y0)
    ones_like_t0 = jnp.ones_like(t0)
    _, t_y_sigma = eqx.filter_jvp(
        lambda t, y: diffusion.vf(t, y, args),
        (
            t0,
            y0,
        ),
        (
            ones_like_t0,
            ones_like_y0,
        ),
    )
    # check if the PyTree is just made of Nones (inside other containers)
    if len(jtu.tree_leaves(t_y_sigma)) > 0:
        raise ValueError(
            "Vector field of the diffusion term should be constant, "
            "independent of y."
        )


class AbstractSemiShARK(AbstractStratonovichSolver):
    r"""Exponential version of the Shifted Additive-noise Runge-Kutta-Three method,
     designed by James Foster.
     Only works for semi-linear additive noise SDEs of the form
    $$
        dy(t) = \gamma y(t) dt + f(t, y(t)) dt + dW(t).
    $$
    """

    term_structure: ClassVar = _TermStructure
    term_compatible_contr_kwargs = (dict(), dict(use_levy=True))
    interpolation_cls = LocalLinearInterpolation
    minimal_levy_area = AbstractSpaceTimeLevyArea
    taylor_threshold: RealScalarLike = eqx.field(static=True, default=0.01)

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.5

    def init(
        self,
        terms: _TermStructure,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        _, diffusion = terms.terms
        diffusion_vf_check(diffusion, t0, y0, args)

    def coeffs_shark(self, x) -> _FiveScalars:
        cond = x < self.taylor_threshold
        return lax.cond(cond, coeffs_shark_taylor, coeffs_shark_direct, x)

    def phi_01(self, x) -> _TwoScalars:
        cond = x < self.taylor_threshold
        return lax.cond(cond, phi_01_taylor, phi_01_direct, x)

    @abc.abstractmethod
    def inner_step(self, t0, t1, y0, w, hh, g, f, args) -> Y:
        raise NotImplementedError

    def step(
        self,
        terms: _TermStructure,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        dtype = jnp.result_type(*jtu.tree_leaves(y0))
        drift, diffusion = terms.terms
        if isinstance(drift, WrapTerm):
            drift = drift.term
        assert isinstance(drift, SemiLinearTerm)

        bm_inc = diffusion.contr(t0, t1, use_levy=True)
        assert isinstance(bm_inc, self.minimal_levy_area)
        w = jtu.tree_map(lambda x: jnp.asarray(x, dtype), bm_inc.W)
        hh = jtu.tree_map(lambda x: jnp.asarray(x, dtype), bm_inc.H)

        gamma = jnp.asarray(drift.gamma, dtype)

        y_out = self.inner_step(t0, t1, y0, w, hh, gamma, drift.f, args)

        dense_info = dict(y0=y0, y1=y_out)
        return y_out, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)


class SemiShARK(AbstractSemiShARK):
    def inner_step(self, t0, t1, y0, w, hh, g, f, args) -> Y:
        h = t1 - t0

        gh = h * g
        exp_gh, phi1_gh, phi2_gh, c4_gh, c5_gh = self.coeffs_shark(gh)

        # w_tilde = (phi1_gh * w**ω + c3_gh * hh**ω).ω
        w_tilde = ((1 + gh / 2) * w**ω + gh * hh**ω).ω
        hh_tilde = (c4_gh * w**ω + c5_gh * hh**ω).ω

        # The first stage y1 is evaluated at time t0
        y1 = (y0**ω + hh_tilde**ω).ω

        f1 = f(t0, y1, args)

        # The second stage y2 is evaluated at time t0 + alpha * h
        alpha = 5 / 6
        agh = alpha * gh
        exp_agh, phi1_agh = self.phi_01(agh)

        y2 = (
            exp_agh * y0**ω
            + h * alpha * phi1_agh * f1**ω
            + alpha * w_tilde**ω
            + exp_agh * hh_tilde**ω
            # + hh_tilde ** ω
        ).ω

        f2 = f(t0 + alpha * h, y2, args)

        y_out = (
            exp_gh * y0**ω
            + h * phi1_gh * f1**ω
            + h / alpha * phi2_gh * (f2**ω - f1**ω)
            + w_tilde**ω
        ).ω

        return y_out


class SemiSEA(AbstractStratonovichSolver):
    r"""A Shifted-Euler-Additive method for semi-linear additive noise SDEs
    designed by James Foster. Only works for SDEs of the form
    $$
        dy(t) = \gamma y(t) dt + f(t, y(t)) dt + dW(t).
    $$
    """

    term_structure: ClassVar = _TermStructure
    term_compatible_contr_kwargs = (dict(), dict(use_levy=True))
    interpolation_cls = LocalLinearInterpolation
    minimal_levy_area = AbstractSpaceTimeLevyArea
    taylor_threshold: RealScalarLike = eqx.field(static=True, default=0.1)

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 1.0

    def init(
        self,
        terms: _TermStructure,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        _, diffusion = terms.terms
        diffusion_vf_check(diffusion, t0, y0, args)

    def coeffs_sea(self, x) -> _FiveScalars:
        cond = x < self.taylor_threshold
        return lax.cond(cond, coeffs_sea_taylor, coeffs_sea_direct, x)

    def step(
        self,
        terms: _TermStructure,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        dtype = jnp.result_type(*jtu.tree_leaves(y0))
        drift, diffusion = terms.terms
        if isinstance(drift, WrapTerm):
            drift = drift.term
        assert isinstance(drift, SemiLinearTerm)

        bm_inc = diffusion.contr(t0, t1, use_levy=True)
        assert isinstance(bm_inc, self.minimal_levy_area)
        w = jtu.tree_map(lambda x: jnp.asarray(x, dtype), bm_inc.W)
        hh = jtu.tree_map(lambda x: jnp.asarray(x, dtype), bm_inc.H)

        gamma = jnp.asarray(drift.gamma, dtype)
        h = t1 - t0

        gh = h * gamma

        exp_gh, phi1_gh, c3_gh, c6_gh, c7_gh = self.coeffs_sea(gh)

        w_tilde = (phi1_gh * w**ω + c3_gh * hh**ω).ω
        hh_plus_cw_tilde = (c6_gh * w**ω + c7_gh * hh**ω).ω

        # The first stage y1 is evaluated at time t0+h/2
        y1 = (y0**ω + hh_plus_cw_tilde**ω).ω

        f1 = drift.f(t0 + h / 2, y1, args)

        y_out = (exp_gh * y0**ω + h * phi1_gh * f1**ω + w_tilde**ω).ω

        dense_info = dict(y0=y0, y1=y_out)
        return y_out, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)
