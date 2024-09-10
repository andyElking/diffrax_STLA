from collections.abc import Callable
from typing import Any, ClassVar, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import (
    AbstractItoSolver,
    AbstractTerm,
    LocalLinearInterpolation,
    MultiTerm,
    RESULTS,
)
from diffrax._brownian import foster_levy_area
from diffrax._custom_types import (
    AbstractSpaceTimeTimeLevyArea,
    Args,
    BoolScalarLike,
    DenseInfo,
    RealScalarLike,
    VF,
    Y,
)
from jax import Array
from jaxtyping import PyTree


_term_structure: TypeAlias = MultiTerm[
    tuple[
        AbstractTerm[Any, RealScalarLike],
        AbstractTerm[Any, AbstractSpaceTimeTimeLevyArea],
    ]
]
_ErrorEstimate: TypeAlias = RealScalarLike
# solver state will just be a random key which we will keep splitting
_SolverState: TypeAlias = tuple[Array, Array, Array, Array, Array, Array, Array]


def vf_derivatives(drift, diffusion, t0, y0, args, d):
    n = y0.shape[0]

    # f has signature n -> n
    def f(y):
        return drift.vf(t0, y, args)

    # g has signature n -> (n, d)
    def g(y):
        return diffusion.vf(t0, y, args)

    # We use jacobian vector product to compute f'f, g'g, f'g and g'f

    f_y = f(y0)  # (n,)
    g_y = g(y0)  # (n, d)

    _, f_prime_f = jax.jvp(f, (y0,), (f_y,))  # (n,)
    _, g_prime_f = jax.jvp(g, (y0,), (f_y,))  # (n, d)

    # For the other two we need to vmap over the second dimension of tangents g(y)
    def f_prime_g_fun(g_col):
        _, f_prime_g_col = jax.jvp(f, (y0,), (g_col,))
        assert f_prime_g_col.shape == (n,)
        return f_prime_g_col

    # f'(y) has shape (n, n), g(y) has shape (n, d) so we need to vmap over
    # the second dimension. The output of f_prime_g_fun is (n,), so the output
    # of f_prime_g will be (n, d)
    f_prime_g = jax.vmap(f_prime_g_fun, in_axes=1, out_axes=1)(g_y)
    assert f_prime_g.shape == (n, d)

    def g_prime_g_fun(g_col):
        assert g_col.shape == (n,)
        _, g_prime_g_col = jax.jvp(g, (y0,), (g_col,))
        assert g_prime_g_col.shape == (n, d)
        return g_prime_g_col

    # g'(y) has shape (n, d, n), g(y) has shape (n, d) so we need to vmap over
    # the second dimension. The output of g_prime_g_fun is (n, d), so the output
    # of g_prime_g will be (n, d, d)
    g_prime_g = jax.vmap(g_prime_g_fun, in_axes=1, out_axes=1)(g_y)
    assert g_prime_g.shape == (n, d, d)

    return f_y, g_y, f_prime_f, g_prime_f, f_prime_g, g_prime_g


def compute_next_dt(f_prime_g, g_prime_f, g_prime_g, error_mult):
    err_f = jnp.sum((f_prime_g - g_prime_f) ** 2)
    err_g = jnp.sum((g_prime_g - g_prime_g.swapaxes(-1, -2)) ** 2)

    h_proposal = (-err_g + jnp.sqrt(err_g**2 + 4 * err_f * error_mult)) / (2 * err_f)
    return h_proposal


class Talay(AbstractItoSolver):
    term_structure: ClassVar = _term_structure
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation
    key: Array
    error_mult: RealScalarLike
    minimal_levy_area = AbstractSpaceTimeTimeLevyArea
    term_compatible_contr_kwargs = (dict(), dict(use_levy=True))
    use_levy_area: bool = eqx.field(static=True)

    def __init__(
        self, key: Array, error_mult: RealScalarLike = 1.0, use_levy_area: bool = True
    ):
        self.key = key
        self.use_levy_area = use_levy_area
        self.error_mult = error_mult

    def order(self, terms):
        raise ValueError("`Talay` should not be used to solve ODEs.")

    def strong_order(self, terms):
        return 1

    def init(
        self,
        terms: _term_structure,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        assert isinstance(y0, Array)
        if y0.ndim == 0:
            y0 = y0[None]
        assert y0.ndim == 1

        drift, diffusion = terms.terms
        whk: AbstractSpaceTimeTimeLevyArea = diffusion.contr(t0, t1, use_levy=True)

        d = jnp.shape(whk.W)[0]

        f_y, g_y, f_prime_f, g_prime_f, f_prime_g, g_prime_g = vf_derivatives(
            drift, diffusion, t0, y0, args, d
        )

        return self.key, f_y, g_y, f_prime_f, g_prime_f, f_prime_g, g_prime_g

    def step(
        self,
        terms: _term_structure,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del made_jump

        assert isinstance(y0, Array)
        dtype = y0.dtype
        if y0.ndim == 0:
            y0 = y0[None]
        assert y0.ndim == 1
        n = y0.shape[0]

        drift, diffusion = terms.terms
        h = drift.contr(t0, t1)

        whk: AbstractSpaceTimeTimeLevyArea = diffusion.contr(t0, t1, use_levy=True)

        def recast_bm(bm):
            bm = jnp.asarray(bm, dtype=dtype)
            if bm.ndim == 0:
                bm = bm[None]
            assert bm.ndim == 1
            return bm

        w = recast_bm(whk.W)  # (d,)
        hh = recast_bm(whk.H)  # (d,)
        kk = recast_bm(whk.K)  # (d,)
        d = w.shape[0]

        key, f_y, g_y, f_prime_f, g_prime_f, f_prime_g, g_prime_g = solver_state

        if self.use_levy_area:
            # Compute space-space Levy area using Foster's approximation
            # Split the key
            state_key, levy_key = jr.split(key, 2)
            # space-space levy area is an anti-symmetric d*d matrix,
            # but the function returns the flattened upper triangle
            triu_indices = jnp.triu_indices(d, 1)
            la_flat = foster_levy_area(levy_key, triu_indices, w, hh, kk, dt=h)
            # Unflatten the upper triangle
            la = jnp.zeros((d, d), dtype=dtype)
            la = la.at[triu_indices].set(la_flat)
        else:
            state_key = key
            la = 0

        int_w_dw = (
            0.5 * (w[None, :] * w[:, None]) - (h / 2) * jnp.eye(d, dtype=dtype) + la
        )  # (d, d)
        assert int_w_dw.shape == (d, d)

        # Compute the Talay step

        gg_int_w = jnp.tensordot(g_prime_g, int_w_dw, axes=2)  # (n,)
        assert gg_int_w.shape == (n,)

        y1 = (
            y0
            + f_y * h
            + g_y @ w
            + gg_int_w
            + f_prime_g @ (h * (w / 2 + hh))
            + g_prime_f @ (h * (w / 2 - hh))
            + f_prime_f * h**2 / 2
        )
        assert y1.shape == (n,)

        dense_info = dict(y0=y0, y1=y1)

        # Now compute the vector fields and Lie brackets for next step.
        # These are then also used to compute the error estimate.
        f_y, g_y, f_prime_f, g_prime_f, f_prime_g, g_prime_g = vf_derivatives(
            drift, diffusion, t1, y1, args, d
        )

        # Compute errors for the next step
        h_proposal = compute_next_dt(f_prime_g, g_prime_f, g_prime_g, self.error_mult)

        solver_state = state_key, f_y, g_y, f_prime_f, g_prime_f, f_prime_g, g_prime_g

        return y1, h_proposal, dense_info, solver_state, RESULTS.successful

    def func(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)
