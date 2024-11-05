from functools import partial
from typing import Any, Optional
from typing_extensions import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from diffrax import AbstractItoSolver
from diffrax._custom_types import (
    AbstractBrownianIncrement,
    BoolScalarLike,
    DenseInfo,
    RealScalarLike,
    VF,
)
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax._solution import RESULTS
from diffrax._term import AbstractTerm, MultiTerm
from jaxtyping import Array, PyTree


_ErrorEstimate: TypeAlias = Optional[Array]
_SolverState: TypeAlias = Array
_CarryType: TypeAlias = tuple[Array, Array, Array, Array]
_term_structure: TypeAlias = MultiTerm[
    tuple[
        AbstractTerm[Any, RealScalarLike],
        AbstractTerm[Any, AbstractBrownianIncrement],
    ]
]


class W2ItoFoster(AbstractItoSolver[_SolverState]):
    term_structure = _term_structure
    interpolation_cls = LocalLinearInterpolation
    term_compatible_contr_kwargs = (dict(), dict(use_levy=True))
    key: Array
    error_mode: int = eqx.field(static=True)
    minimal_levy_area = AbstractBrownianIncrement

    def __init__(self, key, error_mode: bool = False):
        self.key = key
        self.error_mode = error_mode

    def init(
        self,
        terms: _term_structure,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Array,
        args: PyTree,
    ) -> _SolverState:
        return self.key

    def step(
        self,
        terms: _term_structure,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Array,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Array, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del made_jump

        dtype = jnp.result_type(*jtu.tree_leaves(y0))
        drift, diffusion = terms.terms

        # time increment
        h = drift.contr(t0, t1)
        bm_inc: AbstractBrownianIncrement = diffusion.contr(t0, t1, use_levy=True)
        w = jnp.asarray(bm_inc.W, dtype=dtype)
        if w.ndim == 0:
            w = w[None]
        assert w.ndim == 1
        d = w.shape[0]
        n = y0.shape[0]
        assert y0.ndim == 1
        assert w.ndim == 1

        @jax.jit
        def g_diag(_t, _y):
            assert _y.shape == (n, d)
            vec_g = jax.vmap(diffusion.vf, in_axes=(None, 1, None), out_axes=2)
            g_y_full = vec_g(_t, _y, args)
            assert g_y_full.shape == (
                n,
                d,
                d,
            ), f"Expected {(n, d, d)}, got {g_y_full.shape}"
            return jnp.diagonal(g_y_full, axis1=1, axis2=2)

        # Compute the weak random variables
        # Split the key
        state_key, rad_key = jr.split(solver_state, 2)
        eta1, eta2 = jr.rademacher(rad_key, shape=(2,))
        xi = jnp.sqrt(h) * eta2

        triu_indices = jnp.triu_indices(d, 1)
        triu = jnp.zeros((d, d), dtype=dtype)
        triu = triu.at[triu_indices].set(1)
        eta1_triu = eta1 * triu
        one_plusminus_eta1 = jnp.ones((d, d), dtype=dtype) + eta1_triu - eta1_triu.T

        def comp_update(_y0, _h, _w):
            # make a matrix that has ones in upper triangle
            ii = 0.5 * _w[:, None] * one_plusminus_eta1
            ii = ii.at[jnp.diag_indices(d)].set(0)
            ii_diag = (1 / (2 * xi)) * (_w**2 - _h)
            f0_h = drift.vf_prod(t0, _y0, args, _h)
            y_half = _y0 + 1 / 2 * f0_h
            g = partial(diffusion.vf, args=args)
            g_y_half = g(t0, y_half)
            assert g_y_half.shape == (n, d)
            # I denote y^tilde by z
            z_1 = _y0 + f0_h + jnp.tensordot(g_y_half, _w, axes=1)
            f1_h = drift.vf_prod(t0, z_1, args, _h)
            half_xi_g_y_half = 1 / 2 * xi * jnp.sum(g_y_half, axis=1)
            z_half = y_half - 1 / 2 * half_xi_g_y_half
            # zs half have an extra trailing dim of d
            zs_half = (y_half + 1 / 2 * half_xi_g_y_half)[:, None] + jnp.tensordot(
                g_y_half, ii, axes=1
            )
            g_z_half = g(t0, z_half)
            gz_minus_gy = g_z_half - g_y_half
            gs_sum = g_diag(t0, zs_half) + gz_minus_gy

            update = (
                1 / 2 * (f0_h + f1_h)
                + jnp.tensordot(gs_sum, _w, axes=1)
                + 2 * jnp.tensordot(-gz_minus_gy, ii_diag, axes=1)
            )
            return update, ii, y_half, g_y_half, f0_h, gz_minus_gy

        update, ii, y_half, g_y_half, f0_h, gz_minus_gy = comp_update(y0, h, w)

        y1 = y0 + update

        if self.error_mode == 1:
            del ii, y_half, g_y_half, f0_h, gz_minus_gy
            update_backward, _, _, _, _, _ = comp_update(y1, -h, -w)
            error = update + update_backward

        elif self.error_mode in (2, 3):
            # instead of ii we use jj, which is the same as ii but with the diagonal
            # entries equal to w
            jj = jnp.where(jnp.eye(d, dtype=bool), w, ii)
            assert isinstance(jj, jnp.ndarray)

            if self.error_mode == 2:
                zs_1 = y_half[:, None] + jnp.tensordot(g_y_half, jj, axes=1)
                gs_sum_err = 0.5 * (g_diag(t1, zs_1) + g_y_half)

            else:  # self.error_mode == 3
                zs_1_plus_y_half = 2 * y_half[:, None] + jnp.tensordot(
                    g_y_half, jj, axes=1
                )
                gs_sum_err = g_diag(t1, 0.5 * zs_1_plus_y_half)

            alt_update = (
                +f0_h
                + jnp.tensordot(gs_sum_err, w, axes=1)
                + h / xi * jnp.sum(gz_minus_gy, axis=1)
            )
            error = update - alt_update

        else:
            error = None

        dense_info = dict(y0=y0, y1=y1)
        return y1, error, dense_info, state_key, RESULTS.successful

    def func(
        self,
        terms: _term_structure,
        t0: RealScalarLike,
        y0: Array,
        args: PyTree,
    ) -> VF:
        return terms.vf(t0, y0, args)
