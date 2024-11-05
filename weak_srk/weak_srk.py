from typing import Any, ClassVar, Optional, TYPE_CHECKING
from typing_extensions import TypeAlias

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from diffrax._custom_types import (
    AbstractBrownianIncrement,
    BoolScalarLike,
    DenseInfo,
    IntScalarLike,
    RealScalarLike,
    VF,
)
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax._solution import RESULTS
from diffrax._solver.base import AbstractSolver
from diffrax._term import AbstractTerm, MultiTerm
from jaxtyping import Array, Float, PyTree


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar

_ErrorEstimate: TypeAlias = Optional[Array]
_SolverState: TypeAlias = Array
_CarryType: TypeAlias = tuple[Array, Array, Array, Array]
_term_structure: TypeAlias = MultiTerm[
    tuple[
        AbstractTerm[Any, RealScalarLike],
        AbstractTerm[Any, AbstractBrownianIncrement],
    ]
]


class WeakTableau(eqx.Module):
    alpha: Float[np.ndarray, " s"]
    beta0: Float[np.ndarray, " s"]
    beta1: Float[np.ndarray, " s"]

    a0: tuple[np.ndarray, ...]
    b0: tuple[np.ndarray, ...]
    a1: tuple[np.ndarray, ...]
    b1: tuple[np.ndarray, ...]
    b2: tuple[np.ndarray, ...]

    def __post_init__(self):
        assert self.alpha.ndim == 1
        s = self.alpha.shape[0]
        assert self.beta0.shape == self.beta1.shape == (s,)
        for xss in [self.a0, self.b0, self.a1, self.b1, self.b2]:
            assert len(xss) == s - 1
            assert all(xss[i].shape == (i + 1,) for i in range(s - 1))


WeakTableau.__init__.__doc__ = """The coefficients of a
[`diffrax.AbstractSRK`][] method.

See also the documentation for [`diffrax.AbstractSRK`][] for additional details on the
mathematical meaning of each of these arguments.

**Arguments:**

Let `s` denote the number of stages of the solver.

- `alpha`, `beta0`, `beta1`: np.ndarrays of shape `(s,)`.
- `a0`, `b0`, `a1`, `b1`, `b2`: Lower triangular matrices (here we assume the method
    is explicit). A tuple of np.ndarrays, corresponding to the rows of this lower
    triangle. The first array should be of shape `(1,)`. Each subsequent array should
    be of shape `(2,)`, `(3,)` etc. The final array should have shape `(s - 1,)`.
"""


class AbstractWeakSRK(AbstractSolver[_SolverState]):
    term_structure: ClassVar = _term_structure
    interpolation_cls = LocalLinearInterpolation
    term_compatible_contr_kwargs = (dict(), dict(use_levy=True))
    tableau: AbstractClassVar[WeakTableau]
    key: Array

    # Indicates the type of Lévy area used by the solver.
    # The BM must generate at least this type of Lévy area, but can generate
    # more. E.g. if the solver uses space-time Lévy area, then the BM generates
    # space-time-time Lévy area as well that is fine. The other way around would
    # not work. This is mostly an easily readable indicator so that methods know
    # what kind of BM to use.
    minimal_levy_area = AbstractBrownianIncrement

    def __init__(self, key):
        self.key = key

    def init(
        self,
        terms: _term_structure,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Array,
        args: PyTree,
    ) -> _SolverState:
        return self.key

    def _embed_a_lower(self, _a, dtype):
        num_stages = len(self.tableau.alpha)
        tab_a = np.zeros((num_stages, num_stages))
        for i, a_i in enumerate(_a):
            tab_a[i + 1, : i + 1] = a_i
        return jnp.asarray(tab_a, dtype=dtype)

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
        h = t1 - t0

        # Format the tableaus
        a0 = self._embed_a_lower(self.tableau.a0, dtype)
        a1 = self._embed_a_lower(self.tableau.a1, dtype)
        b0 = self._embed_a_lower(self.tableau.b0, dtype)
        b1 = self._embed_a_lower(self.tableau.b1, dtype)
        b2 = self._embed_a_lower(self.tableau.b2, dtype)
        alpha = jnp.asarray(self.tableau.alpha, dtype=dtype)
        beta0 = jnp.asarray(self.tableau.beta0, dtype=dtype)
        beta1 = jnp.asarray(self.tableau.beta1, dtype=dtype)

        # Brownian increment
        bm_inc: AbstractBrownianIncrement = diffusion.contr(t0, t1, use_levy=True)

        w = jnp.asarray(bm_inc.W, dtype=dtype)
        if w.ndim == 0:
            w = w[None]
        assert w.ndim == 1
        d = w.shape[0]

        # Compute the weak random variables
        # Split the key
        state_key, rad_key = jr.split(solver_state, 2)
        eta1, eta2 = jr.rademacher(rad_key, shape=(2,))
        xi = jnp.sqrt(h) * eta2
        # make a matrix that has ones in upper triangle
        triu_indices = jnp.triu_indices(d, 1)
        triu = jnp.zeros((d, d), dtype=dtype)
        triu = triu.at[triu_indices].set(1)
        eta1_triu = eta1 * triu
        one_plusminus_eta1 = (
            jnp.ones((d, d), dtype=dtype)
            - jnp.eye(d, dtype=dtype)
            + eta1_triu
            - eta1_triu.T
        )
        ii = w[:, None] * 0.5 * one_plusminus_eta1
        # the actual diagonal is used elsewhere
        ii_diag = (1 / (2 * xi)) * (w**2 - h)

        s = len(alpha)  # num stages
        n = y0.shape[0]
        assert y0.ndim == 1
        assert w.ndim == 1

        h_fs = jnp.zeros((s, n), dtype=dtype)
        w_gs = jnp.zeros((s, n), dtype=dtype)
        xi_gs = jnp.zeros((s, n, d), dtype=dtype)
        ii_gs = jnp.zeros((s, n, d), dtype=dtype)

        carry: _CarryType = (h_fs, w_gs, xi_gs, ii_gs)

        stage_nums = jnp.arange(s)

        scan_inputs = (stage_nums, a0, b0, a1, b1, b2)

        def sum_prev_stages(_stage_out_buff, _a_j):
            # Unwrap the buffer
            _stage_out_view = _stage_out_buff[...]
            # Sum up the previous stages weighted by the coefficients in the tableau
            return jnp.tensordot(jnp.conj(_a_j), _stage_out_view, axes=1)

        def insert_jth_stage(results, k_j, j):
            # Insert the result of the jth stage into the buffer
            return jtu.tree_map(
                lambda k_j_leaf, res_leaf: res_leaf.at[j].set(k_j_leaf), k_j, results
            )

        def stage(
            _carry: _CarryType,
            x: tuple[IntScalarLike, Array, Array, Array, Array, Array],
        ):
            # Represents the jth stage of the SRK.

            j, a0_j, b0_j, a1_j, b1_j, b2_j = x
            # a_levy_list_j = [aH_j, aK_j] (if those entries exist) where
            # aH_j is the row in the aH matrix corresponding to stage j
            # same for aK_j, but for space-time-time Lévy area K.
            _h_fs, _w_gs, _xi_gs, _ii_gs = _carry
            a0_h_f_sum = sum_prev_stages(_h_fs, a0_j)
            b0_w_g_sum = sum_prev_stages(_w_gs, b0_j)
            a1_h_f_sum = sum_prev_stages(_h_fs, a1_j)
            assert a0_h_f_sum.shape == a1_h_f_sum.shape == b0_w_g_sum.shape == (n,), (
                f"Expected {(n,)}, got {a0_h_f_sum.shape},"
                f" {a1_h_f_sum.shape} and {b0_w_g_sum.shape}"
            )
            b1_xi_g_sum = sum_prev_stages(_xi_gs, b1_j)
            b2_ii_g_sum = sum_prev_stages(ii_gs, b2_j)
            assert (
                b1_xi_g_sum.shape == b2_ii_g_sum.shape == (n, d)
            ), f"Expected {(n, d)}, got {b1_xi_g_sum.shape}, {b2_ii_g_sum.shape}"

            z0_j = y0 + a0_h_f_sum + b0_w_g_sum
            zk_j = y0[:, None] + a1_h_f_sum[:, None] + b1_xi_g_sum + b2_ii_g_sum

            # Compute the drift term
            h_f_j = drift.vf_prod(t0, z0_j, args, h)
            _h_fs = insert_jth_stage(_h_fs, h_f_j, j)

            # for diffusion it is a bit more complicated.
            # g: (n,) -> (n, d) and we want to apply the kth dim of g
            # to the kth dim of zk_j. So g_j[:, k] = g(zk_j[:, k])[:, k]
            # We can do this by vmaping g over zk_j and then taking the diagonal
            # XLA will optimise away the unnecessary computation.
            vec_g = jax.vmap(diffusion.vf, in_axes=(None, 1, None), out_axes=2)
            g_z_j_full = vec_g(t0, zk_j, args)
            assert g_z_j_full.shape == (
                n,
                d,
                d,
            ), f"Expected {(n, d, d)}, got {g_z_j_full.shape}"
            g_j = jnp.diagonal(g_z_j_full, axis1=1, axis2=2)
            del g_z_j_full

            w_g_j = jnp.tensordot(g_j, w, axes=1)
            assert w_g_j.shape == (n,)
            _w_gs = insert_jth_stage(_w_gs, w_g_j, j)
            xi_g_j = xi * g_j
            ii_g_j = jnp.tensordot(g_j, ii, axes=1)
            assert xi_g_j.shape == ii_g_j.shape == (n, d)
            _xi_gs = insert_jth_stage(_xi_gs, xi_g_j, j)
            _ii_gs = insert_jth_stage(_ii_gs, ii_g_j, j)

            # all g_js are used in the output stage, so we put them
            # into the output, not the carry. They will end up being
            # accumulated over the stages at the end.

            return (_h_fs, _w_gs, _xi_gs, _ii_gs), g_j

        scan_out = eqxi.scan(
            stage,
            carry,
            scan_inputs,
            s,
            buffers=lambda x: x,
            kind="checkpointed",
            checkpoints="all",
        )

        # output of lax.scan is ((h_fs, w_gs, xi_gs, ii_gs), gs)
        (h_fs, w_gs, _, _), gs = scan_out
        assert gs.shape == (s, n, d)
        ii_diag_gs = jnp.tensordot(gs, ii_diag, axes=1)
        assert ii_diag_gs.shape == w_gs.shape == h_fs.shape == (s, n)

        drift_result = sum_prev_stages(h_fs, alpha)
        diffusion_result = sum_prev_stages(w_gs, beta0) + sum_prev_stages(
            ii_diag_gs, beta1
        )

        y1 = y0 + drift_result + diffusion_result
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
