from dataclasses import dataclass
from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω

from ..custom_types import Bool, DenseInfo, LevyVal, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..solution import RESULTS
from ..term import AbstractTerm, ControlTerm, MultiTerm, ODETerm
from .base import AbstractItoSolver


_ErrorEstimate = None
_SolverState = None


@dataclass(frozen=True)
class StochasticButcherTableau:
    """A Butcher Tableau for Additive-noise SRK methods."""

    # Only supports explicit SRK so far
    c: np.ndarray
    b: np.ndarray
    a: list[np.ndarray]

    # coefficients for W and H (of shape (len(c)+1,)
    cw: np.ndarray
    ch: np.ndarray
    cw_last: Scalar
    ch_last: Scalar

    def __post_init__(self):
        assert self.c.ndim == 1
        for a_i in self.a:
            assert a_i.ndim == 1
        assert self.b.ndim == 1
        assert self.c.shape[0] == len(self.a)
        assert all(i + 1 == a_i.shape[0] for i, a_i in enumerate(self.a))
        assert self.c.shape[0] + 1 == self.b.shape[0]
        assert self.cw.shape[0] == self.b.shape[0]
        assert self.ch.shape[0] == self.b.shape[0]
        for i, (a_i, c_i) in enumerate(zip(self.a, self.c)):
            assert np.allclose(sum(a_i), c_i)
        assert np.allclose(sum(self.b), 1.0)

        # TODO: add checks for whether the method is FSAL


class ANSR(AbstractItoSolver):
    r"""Additive-Noise Stochastic Runge-Kutta method.

    The second term in the MultiTerm must be a Control term with
    control=VirtualBrownianTree(compute_stla=True), since this method
    makes use of space-time Levy area.

    Given the SDE
    $dX_t = f(t, X_t) dt + σ dW_t$

    We construct the SRK as follows:

    $y_1 = y_0 + h \Big(\sum_{j=1}^s b_j k_j \Big) + σ \, (c^W_{s+1} ΔW + c^H_{s+1} ΔH)$

    $k_j = f(t_0 + c_j h , z_j)$

    $z_j = y_0 + h \Big(\sum_{i=1}^{j-1} a_{j,i} k_i \Big) + σ \, (c^W_j ΔW + c^H_j ΔH)$

    where $ΔW := W_{t_0, t_1}$ is the increment of the Brownian motion and
    $ΔH := H_{t_0, t_1}$ is its corresponding space-time Levy Area.

    The values $( a_{i,j} ) , b_j, c_j, c^W_j, c^H_j$ are provided
    in the StochasticButcherTableau.
    """

    term_structure = MultiTerm[Tuple[ODETerm, ControlTerm]]
    interpolation_cls = LocalLinearInterpolation
    tableau: StochasticButcherTableau

    def order(self, terms):
        return 1  # should be modified depending on tableau

    def strong_order(self, terms):
        return 0.5  # should be modified depending on tableau

    def init(
        self,
        terms: MultiTerm[Tuple[ODETerm, ControlTerm]],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> _SolverState:
        return None

    def embed_a_lower(self, dtype):
        """
        Takes the lower-triangular (a_j_i) and embeds it in an n*n array.
        """
        num_stages = len(self.tableau.b)
        a = self.tableau.a
        tab_a = np.zeros((num_stages, num_stages))
        for i, a_i in enumerate(a):
            tab_a[i + 1, : i + 1] = a_i
        return jnp.asarray(tab_a, dtype=dtype)

    def step(
        self,
        terms: MultiTerm[Tuple[ODETerm, ControlTerm]],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        h = t1 - t0
        drift, diffusion = terms.terms

        # compute the Brownian increment and space-time Levy area
        levy = diffusion.contr(t0, t1, use_levy=True)
        assert isinstance(levy, LevyVal) and (levy.H is not None), (
            "The diffusion should be a ControlTerm controlled by either a"
            "VirtualBrownianTree or an UnsafeBrownianPath with"
            "`spacetime_levyarea` set to True."
        )
        w = levy.W
        hh = levy.H
        sigma = diffusion.vf(t0, y0, args)

        def stage(
            carry: tuple[int, list[PyTree]], x: tuple[jax.Array, Scalar, Scalar, Scalar]
        ):
            # Represents the jth stage of the SRK.
            # carry = (j, hks_{j-1}) where
            # hks_{j-1} = [hk1, hk2, ..., hk_{j-1}, 0, 0, ..., 0]
            # hki = drift.vf_prod(t0 + c_i*h, y_i, args, h) (i.e. hki = h * k_i)
            a_j, c_j, cw_j, ch_j = x
            j, hks_j = carry
            diffusion_control = cw_j * w + ch_j * hh

            # compute Σ_{i=1}^{j-1} a_j_i hk_i
            def lin_comb_a_j(hk_leaf):
                return jnp.tensordot(a_j, hk_leaf, axes=1)

            a_j_mult_k = jtu.tree_map(lin_comb_a_j, hks_j)

            # z_j = y_0 + h (Σ_{i=1}^{j-1} a_j_i k_i) + σ * (cw_j * ΔW + ch_j * ΔH)
            z_j = (
                y0**ω
                + a_j_mult_k**ω
                + (diffusion.prod(sigma, diffusion_control)) ** ω
            ).ω

            hk_j = drift.vf_prod(t0 + c_j * h, z_j, args, h)
            hks_j = jtu.tree_map(
                lambda ks_leaf, k_j_leaf: ks_leaf.at[j].set(k_j_leaf), hks_j, hk_j
            )
            # note that carry will already contain the whole stack of
            # k_js, so no need for second return value
            return (j + 1, hks_j), None

        a = self.embed_a_lower(jnp.dtype(y0))
        c = jnp.insert(jnp.asarray(self.tableau.c, dtype=jnp.dtype(y0)), 0, 0.0)
        b = jnp.asarray(self.tableau.b, dtype=jnp.dtype(y0))
        cw = jnp.asarray(self.tableau.cw, dtype=jnp.dtype(y0))
        ch = jnp.asarray(self.tableau.ch, dtype=jnp.dtype(y0))
        cw_last = jnp.asarray(self.tableau.cw_last, dtype=jnp.dtype(y0))
        ch_last = jnp.asarray(self.tableau.ch_last, dtype=jnp.dtype(y0))
        # hks is a PyTree of the same shape as y0, except that the arrays inside have
        # an additional batch dimension of size len(b) (i.e. num stages)
        hks = jtu.tree_map(
            lambda leaf: jnp.zeros(shape=(len(b),) + leaf.shape, dtype=leaf.dtype), y0
        )
        carry = (0, hks)

        # output of lax.scan is ((num_stages, hks), [None, None, ... , None])
        (_, hks), _ = lax.scan(stage, carry, (a, c, cw, ch), length=len(b))

        # compute Σ_{j=1}^s b_j k_j
        def lin_comb_b(k_leaf):
            return jnp.tensordot(b, k_leaf, axes=1)

        b_mult_k = jtu.tree_map(lin_comb_b, hks)

        diffusion_contr = cw_last * w + ch_last * hh
        y1 = (y0**ω + b_mult_k**ω + (diffusion.prod(sigma, diffusion_contr)) ** ω).ω
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
