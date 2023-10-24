from dataclasses import dataclass, field
from typing import Tuple, Optional

import jax
import jax.tree_util as jtu
import numpy as np
import jax.numpy as jnp
import jax.lax as lax
import equinox.internal as eqxi
from equinox.internal import ω
from jax import debug

from ..custom_types import Bool, DenseInfo, PyTree, Scalar, LevyVal
from ..local_interpolation import LocalLinearInterpolation
from ..solution import RESULTS
from ..term import AbstractTerm, MultiTerm, ODETerm, ControlTerm, _sum
from .base import AbstractItoSolver

_ErrorEstimate = None
_SolverState = None


@dataclass(frozen=True)
class StochasticButcherTableau:
    """A Butcher Tableau for Additive-noise SRK methods.

    Given the SDE
    dX_t = f(t, X_t) dt + σ dW_t

    We construct the SRK as follows:
    y_1 = y_0 + h (Σ_{j=1}^s b_j k_j) + σ * (cw_last * ΔW + ch_last * ΔH)
    k_j = f(t_0 + c_j h, z_j)
    z_j = y_0 + h (Σ_{i=1}^{j-1} a_j_i k_j) + σ * (cw_j * ΔW + ch_j * ΔH)

    where ΔW := W_{t0, t1} is the increment of the Brownian motion and
    ΔH := H_{t0, t1} is its corresponding space-time Levy Area.
    """

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
    """Additive-Noise Stochastic Runge-Kutta method.
    For description see StochasticButcherTableau.
    """

    term_structure = MultiTerm[Tuple[ODETerm, ControlTerm]]
    interpolation_cls = LocalLinearInterpolation
    tableau: StochasticButcherTableau

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.5

    def init(
            self,
            terms: MultiTerm[Tuple[ODETerm, ControlTerm]],
            t0: Scalar,
            t1: Scalar,
            y0: PyTree,
            args: PyTree,
    ) -> _SolverState:
        return None

    def embed_a_lower(self):
        num_stages = len(self.tableau.b)
        a = self.tableau.a
        tab_a = np.zeros((num_stages, num_stages))
        for i, a_i in enumerate(a):
            tab_a[i + 1, : i + 1] = a_i
        return jnp.asarray(tab_a)

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
        levy: LevyVal = diffusion.levy_contr(t0, t1)
        w = levy.W
        hh = levy.H
        sigma = diffusion.vf(t0, y0, args)

        def stage(carry: tuple[int, list[PyTree]], x: tuple[jax.Array, Scalar, Scalar, Scalar]):
            # carry = [k1, k2, ..., k_{j-1}, 0, 0, ..., 0] where ki are already multiplied by h, i.e.
            # ki = drift.vf_prod(t0 + c_i*h, y_i, args, h)
            a_j, c_j, cw_j, ch_j = x
            j, ks = carry
            diffusion_contr = cw_j * w + ch_j * hh

            def lin_comb_a_j(k_leaf):
                return jnp.tensordot(a_j, k_leaf, axes=1)
            a_j_mult_k = jtu.tree_map(lin_comb_a_j, ks)

            z_j = (y0 ** ω + (diffusion.prod(sigma, diffusion_contr)) ** ω + a_j_mult_k ** ω).ω

            k_j = drift.vf_prod(t0 + c_j * h, z_j, args, h)
            ks = jtu.tree_map(lambda ks_leaf, k_j_leaf: ks_leaf.at[j].set(k_j_leaf), ks, k_j)
            # note that carry will already contain the whole stack of k_js, so no need for second return value
            carry = (j+1, ks)
            return carry, None

        a = self.embed_a_lower()
        c = jnp.insert(self.tableau.c, 0, 0.0)
        b, cw, ch = self.tableau.b, self.tableau.cw, self.tableau.ch
        # ks is a PyTree of the same shape as y0, except that the arrays inside have
        # an additional batch dimesnion of size len(b) (i.e. num stages)
        ks = jtu.tree_map(lambda leaf: jnp.zeros(shape=(len(b),) + leaf.shape, dtype=leaf.dtype), y0)
        carry = (0, ks)
        (_, ks), _ = lax.scan(stage, carry, (a, c, cw, ch), length=len(b))

        def lin_comb_b(k_leaf):
            return jnp.tensordot(b, k_leaf, axes=1)
        b_mult_k = jtu.tree_map(lin_comb_b, ks)

        diffusion_contr = self.tableau.cw_last * w + self.tableau.ch_last * hh
        y1 = (y0 ** ω + b_mult_k ** ω + (diffusion.prod(sigma, diffusion_contr)) ** ω).ω
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
