from typing import Optional

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω

from ..brownian.base import AbstractBrownianPath
from ..custom_types import Array, Bool, DenseInfo, LevyVal, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..solution import RESULTS
from ..term import AbstractTerm, MultiTerm, ODETerm
from .base import AbstractStratonovichSolver
from .srk import StochasticButcherTableau


_ErrorEstimate = None
_SolverState = None


class AbstractAdditiveSRK(AbstractStratonovichSolver):
    r"""Additive-Noise Stochastic Runge-Kutta method.

    The second term in the MultiTerm must be a `ControlTerm` with
    `control=VirtualBrownianTree(levy_area="space-time")`, since this method
    makes use of space-time Lévy area.

    Given the SDE
    $dX_t = f(t, X_t) dt + σ dW_t$

    We construct the SRK with $s$ stages as follows:

    $y_{n+1} = y_n + h \Big(\sum_{j=1}^s b_j k_j \Big) + σ \, (b^W
    \, W_{t_0, t_1} + b^H \, H_{t_0, t_1})$

    $k_j = f(t_0 + c_j h , z_j)$

    $z_j = y_n + h \Big(\sum_{i=1}^{j-1} a_{j,i} k_i \Big) + σ
    \, (c^W_j W_{t_0, t_1} + c^H_j H_{t_0, t_1})$

    where $W_{t_0, t_1}$ is the increment of the Brownian motion and
    $H_{t_0, t_1}$ is its corresponding space-time Lévy Area.

    The values $( a_{i,j} ) , b_j, c_j, c^W_j, c^H_j$ are provided
    in the `StochasticButcherTableau`.
    """

    term_structure = MultiTerm[tuple[ODETerm, AbstractTerm]]
    interpolation_cls = LocalLinearInterpolation
    tableau: StochasticButcherTableau

    def init(
        self,
        terms: term_structure,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> _SolverState:
        # Check that the diffusion has levy_area="space-time"
        _, diffusion = terms.terms
        is_bm = lambda x: isinstance(x, AbstractBrownianPath)
        leaves = jtu.tree_leaves(diffusion, is_leaf=is_bm)
        paths = [x for x in leaves if is_bm(x)]
        for path in paths:
            if not path.levy_area == "space-time":
                raise ValueError(
                    "The Brownian path controlling the diffusion "
                    "should be initialised with `levy_area='space-time'`"
                )

        # check that the vector field of the diffusion term is constant
        sigma, (t_sigma, y_sigma) = eqx.filter_jvp(
            lambda t, y: diffusion.vf(t, y, args), (t0, y0), (t0, y0)
        )
        if (t_sigma is not None) or (y_sigma is not None):
            raise ValueError(
                "Vector field of the diffusion term should be constant, "
                "independent of t and y."
            )

        return None

    def _embed_a_lower(self, dtype):
        num_stages = len(self.tableau.b_sol)
        a = self.tableau.a
        tab_a = np.zeros((num_stages, num_stages))
        for i, a_i in enumerate(a):
            tab_a[i + 1, : i + 1] = a_i
        return jnp.asarray(tab_a, dtype=dtype)

    def step(
        self,
        terms: term_structure,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        h = t1 - t0
        drift, diffusion = terms.terms

        # compute the Brownian increment and space-time Lévy area
        bm_inc = diffusion.contr(t0, t1, use_levy=True)
        assert isinstance(bm_inc, LevyVal), (
            "The diffusion should be a ControlTerm controlled by either a"
            "VirtualBrownianTree or an UnsafeBrownianPath"
        )
        sigma = diffusion.vf(t0, y0, args)
        w = bm_inc.W

        levy_areas = []
        if self.tableau.cH is not None:
            assert bm_inc.H is not None, (
                "The diffusion should be a ControlTerm controlled by either a"
                "VirtualBrownianTree or an UnsafeBrownianPath with"
                "`levy_area='space-time'`"
            )
            levy_areas.append(bm_inc.H)

        def add_levy_to_w(_cw, *_c_levy):
            def aux_add_levy(w_leaf, *levy_leaves):
                return _cw * w_leaf + sum(
                    _c * _leaf for _c, _leaf in zip(_c_levy, levy_leaves)
                )

            return aux_add_levy

        def stage(
            carry: tuple[int, PyTree[Array]],
            x: tuple[jax.Array, Scalar, Scalar, Optional[Scalar]],
        ):
            # Represents the jth stage of the SRK.
            # carry = (j, hks_{j-1}) where
            # hks_{j-1} = [hk1, hk2, ..., hk_{j-1}, 0, 0, ..., 0]
            # hki = drift.vf_prod(t0 + c_i*h, y_i, args, h) (i.e. hki = h * k_i)
            a_j, c_j, cW_j, c_levy_j = x
            # for now c_levy_j is just [cH_j]

            j, hks_j = carry
            diffusion_control = jtu.tree_map(
                add_levy_to_w(cW_j, *c_levy_j), w, *levy_areas
            )

            # compute Σ_{i=1}^{j-1} a_j_i hk_i

            a_j_mult_k = jtu.tree_map(lambda lf: jnp.tensordot(a_j, lf, axes=1), hks_j)

            # z_j = y_0 + h (Σ_{i=1}^{j-1} a_j_i k_i) + σ * (cW_j * ΔW + cH_j * ΔH)
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

        dtype = jnp.dtype(jtu.tree_leaves(y0)[0])
        a = self._embed_a_lower(dtype)
        c = jnp.asarray(np.insert(self.tableau.c, 0, 0.0), dtype=dtype)
        b_sol = jnp.asarray(self.tableau.b_sol, dtype=dtype)
        cW = jnp.asarray(self.tableau.cW, dtype=dtype)
        bW = jnp.asarray(self.tableau.bW, dtype=dtype)
        b_levy = []
        c_levy = []
        if self.tableau.cH is not None:
            c_levy.append(jnp.asarray(self.tableau.cH, dtype=dtype))
            b_levy.append(jnp.asarray(self.tableau.bH, dtype=dtype))

        # hks is a PyTree of the same shape as y0, except that the arrays inside have
        # an additional batch dimension of size len(b_sol) (i.e. num stages)
        hks = jtu.tree_map(
            lambda leaf: jnp.zeros(shape=(len(b_sol),) + leaf.shape, dtype=leaf.dtype),
            y0,
        )
        carry = (0, hks)

        scan_inputs = (a, c, cW, c_levy)
        # output of lax.scan is ((num_stages, hks), None)
        (_, hks), _ = lax.scan(stage, carry, scan_inputs, length=len(b_sol))

        # compute Σ_{j=1}^s b_j k_j
        if self.tableau.b_error is None:
            error = None
        else:
            b_err = jnp.asarray(self.tableau.b_error, dtype=dtype)
            error = jtu.tree_map(
                lambda lf: jnp.abs(jnp.tensordot(b_err, lf, axes=1)), hks
            )

        # y1 = y0 + (Σ_{i=1}^{s} b_j * h*k_j) + σ * (bW * ΔW + bH * ΔH)

        stages = jtu.tree_map(lambda lf: jnp.tensordot(b_sol, lf, axes=1), hks)

        diffusion_contr = jtu.tree_map(add_levy_to_w(bW, *b_levy), w, *levy_areas)

        y1 = (y0**ω + stages**ω + (diffusion.prod(sigma, diffusion_contr)) ** ω).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, error, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> PyTree:
        return terms.vf(t0, y0, args)
