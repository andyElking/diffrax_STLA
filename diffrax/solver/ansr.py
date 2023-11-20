from dataclasses import dataclass
from typing import Optional, Tuple

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
from ..term import _ControlTerm, AbstractTerm, MultiTerm, ODETerm
from .base import AbstractStratonovichSolver


_ErrorEstimate = None
_SolverState = None


@dataclass(frozen=True)
class StochasticButcherTableau:
    """A Butcher Tableau for Additive-noise SRK methods."""

    # Only supports explicit SRK so far
    c: np.ndarray
    b_sol: np.ndarray
    b_error: Optional[np.ndarray]
    a: list[np.ndarray]

    # coefficients for W and H (of shape (len(c)+1,)
    bW: np.ndarray
    # bH is None if spacetime_levyarea=False
    bH: Optional[np.ndarray]

    # assuming SDE has additive noise we only need a 1-dimensional array
    # for the coefficients in front of the Brownian increment and the
    # space-time Lévy area.
    cW: Optional[np.ndarray]
    cH: Optional[np.ndarray]

    # If the SDE has non-additive noise, we need an equivalent of the
    # matrix a, one for the Brownian motion and one for its space-time
    # Lévy area.
    aW: Optional[list[np.ndarray]]
    aH: Optional[list[np.ndarray]]

    additive_noise: bool = True

    def __post_init__(self):
        assert self.c.ndim == 1
        for a_i in self.a:
            assert a_i.ndim == 1
        assert self.b_sol.ndim == 1
        assert (self.b_error is None) or self.b_error.ndim == 1
        assert self.c.shape[0] == len(self.a)
        assert all(i + 1 == a_i.shape[0] for i, a_i in enumerate(self.a))
        assert (self.b_error is None) or self.b_error.shape[0] == self.b_sol.shape[0]
        assert self.c.shape[0] + 1 == self.b_sol.shape[0]

        if self.additive_noise:
            # Then we need one coefficient for the Brownian motion per stage
            # and one for the final output.
            assert self.cW is not None
            assert self.cW.shape[0] == self.b_sol.shape[0]
            assert self.bW.ndim == 0  # only one coefficient for the final output

            if self.bH is not None:  # i.e. if we use space-time Lévy area
                assert self.bH.ndim == 0  # only one coefficient for the final output
                assert self.cH is not None
                assert self.cH.shape[0] == self.b_sol.shape[0]
        else:
            # Then we need a matrix of coefficients for the Brownian motion
            assert self.aW is not None
            assert all(i + 1 == a_i.shape[0] for i, a_i in enumerate(self.aW))
            assert (
                self.bW.shape == self.b_sol.shape
            )  # one coefficient per output of each stage
            if self.bH is not None:
                assert self.aH is not None
                assert all(i + 1 == a_i.shape[0] for i, a_i in enumerate(self.aH))
                assert (
                    self.bH.shape == self.b_sol.shape
                )  # one coefficient per output of each stage

        for i, (a_i, c_i) in enumerate(zip(self.a, self.c)):
            assert np.allclose(sum(a_i), c_i)
        assert np.allclose(sum(self.b_sol), 1.0)

        # TODO: add checks for whether the method is FSAL


StochasticButcherTableau.__init__.__doc__ = """**Arguments:**

Let `k` denote the number of stages of the solver.

- `a`: the lower triangle (without the diagonal) of the Butcher tableau. Should
    be a tuple of NumPy arrays, corresponding to the rows of this lower triangle. The
    first array represents the should be of shape `(1,)`. Each subsequent array should
    be of shape `(2,)`, `(3,)` etc. The final array should have shape `(k - 1,)`.
- `b_sol`: the linear combination of stages to take to produce the output at each step.
    Should be a NumPy array of shape `(k,)`.
- `b_error`: the linear combination of stages to take to produce the error estimate at
    each step. Should be a NumPy array of shape `(k,)`. Note that this is *not*
    differenced against `b_sol` prior to evaluation. (i.e. `b_error` gives the linear
    combination for producing the error estimate directly, not for producing some
    alternate solution that is compared against the main solution).
- `c`: the time increments used in the Butcher tableau.
    Should be a NumPy array of shape `(k-1,)`, as the first stage has time increment 0
- `cW`: The coefficients in front of the Brownian increment at each stage.
    Should be a NumPy array of shape `(k,)`. Only used when `additive_noise=True`.
- `cH`: The coefficients in front of the space-time Lévy area at each stage.
    Should be a NumPy array of shape `(k,)`. Only used when `additive_noise=True`.
- `bW`: Coefficient for the Brownian increment when computing the
    output $y_{n+1}$. Should be a `Scalar`.
- `bH`: Coefficient for the space-time Lévy area when computing the
    output $y_{n+1}$. Should be a `Scalar`.
- `aW`: The coefficients in front of the Brownian increment at each stage.
    Should be of the same shape as `a`. Only used when `additive_noise=False`.
- `aH`: The coefficients in front of the space-time Lévy area at each stage.
    Should be of the same shape as `a`. Only used when `additive_noise=False`.
- `additive_noise`: Whether the SDE has additive noise. If `True`, then `bW` and `bH`
    should be scalars, and `cW` and `cH` should be arrays of shape `(k,)`. If `False`,
    then `bW` and `bH` should be arrays of shape `(k,)`, `cW` and `cH` should be `None`,
    and `aW` and `aH` should be lists of arrays of the same shape as `a`.
"""


class AbstractANSR(AbstractStratonovichSolver):
    r"""Additive-Noise Stochastic Runge-Kutta method.

    The second term in the MultiTerm must be a `ControlTerm` with
    `control=VirtualBrownianTree(spacetime_levyarea=True)`, since this method
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

    term_structure = MultiTerm[Tuple[ODETerm, _ControlTerm]]
    interpolation_cls = LocalLinearInterpolation
    tableau: StochasticButcherTableau

    def init(
        self,
        terms: MultiTerm[Tuple[ODETerm, _ControlTerm]],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> _SolverState:
        # Check that the diffusion has spacetime_levyarea enabled
        _, diffusion = terms.terms
        is_bm = lambda x: isinstance(x, AbstractBrownianPath)
        leaves = jtu.tree_leaves(diffusion, is_leaf=is_bm)
        paths = [x for x in leaves if is_bm(x)]
        for path in paths:
            if not path.spacetime_levyarea:
                raise ValueError(
                    "The Brownian path controlling the diffusion "
                    "should be initialised with `compute_stla=True`"
                )

        if self.tableau.additive_noise:
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
        terms: MultiTerm[Tuple[ODETerm, _ControlTerm]],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        dtype = jnp.result_type(*jtu.tree_leaves(y0))
        drift, diffusion = terms.terms
        additive_noise = self.tableau.additive_noise

        # time increment
        h = t1 - t0
        # Brownian increment (and space-time Lévy area)
        bm_inc = diffusion.contr(t0, t1, use_levy=True)
        assert isinstance(bm_inc, LevyVal) and (bm_inc.H is not None), (
            "The diffusion should be a ControlTerm controlled by either a"
            "VirtualBrownianTree or an UnsafeBrownianPath with"
            "`spacetime_levyarea=True`"
        )
        w = bm_inc.W

        levy_areas = []
        if self.tableau.bH is not None:
            assert bm_inc.H is not None
            levy_areas.append(bm_inc.H)

        def add_levy_to_w(_cw, *_c_levy):
            def aux_add_levy(w_leaf, *levy_leaves):
                return _cw * w_leaf + sum(
                    _c * _leaf for _c, _leaf in zip(_c_levy, levy_leaves)
                )

            return aux_add_levy

        a = self._embed_a_lower(dtype)
        c = jnp.asarray(np.insert(self.tableau.c, 0, 0.0), dtype=dtype)
        b_sol = jnp.asarray(self.tableau.b_sol, dtype=dtype)

        # b looks similar regardless of whether we have additive noise or not
        bW = jnp.asarray(self.tableau.bW, dtype=dtype)
        b_levy = []
        if self.tableau.bH is not None:
            b_levy.append(jnp.asarray(self.tableau.bH, dtype=dtype))

        # _hf_list is a PyTree of the same shape as y0, except that the arrays inside
        # have an additional batch dimension of size len(b_sol) (i.e. num stages)
        hf_list = jtu.tree_map(
            lambda leaf: jnp.zeros(shape=(len(b_sol),) + leaf.shape, dtype=leaf.dtype),
            y0,
        )

        a_levy = []  # will contain cH if additive_noise=True, aH otherwise
        # later other kinds of Levy area will be added to this list

        levy_g_list = []  # will contain levy * g(t0 + c_j * h, z_j) for each stage j
        # and for each levy area

        if additive_noise:
            # compute g once since it is constant
            g0 = diffusion.vf(t0, y0, args)
            w_g_0 = diffusion.prod(g0, w)
            a_w = jnp.asarray(self.tableau.cW, dtype=dtype)
            if self.tableau.bH is not None:
                levy_g_list.append(diffusion.prod(g0, bm_inc.H))
                a_levy.append(jnp.asarray(self.tableau.cH, dtype=dtype))

            carry_type = tuple[int, PyTree[Array]]
            carry: carry_type = (0, hf_list)  # just the stage counter and _hf_list

        else:
            # g is not constant, so we need to compute it at each stage
            # we will carry the value of W * g(t0 + c_j * h, z_j)
            wg_list = jtu.tree_map(
                lambda leaf: jnp.zeros(
                    shape=(len(b_sol),) + leaf.shape, dtype=leaf.dtype
                ),
                y0,
            )
            # do the same for each type of Levy area
            if self.tableau.bH is not None:
                wh_list = jtu.tree_map(
                    lambda leaf: jnp.zeros(
                        shape=(len(b_sol),) + leaf.shape, dtype=leaf.dtype
                    ),
                    y0,
                )
                levy_g_list.append(wh_list)

            carry_type = tuple[int, PyTree[Array], PyTree[Array], PyTree[Array]]
            carry: carry_type = (0, hf_list, wg_list, levy_g_list)

            a_w = jnp.asarray(self.tableau.aW, dtype=dtype)

        scan_inputs = (a, c, a_w, a_levy)

        def stage(
            _carry: carry_type,
            x: tuple[jax.Array, Scalar, Scalar, Optional[Scalar]],
        ):
            # Represents the jth stage of the SRK.
            # carry = (j, hks_{j-1}) where
            # _hf_list = [hk1, hk2, ..., hk_{j-1}, 0, 0, ..., 0]
            # hki = drift.vf_prod(t0 + c_i*h, y_i, args, h) (i.e. hki = h * k_i)
            a_j, c_j, a_w_j, a_levy_list_j = x
            # for now a_levy_j is just [cH_j]

            if additive_noise:
                j, _hf_list = _carry
                diffusion_result = jtu.tree_map(
                    add_levy_to_w(a_w_j, *a_levy_list_j), w_g_0, *levy_g_list
                )
            else:
                j, _hf_list, _wg_list, _levy_g_list = _carry
                wg_sum = jtu.tree_map(
                    lambda lf: jnp.tensordot(a_w_j, lf, axes=1), _wg_list
                )
                levy_sum_list = []
                for a_levy_j, levy_g in zip(a_levy_list_j, _levy_g_list):
                    levy_g_sum = jtu.tree_map(
                        lambda lf: jnp.tensordot(a_levy_j, lf, axes=1), levy_g
                    )
                    levy_sum_list.append(levy_g_sum)

                diffusion_result = jtu.tree_map(
                    lambda _wg, *_levy_g: sum(_levy_g, _wg), wg_sum, *levy_sum_list
                )

            # compute Σ_{i=1}^{j-1} a_j_i hk_i

            a_j_mult_k = jtu.tree_map(
                lambda lf: jnp.tensordot(a_j, lf, axes=1), _hf_list
            )

            # z_j = y_0 + h (Σ_{i=1}^{j-1} a_j_i k_i) + σ * (a_w_j * ΔW + cH_j * ΔH)
            z_j = (y0**ω + a_j_mult_k**ω + diffusion_result**ω).ω

            hf_j = drift.vf_prod(t0 + c_j * h, z_j, args, h)

            # wg_j = diffusion.vf_prod(t0 + c_j * h, z_j, args, w)
            levy_g_j_list = []
            if self.tableau.bH is not None:
                levy_g_j_list.append(
                    diffusion.vf_prod(t0 + c_j * h, z_j, args, bm_inc.H)
                )

            # TODO: finish the insertion of stage results into carry

            # insert the stage results into the list
            _hf_list = jtu.tree_map(
                lambda ks_leaf, k_j_leaf: ks_leaf.at[j].set(k_j_leaf), _hf_list, hf_j
            )
            # note that carry will already contain the whole stack of
            # k_js, so no need for second return value
            return (j + 1, _hf_list), None

        # output of lax.scan is ((num_stages, _hf_list), None)
        (_, hf_list), _ = lax.scan(stage, carry, scan_inputs, length=len(b_sol))

        # compute Σ_{j=1}^s b_j k_j
        if self.tableau.b_error is None:
            error = None
        else:
            b_err = jnp.asarray(self.tableau.b_error, dtype=dtype)
            error = jtu.tree_map(
                lambda lf: jnp.abs(jnp.tensordot(b_err, lf, axes=1)), hf_list
            )

        # y1 = y0 + (Σ_{i=1}^{s} b_j * h*k_j) + σ * (bW * ΔW + bH * ΔH)

        stages = jtu.tree_map(lambda lf: jnp.tensordot(b_sol, lf, axes=1), hf_list)

        diffusion_contr = jtu.tree_map(add_levy_to_w(bW, *b_levy), w, *levy_areas)

        y1 = (y0**ω + stages**ω + (diffusion.prod(g0, diffusion_contr)) ** ω).ω
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
