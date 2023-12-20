from dataclasses import dataclass
from typing import Literal, Optional, TypeAlias

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω
from jaxtyping import Array, PyTree

from .._brownian.base import AbstractBrownianPath
from .._custom_types import (
    BoolScalarLike,
    DenseInfo,
    LevyVal,
    RealScalarLike,
    VF,
    Y,
)
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm, MultiTerm, ODETerm
from .base import AbstractStratonovichSolver


_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None
_CarryType: TypeAlias = tuple[int, PyTree[Array], PyTree[Array], PyTree[Array]]
_LA: TypeAlias = Literal["", "space-time", "space-time-time"]


@dataclass(frozen=True)
class StochasticButcherTableau:
    """A Butcher Tableau for Stochastic Runge-Kutta methods."""

    # Only supports explicit SRK so far
    c: np.ndarray
    b_sol: np.ndarray
    b_error: Optional[np.ndarray]
    a: list[np.ndarray]

    # coefficients for W and H (of shape (len(c)+1,)
    bW: np.ndarray
    bW_error: Optional[np.ndarray] = None
    # bH is None if spacetime_levyarea=False
    bH: Optional[np.ndarray] = None
    bH_error: Optional[np.ndarray] = None
    bK: Optional[np.ndarray] = None
    bK_error: Optional[np.ndarray] = None

    # assuming SDE has additive noise we only need a 1-dimensional array
    # for the coefficients in front of the Brownian increment and the
    # space-time Lévy area.
    cW: Optional[np.ndarray] = None
    cH: Optional[np.ndarray] = None
    cK: Optional[np.ndarray] = None

    # If the SDE has non-additive noise, we need an equivalent of the
    # matrix a, one for the Brownian motion and one for each type of
    # Lévy area.
    aW: Optional[list[np.ndarray]] = None
    aH: Optional[list[np.ndarray]] = None
    aK: Optional[list[np.ndarray]] = None

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
                assert self.cH is None

            if self.bK is not None:  # i.e. if we use space-time-time Lévy area
                assert self.bH is not None  # can only use K if we also use H
                assert self.bK.ndim == 0
                assert self.cK is not None
                assert self.cK.shape[0] == self.b_sol.shape[0]
            else:
                assert self.cK is None

            # check that all a are None
            assert self.aW is None
            assert self.aH is None
            assert self.aK is None
            assert self.bH_error is None
            assert self.bK_error is None

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
            else:
                assert self.aH is None

            if self.bK is not None:
                assert self.bH is not None  # can only use K if we also use H
                assert self.aK is not None
                assert all(i + 1 == a_i.shape[0] for i, a_i in enumerate(self.aK))
                assert self.bK.shape == self.b_sol.shape
            else:
                assert self.aK is None

            # check that all c are None
            assert self.cW is None
            assert self.cH is None
            assert self.cK is None

            if self.b_error is not None:
                assert self.bW_error is not None
                assert self.bW_error.shape == self.b_error.shape
                if self.bH is not None:
                    assert self.bH_error is not None
                    assert self.bH_error.shape == self.b_error.shape
                else:
                    assert self.bH_error is None
                if self.bK is not None:
                    assert self.bK_error is not None
                    assert self.bK_error.shape == self.b_error.shape
                else:
                    assert self.bK_error is None

        for i, (a_i, c_i) in enumerate(zip(self.a, self.c)):
            assert np.allclose(sum(a_i), c_i)
        assert np.allclose(sum(self.b_sol), 1.0)


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


class AbstractSRK(AbstractStratonovichSolver):
    r"""A general Stochastic Runge-Kutta method.

    The second term in the MultiTerm must be a `ControlTerm` with
    `control=VirtualBrownianTree`. Depending on the Butcher tableau, the
    `VirtualBrownianTree` may need to be initialised with 'levy_area="space-time"'
    or 'levy_area="space-time-time"'.

    Given the Stratonovich SDE
    $dX_t = f(t, X_t) dt + g(t, X_t) \circ dW_t$

    We construct the SRK with $s$ stages as follows:

    $y_{n+1} = y_n + h \Big(\sum_{j=1}^s b_j f_j \Big)
    + W_n \Big(\sum_{i=1}^{j-1} b^W_j g_i \Big)
    + H_n \Big(\sum_{i=1}^{j-1} b^H_j g_i \Big)$

    $f_j = f(t_0 + c_j h , z_j)$

    $g_j = g(t_0 + c_j h , z_j)$

    $z_j = y_n + h \Big(\sum_{i=1}^{j-1} a_{j,i} f_i \Big)
    + W_n \Big(\sum_{i=1}^{j-1} a^W_{j,i} g_i \Big)
    + H_n \Big(\sum_{i=1}^{j-1} a^H_{j,i} g_i \Big)$

    where $W_n = W_{t_n, t_{n+1}}$ is the increment of the Brownian motion and
    $H_n = H_{t_n, t_{n+1}}$ is its corresponding space-time Lévy Area.
    A similar term can also be added for the space-time-time Lévy area, K.

    In the special case, when the SDE has additive noise, i.e. when g is
    indepnedent of y (but can still depend on t), then the SDE can be written as
    $dX_t = f(t, X_t) dt + g(t) \, dW_t$, and we can simplify the above to

    $y_{n+1} = y_n + h \Big(\sum_{j=1}^s b_j k_j \Big) + g(t_n) \, (b^W
    \, W_n + b^H \, H_n)$

    $f_j = f(t_n + c_j h , z_j)$

    $z_j = y_n + h \Big(\sum_{i=1}^{j-1} a_{j,i} f_i \Big) + g(t_n)
    \, (c^W_j W_n + c^H_j H_n)$

    When g depends on t, we need to add a correction term to $y_{n+1}$ of
    the form $(g(t_{n+1}) - g(t_n)) \, (1/2 W_n - H_n)$.

    The coefficients are provided in the `StochasticButcherTableau`.
    """

    term_structure = MultiTerm[tuple[ODETerm, AbstractTerm]]
    interpolation_cls = LocalLinearInterpolation
    tableau: StochasticButcherTableau

    minimal_levy_area: _LA

    def __init__(self):
        if self.tableau.bK is not None:
            self.minimal_levy_area: _LA = "space-time-time"
        elif self.tableau.bH is not None:
            self.minimal_levy_area: _LA = "space-time"
        else:
            self.minimal_levy_area: _LA = ""

    def init(
        self,
        terms: term_structure,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: PyTree,
    ) -> _SolverState:
        # Check that the diffusion has the correct Levy area
        _, diffusion = terms.terms

        stla = self.tableau.bH is not None
        sttla = self.tableau.bK is not None

        is_bm = lambda x: isinstance(x, AbstractBrownianPath)
        leaves = jtu.tree_leaves(diffusion, is_leaf=is_bm)
        paths = [x for x in leaves if is_bm(x)]
        for path in paths:
            if sttla:
                if not path.levy_area == "space-time-time":
                    raise ValueError(
                        "The Brownian path controlling the diffusion "
                        "should be initialised with `levy_area='space-time-time'`"
                    )
            elif stla:
                if path.levy_area not in ["space-time", "space-time-time"]:
                    raise ValueError(
                        "The Brownian path controlling the diffusion "
                        "should be initialised with `levy_area='space-time'`"
                        "or `levy_area='space-time-time'`"
                    )

        if self.tableau.additive_noise:
            # check that the vector field of the diffusion term is constant
            ones_like_y0 = jtu.tree_map(lambda x: jnp.ones_like(x), y0)
            _, y_sigma = eqx.filter_jvp(
                lambda y: diffusion.vf(t0, y, args), (y0,), (ones_like_y0,)
            )
            # check if the PyTree is just made of Nones (inside other containers)
            if len(jtu.tree_leaves(y_sigma)) > 0:
                raise ValueError(
                    "Vector field of the diffusion term should be constant, "
                    "independent of y."
                )

        return None

    def _embed_a_lower(self, _a, dtype):
        num_stages = len(self.tableau.b_sol)
        tab_a = np.zeros((num_stages, num_stages))
        for i, a_i in enumerate(_a):
            tab_a[i + 1, : i + 1] = a_i
        return jnp.asarray(tab_a, dtype=dtype)

    def step(
        self,
        terms: term_structure,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        dtype = jnp.result_type(*jtu.tree_leaves(y0))
        drift, diffusion = terms.terms
        additive_noise = self.tableau.additive_noise

        # time increment
        h = t1 - t0

        # First all the drift related stuff
        a = self._embed_a_lower(self.tableau.a, dtype)
        c = jnp.asarray(np.insert(self.tableau.c, 0, 0.0), dtype=dtype)
        b_sol = jnp.asarray(self.tableau.b_sol, dtype=dtype)

        def make_zeros():
            def make_zeros_aux(leaf):
                return jnp.zeros((len(b_sol),) + leaf.shape, dtype=leaf.dtype)

            return jtu.tree_map(make_zeros_aux, y0)

        # _tfs is a PyTree of the same shape as y0, except that the arrays inside
        # have an additional batch dimension of size len(b_sol) (i.e. num stages)
        # This will be one of the entries of the carry of lax.scan. In each stage
        # one of the zeros will get replaced by the value of
        # h * f(t0 + c_j * h, z_j) where z_j is the jth stage of the SRK.
        # The name tf is because the value of f(t0 + c_j * h, z_j) is already
        # multiplied by the time increment h.
        tfs = make_zeros()

        # Now the diffusion related stuff
        # Brownian increment (and space-time Lévy area)
        bm_inc = diffusion.contr(t0, t1, use_levy=True)
        assert isinstance(bm_inc, LevyVal), (
            "The diffusion should be a ControlTerm controlled by either a"
            "VirtualBrownianTree or an UnsafeBrownianPath"
        )
        w = bm_inc.W

        # b looks similar regardless of whether we have additive noise or not
        b_w = jnp.asarray(self.tableau.bW, dtype=dtype)
        b_levy_list = []

        levy_areas = []
        stla = self.tableau.bH is not None
        sttla = self.tableau.bK is not None
        if stla or sttla:
            levy_areas.append(bm_inc.H)
            b_levy_list.append(jnp.asarray(self.tableau.bH, dtype=dtype))
            if sttla:
                assert bm_inc.K is not None, (
                    "The diffusion should be a ControlTerm controlled by either a"
                    "VirtualBrownianTree or an UnsafeBrownianPath with"
                    "`levy_area='space-time-time'`"
                )
                levy_areas.append(bm_inc.K)
                b_levy_list.append(jnp.asarray(self.tableau.bK, dtype=dtype))
            else:
                assert bm_inc.H is not None, (
                    "The diffusion should be a ControlTerm controlled by either a"
                    "VirtualBrownianTree or an UnsafeBrownianPath with"
                    "`levy_area='space-time'` or `levy_area='space-time-time'`"
                )

        def add_levy_to_w(_cw, *_c_levy):
            def aux_add_levy(w_leaf, *levy_leaves):
                return _cw * w_leaf + sum(
                    _c * _leaf for _c, _leaf in zip(_c_levy, levy_leaves)
                )

            return aux_add_levy

        a_levy = []  # will contain cH if additive_noise=True, aH otherwise
        # later other kinds of Levy area will be added to this list

        levy_gs_list = []  # will contain levy * g(t0 + c_j * h, z_j) for each stage j
        # and for each type of levy area (e.g. H, K, etc.)
        if additive_noise:
            # compute g once since it is constant
            g0 = diffusion.vf(t0, y0, args)
            w_g = diffusion.prod(g0, w)
            a_w = jnp.asarray(self.tableau.cW, dtype=dtype)
            if stla:
                levy_gs_list.append(diffusion.prod(g0, bm_inc.H))
                a_levy.append(jnp.asarray(self.tableau.cH, dtype=dtype))
            if sttla:
                levy_gs_list.append(diffusion.prod(g0, bm_inc.K))
                a_levy.append(jnp.asarray(self.tableau.cK, dtype=dtype))

            carry: _CarryType = (0, tfs, None, None)  # just the stage counter and _tfs

        else:
            # g is not constant, so we need to compute it at each stage
            # we will carry the value of W * g(t0 + c_j * h, z_j)
            # Since the carry of lax.scan needs to have constant shape,
            # we initialise a list of zeros of the same shape as y0, which will get
            # filled with the values of W * g(t0 + c_j * h, z_j) at each stage
            wg_list = make_zeros()
            # do the same for each type of Levy area
            if stla:
                levy_gs_list.append(make_zeros())
                a_levy.append(self._embed_a_lower(self.tableau.aH, dtype))
            if sttla:
                levy_gs_list.append(make_zeros())
                a_levy.append(self._embed_a_lower(self.tableau.aK, dtype))

            carry: _CarryType = (0, tfs, wg_list, levy_gs_list)

            a_w = self._embed_a_lower(self.tableau.aW, dtype)

        scan_inputs = (a, c, a_w, a_levy)

        def sum_prev_stages(_stage_out_list, _a_j):
            return jtu.tree_map(
                lambda lf: jnp.tensordot(_a_j, lf, axes=1), _stage_out_list
            )

        def stage(
            _carry: _CarryType,
            x: tuple[Array, Array, Array, list[Array]],
        ):
            # Represents the jth stage of the SRK.
            a_j, c_j, a_w_j, a_levy_list_j = x
            # a_levy_list_j = [aH_j, aK_j] (if those entries exist) where
            # aH_j is the row in the aH matrix corresponding to stage j
            # same for aK_j, but for space-time-time Lévy area K.
            j, _tfs, _wgs, _levy_gs_list = _carry

            if additive_noise:
                # carry = (j, _tfs) where
                # _tfs = Array[hk1, hk2, ..., hk_{j-1}, 0, 0, ..., 0]
                # hki = drift.vf_prod(t0 + c_i*h, y_i, args, h) (i.e. hki = h * k_i)
                assert _wgs is None and _levy_gs_list is None
                assert isinstance(levy_gs_list, list)
                _diffusion_result = jtu.tree_map(
                    add_levy_to_w(a_w_j, *a_levy_list_j),  # type: ignore
                    w_g,
                    *levy_gs_list,
                )
            else:
                # carry = (j, _tfs, _wgs, _levy_gs_list) where
                # _tfs = Array[tf1, tf2, ..., tf{j-1}, 0, 0, ..., 0]
                # _wgs = Array[wg1, wg2, ..., wg{j-1}, 0, 0, ..., 0]
                # _levy_gs_list = [H_gs, K_gs] (if those entries exist) where
                # H_gs = Array[Hg1, Hg2, ..., Hg{j-1}, 0, 0, ..., 0]
                # K_gs = Array[Kg1, Kg2, ..., Kg{j-1}, 0, 0, ..., 0]
                # tfi = drift.vf_prod(t0 + c_i*h, y_i, args, h) (i.e. hki = h * k_i)
                # wgi = diffusion.vf_prod(t0 + c_i*h, y_i, args, w) (i.e. wgi = w * g_i)

                wg_sum = sum_prev_stages(_wgs, a_w_j)
                levy_sum_list = [
                    sum_prev_stages(levy_gs, a_levy_j)
                    for a_levy_j, levy_gs in zip(a_levy_list_j, _levy_gs_list)
                ]

                _diffusion_result = jtu.tree_map(
                    lambda _wg, *_levy_g: sum(_levy_g, _wg), wg_sum, *levy_sum_list
                )

            # compute Σ_{i=1}^{j-1} a_j_i tf_i
            _drift_result = sum_prev_stages(_tfs, a_j)

            # z_j = y_0 + h (Σ_{i=1}^{j-1} a_j_i k_i) + g * (a_w_j * ΔW + cH_j * ΔH)
            z_j = (y0**ω + _drift_result**ω + _diffusion_result**ω).ω

            tf_j = drift.vf_prod(t0 + c_j * h, z_j, args, h)

            def insert_jth_stage(results, res_j):
                return jtu.tree_map(
                    lambda res_leaf, j_leaf: res_leaf.at[j].set(j_leaf), results, res_j
                )

            _tfs = insert_jth_stage(_tfs, tf_j)
            if additive_noise:
                return (j + 1, _tfs, None, None), None

            wg_j = diffusion.vf_prod(t0 + c_j * h, z_j, args, w)
            _wgs = insert_jth_stage(_wgs, wg_j)

            levy_g_list_j = [
                diffusion.vf_prod(t0 + c_j * h, z_j, args, levy) for levy in levy_areas
            ]
            _levy_gs_list = insert_jth_stage(_levy_gs_list, levy_g_list_j)

            return (j + 1, _tfs, _wgs, _levy_gs_list), None

        scan_out = lax.scan(stage, carry, scan_inputs, length=len(b_sol))

        if additive_noise:
            # output of lax.scan is ((num_stages, _tfs), None)
            (_, tfs, _, _), _ = scan_out
            diffusion_result = jtu.tree_map(
                add_levy_to_w(b_w, *b_levy_list), w_g, *levy_gs_list
            )

            # In the additive noise case (i.e. when g is independent of y),
            # we still need a correction term in case the diffusion vector field
            # g depends on t. This term is of the form $(g1 - g0) * (0.5*W_n - H_n)$.
            g1 = diffusion.vf(t1, y0, args)
            g_delta = (0.5 * (g1**ω - g0**ω)).ω  # type: ignore
            if stla:
                time_var_contr = (bm_inc.W**ω - 2.0 * bm_inc.H**ω).ω
                time_var_term = diffusion.prod(g_delta, time_var_contr)
            else:
                time_var_term = diffusion.prod(g_delta, bm_inc.W)
            diffusion_result = (diffusion_result**ω + time_var_term**ω).ω

        else:
            # output of lax.scan is ((num_stages, _tfs, _wgs, _levy_gs_list), None)
            (_, tfs, wgs, levy_gs_list), _ = scan_out
            b_wgs = sum_prev_stages(wgs, b_w)
            b_levy_gs_list = [
                sum_prev_stages(levy_gs, b_levy)
                for b_levy, levy_gs in zip(b_levy_list, levy_gs_list)
            ]
            diffusion_result = jtu.tree_map(
                lambda b_wg, *b_levy_g: sum(b_levy_g, b_wg), b_wgs, *b_levy_gs_list
            )

        # compute Σ_{j=1}^s b_j k_j
        if self.tableau.b_error is None:
            error = None
        else:
            b_err = jnp.asarray(self.tableau.b_error, dtype=dtype)
            drift_error = sum_prev_stages(tfs, b_err)
            if additive_noise:
                error = drift_error
            else:
                bw_err = jnp.asarray(self.tableau.bW_error, dtype=dtype)
                w_err = sum_prev_stages(wgs, bw_err)  # type: ignore
                b_levy_err_list = []
                if stla:
                    b_levy_err_list.append(
                        jnp.asarray(self.tableau.bH_error, dtype=dtype)
                    )
                if sttla:
                    b_levy_err_list.append(
                        jnp.asarray(self.tableau.bK_error, dtype=dtype)
                    )
                levy_err = [
                    sum_prev_stages(levy_gs, b_levy_err)
                    for b_levy_err, levy_gs in zip(b_levy_err_list, levy_gs_list)
                ]
                diffusion_error = jtu.tree_map(
                    lambda _w_err, *_levy_err: sum(_levy_err, _w_err), w_err, *levy_err
                )
                error = (drift_error**ω + diffusion_error**ω).ω

        # y1 = y0 + (Σ_{i=1}^{s} b_j * h*k_j) + g * (b_w * ΔW + bH * ΔH)

        drift_result = sum_prev_stages(tfs, b_sol)

        y1 = (y0**ω + drift_result**ω + diffusion_result**ω).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, error, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: PyTree,
    ) -> VF:
        return terms.vf(t0, y0, args)
