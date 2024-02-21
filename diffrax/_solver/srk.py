import abc
from abc import ABC
from dataclasses import dataclass
from typing import ClassVar, Optional, TYPE_CHECKING, Union
from typing_extensions import TypeAlias

import equinox as eqx
import equinox.internal as eqxi
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω
from jaxtyping import Array, Float, PyTree

from .._brownian import AbstractBrownianPath
from .._custom_types import (
    BoolScalarLike,
    BrownianIncrement,
    DenseInfo,
    FloatScalarLike,
    IntScalarLike,
    RealScalarLike,
    SpaceTimeLevyArea,
    SpaceTimeTimeLevyArea,
    VF,
    Y,
)
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm, MultiTerm, ODETerm
from .base import AbstractSolver


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar

_ErrorEstimate: TypeAlias = Optional[Y]
_SolverState: TypeAlias = None
_CarryType: TypeAlias = tuple[PyTree[Array], PyTree[Array], PyTree[Array]]


@dataclass(frozen=True)
class AbstractBrownianCoeffs(ABC):
    """Abstract base class for noise coefficients."""

    b_w: Union[FloatScalarLike, Float[np.ndarray, " s"]]
    a_w: Union[Float[np.ndarray, " s"], tuple[np.ndarray, ...]]

    @abc.abstractmethod
    def check(self) -> int:
        ...


class _AbstractSpaceTimeLACoeffs(AbstractBrownianCoeffs, ABC):
    b_hh: Union[FloatScalarLike, Float[np.ndarray, " s"]]
    a_hh: Union[Float[np.ndarray, " s"], tuple[np.ndarray, ...]]


class _AbstractSpaceTimeTimeLACoeffs(_AbstractSpaceTimeLACoeffs, ABC):
    b_kk: Union[FloatScalarLike, Float[np.ndarray, " s"]]
    a_kk: Union[Float[np.ndarray, " s"], tuple[np.ndarray, ...]]


@dataclass(frozen=True)
class AdditiveBrownianCoeffs(AbstractBrownianCoeffs):
    """
    Class representing the noise coefficients for additive Brownian motion.

    **Arguments:**

    - `b_w`: The coefficient for the Brownian increment when computing the output.
    - `a_w`: The coefficients in front of the Brownian increment at each stage.
    """

    b_w: FloatScalarLike
    # assuming SDE has additive noise we only need a 1-dimensional array
    # of length s for the coefficients in front of the Brownian increment
    # and/or Lévy areas (where s is the number of stages of the solver).
    # This is the equivalent of the matrix a for the Brownian motion and
    # its Lévy areas.
    a_w: Float[np.ndarray, " s"]

    def check(self):
        assert self.a_w.ndim == 1
        return self.a_w.shape[0]


@dataclass(frozen=True)
class AdditiveSpaceTimeLACoeffs(AdditiveBrownianCoeffs, _AbstractSpaceTimeLACoeffs):
    """
    Class representing the noise coefficients for additive space-time Lévy area.

    **Arguments:**
    - `b_w`: The coefficient for the Brownian increment when computing the output.
    - `a_w`: The coefficients in front of the Brownian increment at each stage.
    - `b_hh`: The coefficient for the space-time Lévy area when computing the output.
    - `a_hh`: The coefficients in front of the space-time Lévy area at each stage.
    """

    b_hh: FloatScalarLike
    a_hh: Float[np.ndarray, " s"]

    def check(self):
        assert self.a_w.ndim == 1
        assert self.a_hh.ndim == 1
        assert self.a_w.shape == self.a_hh.shape
        return self.a_w.shape[0]  # return the number of stages


@dataclass(frozen=True)
class AdditiveSpaceTimeTimeLACoeffs(
    AdditiveSpaceTimeLACoeffs, _AbstractSpaceTimeTimeLACoeffs
):
    """
    Class representing the noise coefficients for additive space-time-time Lévy area.

    **Arguments:**
    - `b_w`: The coefficient for the Brownian increment when computing the output.
    - `a_w`: The coefficients in front of the Brownian increment at each stage.
    - `b_hh`: The coefficient for the space-time Lévy area when computing the output.
    - `a_hh`: The coefficients in front of the space-time Lévy area at each stage.
    - `b_kk`: The coefficient for the space-time-time Lévy area in the output.
    - `a_kk`: The coefficients in front of the space-time-time Lévy area at each stage.
    """

    b_kk: FloatScalarLike
    a_kk: Float[np.ndarray, " s"]

    def check(self):
        super().check()
        assert self.a_kk.ndim == 1
        assert self.a_kk.shape == self.a_w.shape
        return self.a_w.shape[0]  # return the number of stages


@dataclass(frozen=True)
class GeneralBrownianCoeffs(AbstractBrownianCoeffs):
    """Class representing the noise coefficients for general (non-additive) noise.

    **Arguments:**

    - `b_w`: The coefficient for the Brownian increment when computing the output.
    - `a_w`: The coefficients in front of the Brownian increment at each stage.
    - `b_w_error`: The coefficient for the Brownian increment when computing the
        error estimate.
    """

    b_w: Float[np.ndarray, " s"]
    # If the SDE has non-additive noise, we need an equivalent of the
    # matrix a, one for the Brownian motion and one for each type of
    # Lévy area.
    a_w: tuple[np.ndarray, ...]
    b_error: Optional[Float[np.ndarray, " s"]]

    @property
    def b_w_error(self):
        return self.b_error

    def check(self):
        assert self.b_w.ndim == 1
        assert all((i + 1,) == a_i.shape for i, a_i in enumerate(self.a_w))
        assert self.b_w.shape[0] == len(self.a_w) + 1
        if self.b_error is not None:
            assert self.b_w_error.ndim == 1  # pyright: ignore
            assert self.b_w_error.shape == self.b_w.shape  # pyright: ignore
        return self.b_w.shape[0]  # return the number of stages


@dataclass(frozen=True)
class GeneralSpaceTimeLACoeffs(GeneralBrownianCoeffs, _AbstractSpaceTimeLACoeffs):
    """
    Class representing the noise coefficients for general space-time Lévy area.

    **Arguments:**
    - `b_w`: The coefficient for the Brownian increment when computing the output.
    - `a_w`: The coefficients in front of the Brownian increment at each stage.
    - `b_hh`: The coefficient for the space-time Lévy area when computing the output.
    - `a_hh`: The coefficients in front of the space-time Lévy area at each stage.
    - `b_error`: Tuple of (b_w_error, b_hh_error), which are the
        coefficients for the Brownian increment and the space-time Levy area resp.
         when computing the error estimate.
    """

    b_hh: Float[np.ndarray, " s"]
    a_hh: tuple[np.ndarray, ...]
    b_error: Optional[tuple[Float[np.ndarray, " s"], Float[np.ndarray, " s"]]]

    @property
    def b_w_error(self):
        return self.b_error[0] if self.b_error is not None else None

    @property
    def b_hh_error(self):
        return self.b_error[1] if self.b_error is not None else None

    def check(self):
        super().check()
        assert self.b_hh.ndim == 1
        assert self.b_hh.shape == self.b_w.shape
        assert all((i + 1,) == a_i.shape for i, a_i in enumerate(self.a_hh))
        assert self.b_hh.shape[0] == len(self.a_hh) + 1
        if self.b_error is not None:
            assert self.b_hh_error.ndim == 1  # pyright: ignore
            assert self.b_hh_error.shape == self.b_w.shape  # pyright: ignore
        return self.b_w.shape[0]


@dataclass(frozen=True)
class GeneralSpaceTimeTimeLACoeffs(
    GeneralSpaceTimeLACoeffs, _AbstractSpaceTimeTimeLACoeffs
):
    """
    Class representing the noise coefficients for general space-time-time Lévy area.

    **Arguments:**
    - `b_w`: The coefficient for the Brownian increment when computing the output.
    - `a_w`: The coefficients in front of the Brownian increment at each stage.
    - `b_hh`: The coefficient for the space-time Lévy area when computing the output.
    - `a_hh`: The coefficients in front of the space-time Lévy area at each stage.
    - `b_kk`: The coefficient for the space-time-time Lévy area in the output.
    - `a_kk`: The coefficients in front of the space-time-time Lévy area at each stage.
    - `b_error`: Tuple of (b_w_error, b_hh_error, b_kk_error), which are the
        coefficients for the Brownian increment, the space-time Levy area and the
        space-time-time Levy area resp. when computing the error estimate.
    """

    b_kk: Float[np.ndarray, " s"]
    a_kk: tuple[np.ndarray, ...]
    b_error: Optional[
        tuple[Float[np.ndarray, " s"], Float[np.ndarray, " s"], Float[np.ndarray, " s"]]
    ]

    @property
    def b_kk_error(self):
        return self.b_error[2] if self.b_error is not None else None

    def check(self):
        super().check()
        assert self.b_kk.ndim == 1
        assert self.b_kk.shape == self.b_w.shape
        assert all((i + 1,) == a_i.shape for i, a_i in enumerate(self.a_kk))
        assert self.b_kk.shape[0] == len(self.a_kk) + 1
        if self.b_error is not None:
            assert self.b_kk_error.ndim == 1  # pyright: ignore
            assert self.b_kk_error.shape == self.b_w.shape  # pyright: ignore
        return self.b_w.shape[0]


@dataclass(frozen=True)
class StochasticButcherTableau:
    """A Butcher Tableau for Stochastic Runge-Kutta methods."""

    # Only supports explicit SRK so far
    c: np.ndarray
    b_sol: np.ndarray
    b_error: Optional[np.ndarray]
    a: list[np.ndarray]

    # Coefficients for the Brownian increment
    cfs_bm: AbstractBrownianCoeffs

    # For some stages we may not need to evaluate the vector field for both
    # the drift and the diffusion. This avoids unnecessary computations.
    ignore_stage_f: Optional[np.ndarray] = None
    ignore_stage_g: Optional[np.ndarray] = None

    @property
    def additive_noise(self):
        return isinstance(self.cfs_bm, AdditiveBrownianCoeffs)

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

        assert np.allclose(sum(self.b_sol), 1.0)

        assert self.cfs_bm.check() == self.b_sol.shape[0]

        if self.b_error is not None and isinstance(self.cfs_bm, GeneralBrownianCoeffs):
            assert self.cfs_bm.b_error is not None

        if self.ignore_stage_f is not None:
            assert len(self.ignore_stage_f) == len(self.b_sol)
        if self.ignore_stage_g is not None:
            assert len(self.ignore_stage_g) == len(self.b_sol)


StochasticButcherTableau.__init__.__doc__ = """**Arguments:**

Let `s` denote the number of stages of the solver.

- `s`: The number of stages of the solver.
- `a`: The lower triangle (without the diagonal) of the Butcher tableau. Should
    be a tuple of NumPy arrays, corresponding to the rows of this lower triangle. The
    first array should be of shape `(1,)`. Each subsequent array should
    be of shape `(2,)`, `(3,)` etc. The final array should have shape `(s - 1,)`.
- `b_sol`: The linear combination of stages to take to produce the output at each step.
    Should be a NumPy array of shape `(s,)`.
- `b_error`: The linear combination of stages to take to produce the error estimate at
    each step. Should be a NumPy array of shape `(s,)`. Note that this is *not*
    differenced against `b_sol` prior to evaluation. (i.e. `b_error` gives the linear
    combination for producing the error estimate directly, not for producing some
    alternate solution that is compared against the main solution).
- `c`: The time increments used in the Butcher tableau.
    Should be a NumPy array of shape `(s-1,)`, as the first stage has time increment 0.
- `cfs_bm`: An instance of a subclass of `_AbstractBrownianCoeffs` representing
    the coefficients for the Brownian increment and possibly its Levy areas.
"""


class AbstractSRK(AbstractSolver[_SolverState]):
    r"""A general Stochastic Runge-Kutta method.

    The second term in the MultiTerm must be a `ControlTerm` with
    `control=VirtualBrownianTree`. Depending on the Butcher tableau, the
    `VirtualBrownianTree` may need to be initialised with 'levy_area="space-time"'
    or 'levy_area="space-time-time"'.

    Given the Stratonovich SDE
    $dy(t) = f(t, y(t)) dt + g(t, y(t)) \circ dw(t)$

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
    independent of y (but can still depend on t), then the SDE can be written as
    $dy(t) = f(t, y(t)) dt + g(t) \, dw(t)$, and we can simplify the above to

    $y_{n+1} = y_n + h \Big(\sum_{j=1}^s b_j k_j \Big) + g(t_n) \, (b^W
    \, W_n + b^H \, H_n)$

    $f_j = f(t_n + c_j h , z_j)$

    $z_j = y_n + h \Big(\sum_{i=1}^{j-1} a_{j,i} f_i \Big) + g(t_n)
    \, (a^W_j W_n + a^H_j H_n)$

    When g depends on t, we need to add a correction term to $y_{n+1}$ of
    the form $(g(t_{n+1}) - g(t_n)) \, (1/2 W_n - H_n)$.

    The coefficients are provided in the `StochasticButcherTableau`.
    In particular the coefficients b^W, and a^W are provided in `tableau.cfs_bm`,
    as well as b^H, a^H, b^K, and a^K if needed.
    """

    term_structure: ClassVar = MultiTerm[tuple[ODETerm, AbstractTerm]]
    interpolation_cls = LocalLinearInterpolation
    tableau: AbstractClassVar[StochasticButcherTableau]

    # Indicates the type of Levy area used by the solver.
    # The BM must generate at least this type of Levy area, but can generate
    # more. E.g. if the solver uses space-time Levy area, then the BM generates
    # space-time-time Levy area as well that is fine. The other way around would
    # not work. This is mostly an easily readable indicator so that methods know
    # what kind of BM to use.
    @property
    def minimal_levy_area(self) -> type[BrownianIncrement]:
        if isinstance(
            self.tableau.cfs_bm,
            (AdditiveSpaceTimeTimeLACoeffs, GeneralSpaceTimeTimeLACoeffs),
        ):
            return SpaceTimeTimeLevyArea
        elif isinstance(
            self.tableau.cfs_bm, (AdditiveSpaceTimeLACoeffs, GeneralSpaceTimeLACoeffs)
        ):
            return SpaceTimeLevyArea
        else:
            return BrownianIncrement

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

        is_bm = lambda x: isinstance(x, AbstractBrownianPath)
        leaves = jtu.tree_leaves(diffusion, is_leaf=is_bm)
        paths = [x for x in leaves if is_bm(x)]
        for path in paths:
            assert issubclass(path.levy_area, self.minimal_levy_area), (
                f"The diffusion term should be controlled by a Brownian path,"
                f" initialised with"
                f"`levy_area='{self.minimal_levy_area.__name__}'` or a subclass of it."
                f"Got {path.levy_area.__name__}."
            )

        if self.tableau.additive_noise:
            # check that the vector field of the diffusion term does not depend on y
            ones_like_y0 = jtu.tree_map(jnp.ones_like, y0)
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
        if self.tableau.ignore_stage_f is not None:
            ignore_stage_f = jnp.array(self.tableau.ignore_stage_f)
        else:
            ignore_stage_f = None
        if self.tableau.ignore_stage_g is not None:
            ignore_stage_g = jnp.array(self.tableau.ignore_stage_g)
        else:
            ignore_stage_g = None

        # time increment
        h = t1 - t0

        # First the drift related stuff
        a = self._embed_a_lower(self.tableau.a, dtype)
        c = jnp.asarray(np.insert(self.tableau.c, 0, 0.0), dtype=dtype)
        b_sol = jnp.asarray(self.tableau.b_sol, dtype=dtype)

        def make_zeros():
            def make_zeros_aux(leaf):
                return jnp.zeros((len(b_sol),) + leaf.shape, dtype=leaf.dtype)

            return jtu.tree_map(make_zeros_aux, y0)

        # h_kfs is a PyTree of the same shape as y0, except that the arrays inside
        # have an additional batch dimension of size len(b_sol) (i.e. num stages)
        # This will be one of the entries of the carry of lax.scan. In each stage
        # one of the zeros will get replaced by the value of
        # h_kf_j := h * f(t0 + c_j * h, z_j) where z_j is the jth stage of the SRK.
        # The name h_kf_j is because it refers to the values of f (as opposed to g)
        # at stage j, which has already been multiplied by the time increment h.
        h_kfs = make_zeros()

        # Now the diffusion related stuff
        # Brownian increment (and space-time Lévy area)
        bm_inc = diffusion.contr(t0, t1, use_levy=True)
        assert isinstance(bm_inc, self.minimal_levy_area), (
            f"The diffusion term should be controlled by a Brownian path,"
            f" initialised with"
            f"`levy_area='{self.minimal_levy_area.__name__}'` or a subclass of it."
            f"Got {bm_inc.__name__}."
        )
        w = bm_inc.W

        # b looks similar regardless of whether we have additive noise or not
        cfs_bm = self.tableau.cfs_bm
        b_w = jnp.asarray(cfs_bm.b_w, dtype=dtype)
        b_levy_list = []

        levy_areas = []
        if isinstance(cfs_bm, _AbstractSpaceTimeLACoeffs):  # space-time Levy area
            assert isinstance(bm_inc, SpaceTimeLevyArea)
            levy_areas.append(bm_inc.H)
            b_levy_list.append(jnp.asarray(cfs_bm.b_hh, dtype=dtype))

            if isinstance(
                cfs_bm, _AbstractSpaceTimeTimeLACoeffs
            ):  # space-time-time Levy area
                assert isinstance(bm_inc, SpaceTimeTimeLevyArea)
                levy_areas.append(bm_inc.K)
                b_levy_list.append(jnp.asarray(cfs_bm.b_kk, dtype=dtype))

        def add_levy_to_w(_cw, *_c_levy):
            def aux_add_levy(w_leaf, *levy_leaves):
                return _cw * w_leaf + sum(
                    _c * _leaf for _c, _leaf in zip(_c_levy, levy_leaves)
                )

            return aux_add_levy

        a_levy = []  # if noise is additive this is [cH, cK] (if those entries exist)
        # otherwise this is [aH, aK] (if those entries exist)

        levylist_kgs = []  # will contain levy * g(t0 + c_j * h, z_j) for each stage j
        # where levy is either H or K (if those entries exist)
        # this is similar to h_kfs or w_kgs, but for the Levy area(s)

        if isinstance(cfs_bm, AdditiveBrownianCoeffs):  # additive noise
            # compute g once since it is constant
            def _comp_g(_, _t):
                return None, diffusion.vf(_t, y0, args)

            _, (g0, g1) = lax.scan(_comp_g, None, jnp.array([t0, t1]), length=2)
            g_delta = (0.5 * (g1**ω - g0**ω)).ω
            w_kgs = diffusion.prod(g0, w)
            a_w = jnp.asarray(cfs_bm.a_w, dtype=dtype)

            if isinstance(cfs_bm, _AbstractSpaceTimeLACoeffs):  # space-time Levy area
                assert isinstance(bm_inc, SpaceTimeLevyArea)
                levylist_kgs.append(diffusion.prod(g0, bm_inc.H))
                a_levy.append(jnp.asarray(cfs_bm.a_hh, dtype=dtype))

            if isinstance(
                cfs_bm, _AbstractSpaceTimeTimeLACoeffs
            ):  # space-time-time Levy area
                assert isinstance(bm_inc, SpaceTimeTimeLevyArea)
                levylist_kgs.append(diffusion.prod(g0, bm_inc.K))
                a_levy.append(jnp.asarray(cfs_bm.a_kk, dtype=dtype))

            carry: _CarryType = (h_kfs, None, None)

        else:  # general (non-additive) noise
            g_delta = None  # so pyright doesn't complain

            # g is not constant, so we need to compute it at each stage
            # we will carry the value of W * g(t0 + c_j * h, z_j)
            # Since the carry of lax.scan needs to have constant shape,
            # we initialise a list of zeros of the same shape as y0, which will get
            # filled with the values of W * g(t0 + c_j * h, z_j) at each stage
            w_kgs = make_zeros()
            a_w = self._embed_a_lower(cfs_bm.a_w, dtype)

            # do the same for each type of Levy area
            if isinstance(cfs_bm, _AbstractSpaceTimeLACoeffs):  # space-time Levy area
                levylist_kgs.append(make_zeros())
                a_levy.append(self._embed_a_lower(cfs_bm.a_hh, dtype))
            if isinstance(
                cfs_bm, _AbstractSpaceTimeTimeLACoeffs
            ):  # space-time-time Levy area
                levylist_kgs.append(make_zeros())
                a_levy.append(self._embed_a_lower(cfs_bm.a_kk, dtype))

            carry: _CarryType = (h_kfs, w_kgs, levylist_kgs)

        stage_nums = jnp.arange(len(self.tableau.b_sol))

        scan_inputs = (stage_nums, a, c, a_w, a_levy)

        def sum_prev_stages(_stage_out_buff, _a_j):
            # Unwrap the buffer
            _stage_out_view = jtu.tree_map(
                lambda _, _buff: _buff[...], y0, _stage_out_buff
            )
            # Sum up the previous stages weighted by the coefficients in the tableau
            return jtu.tree_map(
                lambda lf: jnp.tensordot(_a_j, lf, axes=1), _stage_out_view
            )

        def insert_jth_stage(results, k_j, j):
            # Insert the result of the jth stage into the buffer
            return jtu.tree_map(
                lambda k_j_leaf, res_leaf: res_leaf.at[j].set(k_j_leaf), k_j, results
            )

        def stage(
            _carry: _CarryType,
            x: tuple[IntScalarLike, Array, Array, Array, list[Array]],
        ):
            # Represents the jth stage of the SRK.

            j, a_j, c_j, a_w_j, a_levy_list_j = x
            # a_levy_list_j = [aH_j, aK_j] (if those entries exist) where
            # aH_j is the row in the aH matrix corresponding to stage j
            # same for aK_j, but for space-time-time Lévy area K.
            _h_kfs, _w_kgs, _levylist_kgs = _carry

            if isinstance(cfs_bm, AdditiveBrownianCoeffs):  # additive noise
                # carry = (_h_kfs, None, None) where
                # _h_kfs = Array[h_kf_1, h_kf_2, ..., hk_{j-1}, 0, 0, ..., 0]
                # h_kf_i = drift.vf_prod(t0 + c_i*h, y_i, args, h)
                assert _w_kgs is None and _levylist_kgs is None
                assert isinstance(levylist_kgs, list)
                _diffusion_result = jtu.tree_map(
                    add_levy_to_w(a_w_j, *a_levy_list_j),
                    w_kgs,
                    *levylist_kgs,
                )
            else:
                # carry = (_h_kfs, _w_kgs, _levylist_kgs) where
                # _h_kfs = Array[h_kf_1, h_kf_2, ..., h_kf_{j-1}, 0, 0, ..., 0]
                # _w_kgs = Array[w_kg_1, w_kg_2, ..., w_kg_{j-1}, 0, 0, ..., 0]
                # _levylist_kgs = [H_gs, K_gs] (if those entries exist) where
                # H_gs = Array[Hg1, Hg2, ..., Hg{j-1}, 0, 0, ..., 0]
                # K_gs = Array[Kg1, Kg2, ..., Kg{j-1}, 0, 0, ..., 0]
                # h_kf_i = drift.vf_prod(t0 + c_i*h, y_i, args, h)
                # w_kg_i = diffusion.vf_prod(t0 + c_i*h, y_i, args, w)
                w_kg_sum = sum_prev_stages(_w_kgs, a_w_j)
                levy_sum_list = [
                    sum_prev_stages(levy_gs, a_levy_j)
                    for a_levy_j, levy_gs in zip(a_levy_list_j, _levylist_kgs)
                ]

                _diffusion_result = jtu.tree_map(
                    lambda _w_kg, *_levy_g: sum(_levy_g, _w_kg),
                    w_kg_sum,
                    *levy_sum_list,
                )

            # compute Σ_{i=1}^{j-1} a_j_i h_kf_i
            _drift_result = sum_prev_stages(_h_kfs, a_j)

            # z_j = y_0 + h (Σ_{i=1}^{j-1} a_j_i k_i) + g * (a_w_j * ΔW + cH_j * ΔH)
            z_j = (y0**ω + _drift_result**ω + _diffusion_result**ω).ω

            def compute_and_insert_kf_j(_h_kfs_in):
                h_kf_j = drift.vf_prod(t0 + c_j * h, z_j, args, h)
                return insert_jth_stage(_h_kfs_in, h_kf_j, j)

            if ignore_stage_f is None:
                _h_kfs = compute_and_insert_kf_j(_h_kfs)
            else:
                drift_pred = jnp.logical_not(ignore_stage_f[j])
                _h_kfs = lax.cond(
                    eqxi.nonbatchable(drift_pred),
                    compute_and_insert_kf_j,
                    lambda what: what,
                    _h_kfs,
                )

            if isinstance(cfs_bm, AdditiveBrownianCoeffs):  # additive noise
                return (_h_kfs, None, None), None

            def compute_and_insert_kg_j(_w_kgs_in, _levylist_kgs_in):
                _w_kg_j = diffusion.vf_prod(t0 + c_j * h, z_j, args, w)
                new_w_kgs = insert_jth_stage(_w_kgs_in, _w_kg_j, j)

                _levylist_kg_j = [
                    diffusion.vf_prod(t0 + c_j * h, z_j, args, levy)
                    for levy in levy_areas
                ]
                new_levylist_kgs = insert_jth_stage(_levylist_kgs_in, _levylist_kg_j, j)
                return new_w_kgs, new_levylist_kgs

            if ignore_stage_g is None:
                _w_kgs, _levylist_kgs = compute_and_insert_kg_j(_w_kgs, _levylist_kgs)
            else:
                diffusion_pred = jnp.logical_not(ignore_stage_g[j])
                _w_kgs, _levylist_kgs = lax.cond(
                    eqxi.nonbatchable(diffusion_pred),
                    compute_and_insert_kg_j,
                    lambda x, y: (x, y),
                    _w_kgs,
                    _levylist_kgs,
                )

            return (_h_kfs, _w_kgs, _levylist_kgs), None

        scan_out = eqxi.scan(
            stage,
            carry,
            scan_inputs,
            len(b_sol),
            buffers=lambda x: x,
            kind="checkpointed",
            checkpoints="all",
        )

        if isinstance(cfs_bm, AdditiveBrownianCoeffs):
            # output of lax.scan is ((num_stages, _h_kfs), None)
            (h_kfs, _, _), _ = scan_out
            diffusion_result = jtu.tree_map(
                add_levy_to_w(b_w, *b_levy_list),
                w_kgs,
                *levylist_kgs,
            )

            # In the additive noise case (i.e. when g is independent of y),
            # we still need a correction term in case the diffusion vector field
            # g depends on t. This term is of the form $(g1 - g0) * (0.5*W_n - H_n)$.
            if isinstance(cfs_bm, _AbstractSpaceTimeLACoeffs):  # space-time Levy area
                assert isinstance(bm_inc, SpaceTimeLevyArea)
                time_var_contr = (bm_inc.W**ω - 2.0 * bm_inc.H**ω).ω
                time_var_term = diffusion.prod(g_delta, time_var_contr)
            else:
                time_var_term = diffusion.prod(g_delta, bm_inc.W)
            diffusion_result = (diffusion_result**ω + time_var_term**ω).ω

        else:
            # output of lax.scan is ((num_stages, _h_kfs, _w_kgs, _levylist_kgs), None)
            (h_kfs, w_kgs, levylist_kgs), _ = scan_out
            b_w_kgs = sum_prev_stages(w_kgs, b_w)
            b_levylist_kgs = [
                sum_prev_stages(levy_gs, b_levy)
                for b_levy, levy_gs in zip(b_levy_list, levylist_kgs)
            ]
            diffusion_result = jtu.tree_map(
                lambda b_w_kg, *b_levy_g: sum(b_levy_g, b_w_kg),
                b_w_kgs,
                *b_levylist_kgs,
            )

        # compute Σ_{j=1}^s b_j k_j
        if self.tableau.b_error is None:
            error = None
        else:
            b_err = jnp.asarray(self.tableau.b_error, dtype=dtype)
            drift_error = sum_prev_stages(h_kfs, b_err)
            if isinstance(cfs_bm, AdditiveBrownianCoeffs):
                error = drift_error
            else:
                assert isinstance(cfs_bm, GeneralBrownianCoeffs)
                bw_err = jnp.asarray(cfs_bm.b_w_error, dtype=dtype)
                w_err = sum_prev_stages(w_kgs, bw_err)
                b_levy_err_list = []
                if isinstance(cfs_bm, GeneralSpaceTimeLACoeffs):
                    b_levy_err_list.append(jnp.asarray(cfs_bm.b_hh_error, dtype=dtype))
                if isinstance(cfs_bm, GeneralSpaceTimeTimeLACoeffs):
                    b_levy_err_list.append(jnp.asarray(cfs_bm.b_kk_error, dtype=dtype))
                levy_err = [
                    sum_prev_stages(levy_gs, b_levy_err)
                    for b_levy_err, levy_gs in zip(b_levy_err_list, levylist_kgs)
                ]
                diffusion_error = jtu.tree_map(
                    lambda _w_err, *_levy_err: sum(_levy_err, _w_err), w_err, *levy_err
                )
                error = (drift_error**ω + diffusion_error**ω).ω

        # y1 = y0 + (Σ_{i=1}^{s} b_j * h*k_j) + g * (b_w * ΔW + b_H * ΔH)

        drift_result = sum_prev_stages(h_kfs, b_sol)

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
