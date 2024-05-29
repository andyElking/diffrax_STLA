from typing import cast, Optional, TypeAlias, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax.internal as lxi
import numpy as np
from jaxtyping import Array, PRNGKeyArray, PyTree
from lineax.internal import complex_to_real_dtype

from .._custom_types import (
    AbstractBrownianIncrement,
    AbstractSpaceSpaceLevyArea,
    AbstractSpaceTimeLevyArea,
    AbstractSpaceTimeTimeLevyArea,
    BrownianIncrement,
    levy_tree_transpose,
    RealScalarLike,
    SpaceSpaceLevyArea,
    SpaceTimeLevyArea,
    SpaceTimeTimeLevyArea,
)
from .._misc import (
    force_bitcast_convert_type,
    is_tuple_of_ints,
    split_by_tree,
)
from .base import AbstractBrownianPath


_AcceptedLevyAreas: TypeAlias = type[
    Union[
        BrownianIncrement,
        SpaceTimeLevyArea,
        SpaceTimeTimeLevyArea,
        SpaceSpaceLevyArea,
    ]
]


class UnsafeBrownianPath(AbstractBrownianPath):
    """Brownian simulation that is only suitable for certain cases.

    This is a very quick way to simulate Brownian motion, but can only be used when all
    of the following are true:

    1. You are using a fixed step size controller. (Not an adaptive one.)

    2. You do not need to backpropagate through the differential equation.

    3. You do not need deterministic solutions with respect to `key`. (This
       implementation will produce different results based on fluctuations in
       floating-point arithmetic.)

    Internally this operates by just sampling a fresh normal random variable over every
    interval, ignoring the correlation between samples exhibited in true Brownian
    motion. Hence the restrictions above. (They describe the general case for which the
    correlation structure isn't needed.)

    Depending on the `levy_area` argument, this can also be used to generate Levy area.
    `levy_area` can be one of the following:

    - `BrownianIncrement`: Only generate Brownian increments W.

    - `SpaceTimeLevyArea`: Generate W and the space-time Levy area H.

    - `SpaceTimeTimeLevyArea`: Generate W, H, and the space-time-time Levy area K.

    - `SpaceSpaceLevyArea`: In addition to W, H, and K, generate an approximate
    space-space Levy area A. This is done via Foster's approximation, which matches
    all the fourth cross moments of A conditional on the values of W, H, and K.

    ??? cite "Reference"

        The space-space Levy area approximation is based on Definition 7.3.5 of

        ```bibtex
        @phdthesis{foster2020a,
          publisher = {University of Oxford},
          school = {University of Oxford},
          title = {Numerical approximations for stochastic differential equations},
          author = {Foster, James M.},
          year = {2020}
        }
        ```

    """

    shape: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    levy_area: _AcceptedLevyAreas = eqx.field(static=True)
    key: PRNGKeyArray

    def __init__(
        self,
        shape: Union[tuple[int, ...], PyTree[jax.ShapeDtypeStruct]],
        key: PRNGKeyArray,
        levy_area: _AcceptedLevyAreas = BrownianIncrement,
    ):
        self.shape = (
            jax.ShapeDtypeStruct(shape, lxi.default_floating_dtype())
            if is_tuple_of_ints(shape)
            else shape
        )
        self.key = key
        self.levy_area = levy_area

        if any(
            not jnp.issubdtype(x.dtype, jnp.inexact)
            for x in jtu.tree_leaves(self.shape)
        ):
            raise ValueError("UnsafeBrownianPath dtypes all have to be floating-point.")

    @property
    def t0(self):
        return -jnp.inf

    @property
    def t1(self):
        return jnp.inf

    @eqx.filter_jit
    def evaluate(
        self,
        t0: RealScalarLike,
        t1: Optional[RealScalarLike] = None,
        left: bool = True,
        use_levy: bool = False,
    ) -> Union[PyTree[Array], AbstractBrownianIncrement]:
        del left
        if t1 is None:
            dtype = jnp.result_type(t0)
            t1 = t0
            t0 = jnp.array(0, dtype)
        else:
            with jax.numpy_dtype_promotion("standard"):
                dtype = jnp.result_type(t0, t1)
            t0 = jnp.astype(t0, dtype)
            t1 = jnp.astype(t1, dtype)
        t0 = eqxi.nondifferentiable(t0, name="t0")
        t1 = eqxi.nondifferentiable(t1, name="t1")
        t1 = cast(RealScalarLike, t1)
        t0_ = force_bitcast_convert_type(t0, jnp.int32)
        t1_ = force_bitcast_convert_type(t1, jnp.int32)
        key = jr.fold_in(self.key, t0_)
        key = jr.fold_in(key, t1_)
        key = split_by_tree(key, self.shape)
        out = jtu.tree_map(
            lambda key, shape: self._evaluate_leaf(
                t0, t1, key, shape, self.levy_area, use_levy
            ),
            key,
            self.shape,
        )
        if use_levy:
            out = levy_tree_transpose(self.shape, out)
            assert isinstance(out, self.levy_area)
        return out

    @staticmethod
    def _evaluate_leaf(
        t0: RealScalarLike,
        t1: RealScalarLike,
        key,
        shape: jax.ShapeDtypeStruct,
        levy_area: _AcceptedLevyAreas,
        use_levy: bool,
    ):
        key_w, key_hh, key_kk, key_aa = jr.split(key, 4)
        w_01 = jr.normal(key_w, shape.shape, shape.dtype)
        dt = jnp.asarray(t1 - t0, dtype=shape.dtype)
        tdtype = complex_to_real_dtype(shape.dtype)
        real_dt = jnp.asarray(dt, dtype=tdtype)
        root_dt = jnp.sqrt(dt)
        w = w_01 * root_dt
        if not use_levy:
            # We don't need to generate Levy area.
            # Due to how we split the keys `w` will not depend on `levy_area`.
            return w

        if issubclass(levy_area, AbstractSpaceTimeLevyArea):
            hh_01 = jr.normal(key_hh, shape.shape, shape.dtype) / np.sqrt(12)
            hh = hh_01 * root_dt
        else:
            assert levy_area == BrownianIncrement
            return BrownianIncrement(dt=real_dt, W=w)  # noqa

        if issubclass(levy_area, AbstractSpaceTimeTimeLevyArea):
            kk_01 = jr.normal(key_kk, shape.shape, shape.dtype) / np.sqrt(720)
            kk = kk_01 * root_dt
        else:
            assert levy_area == SpaceTimeLevyArea
            return SpaceTimeLevyArea(dt=real_dt, W=w, H=hh)  # noqa

        if issubclass(levy_area, AbstractSpaceSpaceLevyArea):
            assert levy_area == SpaceSpaceLevyArea
            assert shape.shape[-1] >= 2, (
                f"SpaceSpaceLevyArea requires the Brownian motion to have"
                f" shape (..., d), where d > 1. Got {shape.shape}."
            )
            bm_dim = shape.shape[-1]
            triu_indices = np.triu_indices(bm_dim, k=1)
            aa_01 = _4moment_levy_area(key_aa, triu_indices, w_01, hh_01, kk_01)  # noqa
            aa = dt * aa_01
            return SpaceSpaceLevyArea(dt=real_dt, W=w, H=hh, K=kk, A=aa)  # noqa
        else:
            assert levy_area == SpaceTimeTimeLevyArea
            return SpaceTimeTimeLevyArea(dt=real_dt, W=w, H=hh, K=kk)  # noqa


UnsafeBrownianPath.__init__.__doc__ = """
**Arguments:**

- `shape`: Should be a PyTree of `jax.ShapeDtypeStruct`s, representing the shape, 
    dtype, and PyTree structure of the output. For simplicity, `shape` can also just 
    be a tuple of integers, describing the shape of a single JAX array. In that case
    the dtype is chosen to be the default floating-point dtype.
- `key`: A random key.
- `levy_area`: Whether to additionally generate Lévy area. This is required by some SDE
    solvers.
"""


def _4moment_levy_area(key, triu_indices, w: Array, hh: Array, kk: Array):
    """Foster's approxiamtion of space-space Levy area  based on Definition 7.3.5 of

    ```bibtex
    @phdthesis{foster2020a,
      publisher = {University of Oxford},
      school = {University of Oxford},
      title = {Numerical approximations for stochastic differential equations},
      author = {Foster, James M.},
      year = {2020}
    }
    ```

    """
    bm_dim = w.shape[-1]
    assert w.shape == hh.shape == kk.shape
    levy_dim = int(bm_dim * (bm_dim - 1) // 2)
    levy_shape = w.shape[:-1] + (levy_dim,)

    key_exp, key_ber, key_uni, key_rad = jr.split(key, 4)

    squared_kk = jnp.square(kk)
    C = jr.exponential(key_exp, w.shape) * (8 / 15)
    c = np.sqrt(1 / 3) - (8 / 15)
    p = 21130 / 25621
    ber = jnp.astype(jr.bernoulli(key_ber, p, shape=levy_shape), w.dtype)
    uni = jr.uniform(key_uni, shape=levy_shape, minval=-np.sqrt(3), maxval=np.sqrt(3))
    rademacher = jnp.astype(jr.rademacher(key_rad, shape=levy_shape), w.dtype)

    ksi = ber * uni + (1 - ber) * rademacher

    C_plus_c = C + c
    sigma = (3 / 28) * (C_plus_c[..., triu_indices[0]] * C_plus_c[..., triu_indices[1]])

    sigma = sigma + (144 / 28) * (
        squared_kk[..., triu_indices[0]] + squared_kk[..., triu_indices[1]]
    )
    sigma = jnp.sqrt(sigma)

    w_i = w[..., triu_indices[0]]
    w_j = w[..., triu_indices[1]]
    hh_i = hh[..., triu_indices[0]]
    hh_j = hh[..., triu_indices[1]]
    kk_i = kk[..., triu_indices[0]]
    kk_j = kk[..., triu_indices[1]]

    tilde_a = ksi * sigma

    aa_out = hh_i * w_j - w_i * hh_j + 12.0 * (kk_i * hh_j - hh_i * kk_j) + tilde_a

    return aa_out
