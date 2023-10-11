from dataclasses import field
from typing import Optional, Tuple, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu

from ..custom_types import Array, PyTree, Scalar
from ..misc import is_tuple_of_ints, split_by_tree
from .base import AbstractSTLAPath


#
# The notation here comes from section 5.5.2 of
#
# @phdthesis{kidger2021on,
#     title={{O}n {N}eural {D}ifferential {E}quations},
#     author={Patrick Kidger},
#     year={2021},
#     school={University of Oxford},
# }
#


class _State(eqx.Module):
    s: Scalar
    t: Scalar
    u: Scalar
    w_s: Scalar
    w_t: Scalar
    w_u: Scalar
    la_s: Scalar
    la_t: Scalar
    la_u: Scalar
    key: "jax.random.PRNGKey"


class VirtualSTLATree(AbstractSTLAPath):
    """Brownian simulation that discretises the interval `[t0, t1]` to tolerance `tol`,
    and is piecewise quadratic at that discretisation.

    ??? cite "Reference"

        ```bibtex
        @article{li2020scalable,
          title={Scalable gradients for stochastic differential equations},
          author={Li, Xuechen and Wong, Ting-Kam Leonard and Chen, Ricky T. Q. and
                  Duvenaud, David},
          journal={International Conference on Artificial Intelligence and Statistics},
          year={2020}
        }
        ```

        (The implementation here is a slight improvement on the reference
        implementation, by being piecwise quadratic rather than piecewise linear. This
        corrects a small bias in the generated samples.)
    """

    t0: Scalar = field(init=True)
    t1: Scalar = field(init=True)  # override init=False in AbstractPath
    tol: Scalar
    shape: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    key: "jax.random.PRNGKey"  # noqa: F821

    def __init__(
        self,
        t0: Scalar,
        t1: Scalar,
        tol: Scalar,
        shape: Union[Tuple[int, ...], PyTree[jax.ShapeDtypeStruct]],
        key: "jax.random.PRNGKey",
    ):
        self.t0 = t0
        self.t1 = t1
        self.tol = tol
        self.shape = (
            jax.ShapeDtypeStruct(shape, jax.dtypes.canonicalize_dtype(None))
            if is_tuple_of_ints(shape)
            else shape
        )
        if any(
            not jnp.issubdtype(x.dtype, jnp.inexact)
            for x in jtu.tree_leaves(self.shape)
        ):
            raise ValueError(
                "VirtualBrownianTree dtypes all have to be floating-point."
            )
        self.key = split_by_tree(key, self.shape)

    @eqx.filter_jit
    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True
    ) -> PyTree[Array]:
        del left
        t0 = eqxi.nondifferentiable(t0, name="t0")
        if t1 is None:
            return self._evaluate(t0)
        else:
            t1 = eqxi.nondifferentiable(t1, name="t1")
            # return _evaluate(t1) - _evaluate(t0)
            # TODO: this doesn't work for STLA yet
            return jtu.tree_map(
                lambda x, y: x - y,
                self._evaluate(t1),
                self._evaluate(t0),
            )

    def _evaluate(self, τ: Scalar) -> PyTree[Array]:
        """Maps the _evaluate_leaf function at time τ using self.key onto self.shape
        Args:
            τ:
        """
        # TODO: how should STLA and w be zipped? Inside the PyTree, or as separate PyTrees of the same shape?
        map_func = lambda key, shape: self._evaluate_leaf(key, τ, shape)
        return jtu.tree_map(map_func, self.key, self.shape)

    def _midpoint_bridge(self, s, u, w_s, w_u, la_s, la_u, key, shape, dtype):
        """For t = (s+u)/2 evaluates w_t and la_t conditional on w_s, w_u, la_s, and la_u
        Args:
            s: start time
            u: end time
            w_s: value of BM at s
            w_u: value of BM at u
            la_s: space-time Levy integral at s
            la_u: space-time Levy integral at u
            key:
            shape:
            dtype:
        """
        h = u - s

        n_key, z_key = jrandom.split(key, 2)
        n = jrandom.normal(n_key, shape, dtype) * jnp.sqrt((1 / 16) * h)
        z = jrandom.normal(z_key, shape, dtype) * jnp.sqrt((1 / 12) * h)

        w_t = 3/(2*h) * (la_u - la_s) - 1/4 * (w_u + w_s) + z
        la_t = 0.5 * (la_u + la_s) + h/4 * (n - w_u + w_s)
        return w_t, la_t

    def _evaluate_leaf(
        self,
        key,
        τ: Scalar,
        shape: jax.ShapeDtypeStruct,
    ) -> (Array, Array):
        shape, dtype = shape.shape, shape.dtype

        # reshuffle t0 and t1 so that t0 < t1
        cond = self.t0 < self.t1
        t0 = jnp.where(cond, self.t0, self.t1).astype(dtype)
        t1 = jnp.where(cond, self.t1, self.t0).astype(dtype)

        t0 = eqxi.error_if(
            t0,
            τ < t0,
            "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1].",
        )
        t1 = eqxi.error_if(
            t1,
            τ > t1,
            "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1].",
        )
        # Clip because otherwise the while loop below won't terminate, and the above
        # errors are only raised after everything has finished executing.
        τ = jnp.clip(τ, t0, t1).astype(dtype)

        key, init_key_w, init_key_la = jrandom.split(key, 3)
        thalf = t0 + 0.5 * (t1 - t0)
        w_t1 = jrandom.normal(init_key_w, shape, dtype) * jnp.sqrt(t1 - t0)

        la_std = jnp.sqrt(1/12 * jnp.power(t1 - t0, 3))
        la_mean = 0.5 * (t1 - t0) * w_t1
        la_t1 = la_std * jrandom.normal(init_key_la, shape, dtype) + la_mean
        w_thalf, la_thalf = self._midpoint_bridge(t0, t1, 0, w_t1, 0, la_t1, key, shape, dtype)
        init_state = _State(
            s=t0,
            t=thalf,
            u=t1,
            w_s=jnp.zeros_like(w_t1),
            w_t=w_thalf,
            w_u=w_t1,
            la_s = 0,
            la_t = la_thalf,
            la_u = la_t1,
            key=key,
        )

        def _cond_fun(_state):
            # Slight adaptation on the version of the algorithm given in the
            # above-referenced thesis. There the returned value is snapped to one of
            # the dyadic grid points, so they just stop once
            # jnp.abs(τ - state.s) > self.tol
            # Here, because we use quadratic splines to get better samples, we always
            # iterate down to the level of the spline.
            return (_state.u - _state.s) > self.tol

        def _body_fun(_state):
            """ Single-step of binary search for τ.
            Args:
                _state:

            Returns:

            """
            _key1, _key2 = jrandom.split(_state.key, 2)
            _cond = τ > _state.t
            _s = jnp.where(_cond, _state.t, _state.s)
            _u = jnp.where(_cond, _state.u, _state.t)
            _w_s = jnp.where(_cond, _state.w_t, _state.w_s)
            _w_u = jnp.where(_cond, _state.w_u, _state.w_t)
            _la_s = jnp.where(_cond, _state.la_t, _state.la_s)
            _la_u = jnp.where(_cond, _state.la_u, _state.la_t)
            _key = jnp.where(_cond, _key1, _key2)
            _t = _s + 0.5 * (_u - _s)
            _w_t, _la_t = self._midpoint_bridge(_s, _u, _w_s, _w_u, _la_s, _la_u, _key, shape, dtype)
            return _State(s=_s, t=_t, u=_u,
                          w_s=_w_s, w_t=_w_t, w_u=_w_u,
                          la_s = _la_s, la_t = _la_t, la_u = _la_u,
                          key=_key)

        final_state = lax.while_loop(_cond_fun, _body_fun, init_state)
        # Quadratic interpolation.
        # We have w_s, w_t, w_u available to us, and interpolate with the unique
        # parabola passing through them.
        # Why quadratic and not just "linear from w_s to w_t to w_u"? Because linear
        # only gets the conditional mean correct at every point, but not the
        # conditional variance. This means that the Virtual Brownian Tree will pass
        # statistical tests comparing w(t)|(w(s),w(u)) against the true Brownian
        # bridge. (Provided s, t, u are greater than the discretisation level `tol`.)
        # (If you just do linear then you find that the variance is *ever so slightly*
        # too small.)

        # TODO: add stla quadratic interpolation.

        s = final_state.s
        u = final_state.u
        w_s = final_state.w_s
        w_t = final_state.w_t
        w_u = final_state.w_u
        rescaled_τ = (τ - s) / (u - s)
        # Fit polynomial as usual.
        # The polynomial ax^2 + bx + c is found by solving
        # [s^2 s 1][a]   [w_s]
        # [t^2 t 1][b] = [w_t]
        # [u^2 u 1][c]   [w_u]
        #
        # `A` is the inverse of the above matrix, with s=0, t=0.5, u=1.
        A = jnp.array([[2, -4, 2], [-3, 4, -1], [1, 0, 0]])
        coeffs = jnp.tensordot(A, jnp.stack([w_s, w_t, w_u]), axes=1)
        w_τ = jnp.polyval(coeffs, rescaled_τ)

        # provisionally set the Levy area to the midpoint value instead of interpolating
        la_τ = final_state.la_t
        return w_τ, la_τ


VirtualSTLATree.__init__.__doc__ = """
**Arguments:**

- `t0`: The start of the interval the Brownian motion is defined over.
- `t1`: The start of the interval the Brownian motion is defined over.
- `tol`: The discretisation that `[t0, t1]` is discretised to.
- `shape`: Should be a PyTree of `jax.ShapeDtypeStruct`s, representing the shape, 
    dtype, and PyTree structure of the output. For simplicity, `shape` can also just 
    be a tuple of integers, describing the shape of a single JAX array. In that case
    the dtype is chosen to be the default floating-point dtype.
- `key`: A random key.

!!! info

    If using this as part of an SDE solver, and you know (or have an estimate of) the
    step sizes made in the solver, then you can optimise the computational efficiency
    of the Virtual Brownian Tree by setting `tol` to be just slightly smaller than the
    step size of the solver.

The Brownian motion is defined to equal 0 at `t0`.
"""
