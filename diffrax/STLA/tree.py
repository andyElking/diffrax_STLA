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


def is_tuple_of_arrays(obj):
    return isinstance(obj, tuple) and len(obj) == 2 and all(isinstance(x, jax.Array) for x in obj)


@jax.jit
def wh_from_wj(h: Scalar, x0: (jax.Array, jax.Array), x1: (jax.Array, jax.Array)) -> (jax.Array, jax.Array):
    w0, j0 = x0
    w1, j1 = x1
    w_01 = w1 - w0
    h = h.astype(w0.dtype)
    hh_01 = (j1 - j0) * (1 / h) - 0.5 * (w0 + w1)
    return w_01, hh_01


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
    ) -> (PyTree[Array], PyTree[Array]):
        # TODO: add an option where only W is computed, not H.
        del left
        t0 = eqxi.nondifferentiable(t0, name="t0")
        if t1 is None:
            t1 = eqxi.nondifferentiable(0, name="t1")
        else:
            t1 = eqxi.nondifferentiable(t1, name="t1")

        h = t1 - t0
        w_j_0 = self._evaluate(t0)
        w_j_1 = self._evaluate(t1)
        wh = jtu.tree_map(lambda x0, x1: wh_from_wj(h, x0, x1), w_j_0, w_j_1, is_leaf=is_tuple_of_arrays)
        w_out, hh_out = jtu.tree_transpose(
            outer_treedef=jax.tree_structure(self.shape),
            inner_treedef=jax.tree_structure((0, 0)),
            pytree_to_transpose=wh
        )
        # w_out = jtu.tree_map(get_w, x0, x1, is_leaf=lambda x: isinstance(x, _STLA_proc_value))
        # hh_out = jtu.tree_map(get_h, x0, x1, is_leaf=lambda x: isinstance(x, _STLA_proc_value))
        return w_out, hh_out

    def _evaluate(self, τ: Scalar) -> PyTree[Array]:
        """Maps the _evaluate_leaf function at time τ using self.key onto self.shape
        Args:
            τ:
        """
        map_func = lambda key, shape: self._evaluate_leaf(key, τ, shape)
        return jtu.tree_map(map_func, self.key, self.shape)

    def _brownian_arch(self, s, u, w_s, w_u, la_s, la_u, key, shape, dtype):
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
        # TODO: check for cancellation errors when applying Chen's relation
        h = u - s

        n_key, z_key = jrandom.split(key, 2)
        n = jrandom.normal(n_key, shape, dtype) * jnp.sqrt((1 / 12) * h)
        z = jrandom.normal(z_key, shape, dtype) * jnp.sqrt((1 / 16) * h)

        w_t = (3 / (2 * h)) * (la_u - la_s) - (1 / 4) * (w_u + w_s) + z
        la_t = 0.5 * (la_u + la_s) + (h / 8) * (w_s - w_u) + (h / 4) * n
        return w_t, la_t

    def _evaluate_leaf(
            self,
            key,
            r: Scalar,
            shape: jax.ShapeDtypeStruct,
    ) -> (Array, Array):
        shape, dtype = shape.shape, shape.dtype

        # reshuffle t0 and t1 so that t0 < t1
        cond = self.t0 < self.t1
        t0 = jnp.where(cond, self.t0, self.t1).astype(dtype)
        t1 = jnp.where(cond, self.t1, self.t0).astype(dtype)

        t0 = eqxi.error_if(
            t0,
            r < t0,
            "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1].",
        )
        t1 = eqxi.error_if(
            t1,
            r > t1,
            "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1].",
        )
        # Clip because otherwise the while loop below won't terminate, and the above
        # errors are only raised after everything has finished executing.
        r = jnp.clip(r, t0, t1).astype(dtype)

        key, init_key_w, init_key_la, midpoint_key = jrandom.split(key, 4)
        thalf = t0 + 0.5 * (t1 - t0)
        w_t1 = jrandom.normal(init_key_w, shape, dtype) * jnp.sqrt(t1 - t0)

        la_std = jnp.sqrt((1 / 12) * jnp.power(t1 - t0, 3))
        la_mean = 0.5 * (t1 - t0) * w_t1
        la_t1 = la_std * jrandom.normal(init_key_la, shape, dtype) + la_mean
        w_thalf, la_thalf = self._brownian_arch(t0, t1, 0, w_t1, 0, la_t1, midpoint_key, shape, dtype)
        init_state = _State(
            s=t0,
            t=thalf,
            u=t1,
            w_s=jnp.zeros_like(w_t1),
            w_t=w_thalf,
            w_u=w_t1,
            la_s=jnp.zeros_like(la_t1),
            la_t=la_thalf,
            la_u=la_t1,
            key=key,
        )

        def _cond_fun(_state):
            # Slight adaptation on the version of the algorithm given in the
            # above-referenced thesis. There the returned value is snapped to one of
            # the dyadic grid points, so they just stop once
            # jnp.abs(τ - state.s) > self.tol
            # Here, because we use quadratic splines to get better samples, we always
            # iterate down to the level of the spline.
            # return jnp.all(jnp.array([(_state.u - _state.s) > self.tol,
            #                           r < _state.u - 0.1 * self.tol,
            #                           r > _state.s + 0.1 * self.tol]))
            return (_state.u - _state.s) > self.tol

        def _body_fun(_state):
            """ Single-step of binary search for τ.
            Args:
                _state:

            Returns:

            """
            _key1, _key2 = jrandom.split(_state.key, 2)
            _cond = r > _state.t
            _s = jnp.where(_cond, _state.t, _state.s)
            _u = jnp.where(_cond, _state.u, _state.t)
            _w_s = jnp.where(_cond, _state.w_t, _state.w_s)
            _w_u = jnp.where(_cond, _state.w_u, _state.w_t)
            _la_s = jnp.where(_cond, _state.la_t, _state.la_s)
            _la_u = jnp.where(_cond, _state.la_u, _state.la_t)
            _key = jnp.where(_cond, _key1, _key2)
            _t = _s + 0.5 * (_u - _s)

            # TODO: added more key splitting for generating the midpoint. Not sure if this is required.
            _key, _midpoint_key = jrandom.split(_key, 2)
            _w_t, _la_t = self._brownian_arch(_s, _u, _w_s, _w_u, _la_s, _la_u, _midpoint_key, shape, dtype)
            return _State(s=_s, t=_t, u=_u,
                          w_s=_w_s, w_t=_w_t, w_u=_w_u,
                          la_s=_la_s, la_t=_la_t, la_u=_la_u,
                          key=_key)

        final_state = lax.while_loop(_cond_fun, _body_fun, init_state)

        # Now we check if r==s or r==u, in which case we just output the value at that endpoint.
        # Otherwise we interpolate in between.
        cancellation_err_tol = self.tol * (2 ** -3)

        def _final_interpolation(_state: _State) -> (jax.Array, jax.Array):
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

            s = _state.s
            t = _state.t
            u = _state.u
            w_s = _state.w_s
            w_t = _state.w_t
            w_u = _state.w_u
            la_s = _state.la_s
            la_t = _state.la_t
            la_u = _state.la_u

            # This is different from original quadratic interpolation
            # TODO: check if it gives the right result
            h = u - s
            w_su = w_u - w_s
            w_st = w_t - w_s
            H_su = (1 / h) * (la_u - la_s) - 0.5 * (w_s + w_u)
            H_st = (2 / h) * (la_t - la_s) - 0.5 * (w_s + w_t)
            x1 = (4 / jnp.sqrt(h)) * (w_st - 0.5 * w_su - 1.5 * H_su)
            x2 = jnp.sqrt(12 / h) * (w_st + 2 * H_st - 0.5 * w_su - 2 * H_su)
            sr = r - s
            ru = u - r
            d = jnp.sqrt(jnp.power(sr, 3) + jnp.power(ru, 3))
            a = (1 / (2 * h * d)) * jnp.power(sr, 7 / 2) * jnp.sqrt(ru)
            b = (1 / (2 * h * d)) * jnp.power(ru, 7 / 2) * jnp.sqrt(sr)
            c = (1 / (jnp.sqrt(12) * d)) * jnp.power(sr, 3 / 2) * jnp.power(ru, 3 / 2)

            w_sr = (sr / h) * w_su + 6 * sr * ru / jnp.square(h) * H_su + 2 * ((a + b) / h) * x1
            H_sr = jnp.square(sr / h) * H_su - (a / sr) * x1 + (c / sr) * x2
            w_r = w_s + w_sr
            la_r = la_s + sr * H_sr + (sr / 2) * (w_s + w_r)

            return w_r, la_r

        def _equal_to_s_cond(_state: _State):
            return r < (_state.s + cancellation_err_tol)

        def _return_s_value(_state: _State) -> (jax.Array, jax.Array):
            return _state.w_s, _state.la_s

        def _equal_to_u_cond(_state: _State) -> bool:
            return r > (_state.u - cancellation_err_tol)

        def _return_u_value(_state: _State) -> (jax.Array, jax.Array):
            return _state.w_u, _state.la_u

        def _not_equal_to_s_fun(_state: _State) -> (jax.Array, jax.Array):
            return jax.lax.cond(_equal_to_u_cond(_state), _return_u_value, _final_interpolation, _state)

        return jax.lax.cond(_equal_to_s_cond(final_state), _return_s_value, _not_equal_to_s_fun, final_state)


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
