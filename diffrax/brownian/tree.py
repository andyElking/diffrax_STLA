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
from .base import AbstractBrownianPath, LevyVal


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
    J_s: Scalar = field(default=None)
    J_t: Scalar = field(default=None)
    J_u: Scalar = field(default=None)
    key: "jax.random.PRNGKey"


class VirtualBrownianTree(AbstractBrownianPath):
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
            self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True, use_levy: bool = False
    ) -> PyTree[Array]:
        del left
        t0 = eqxi.nondifferentiable(t0, name="t0")
        if t1 is None:
            out = self._evaluate(t0)
        else:
            t1 = eqxi.nondifferentiable(t1, name="t1")

            out = jtu.tree_map(
                lambda x, y: x - y,
                self._evaluate(t1),
                self._evaluate(t0),
            )
        return LevyVal(h=t1 - t0, W=out) if use_levy else out

    def _evaluate(self, τ: Scalar) -> PyTree[Array]:
        """Maps the _evaluate_leaf function at time τ using self.key onto self.shape
        Args:
            τ:
        """
        map_func = lambda key, shape: self._evaluate_leaf(key, τ, shape)
        return jtu.tree_map(map_func, self.key, self.shape)

    def _brownian_bridge(self, s, t, u, w_s, w_u, key, shape, dtype):
        """Evaluates the BM at a time t between times s<u, for which the BM has
        already been evaluated.
        Args:
            s: start time
            t: evaluation time
            u: end time
            w_s: value of BM at s
            w_u: value of BM at u
            key:
            shape:
            dtype:
        """
        mean = w_s + (w_u - w_s) * ((t - s) / (u - s))
        var = (u - t) * (t - s) / (u - s)
        std = jnp.sqrt(var)
        return mean + std * jrandom.normal(key, shape, dtype)

    def _evaluate_leaf(
            self,
            key,
            τ: Scalar,
            shape: jax.ShapeDtypeStruct,
    ) -> Array:
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

        key, init_key = jrandom.split(key, 2)
        thalf = t0 + 0.5 * (t1 - t0)
        w_t1 = jrandom.normal(init_key, shape, dtype) * jnp.sqrt(t1 - t0)
        w_thalf = self._brownian_bridge(t0, thalf, t1, 0, w_t1, key, shape, dtype)
        init_state = _State(
            s=t0,
            t=thalf,
            u=t1,
            w_s=jnp.zeros_like(w_t1),
            w_t=w_thalf,
            w_u=w_t1,
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
            """Single-step of binary search for τ.
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
            _key = jnp.where(_cond, _key1, _key2)
            _t = _s + 0.5 * (_u - _s)
            _w_t = self._brownian_bridge(_s, _t, _u, _w_s, _w_u, _key, shape, dtype)
            return _State(s=_s, t=_t, u=_u, w_s=_w_s, w_t=_w_t, w_u=_w_u, key=_key)

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
        return jnp.polyval(coeffs, rescaled_τ)


VirtualBrownianTree.__init__.__doc__ = """
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


@jax.jit
def levy_diff(x0: LevyVal, x1: LevyVal) -> LevyVal:
    pred = x1.J is None or x0.J is None
    return jax.cond(pred, LevyVal(h=x1.h - x0.h, W=x1.W - x0.W), wj_to_wh_diff(x0, x1))


@jax.jit
def wj_to_wh_diff(x0: LevyVal, x1: LevyVal) -> LevyVal:
    h = (x1.h - x0.h).astype(x0.W.dtype)
    inverse_h = jnp.nan_to_num(1 / h)
    w_01 = x1.W - x0.W
    hh_01 = (x1.J - x0.J) * inverse_h - 0.5 * (x1.W + x0.W)
    return LevyVal(h=h, W=w_01, H=hh_01)


@jax.jit
def wj_to_wh_single(x0: LevyVal) -> LevyVal:
    jax.cond(x0.J is None, x0, LevyVal(h=x0.h, W=x0.W, H=x0.J * jnp.nan_to_num(1 / x0.h) - 0.5 * x0.W))


class VirtualLevyTree(AbstractBrownianPath):
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
    _interval_len: Scalar
    tol: Scalar = field(init=True)
    shape: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    compute_stla: bool = field(default=False, init=True)
    key: "jax.random.PRNGKey"  # noqa: F821

    def __init__(
            self,
            t0: Scalar,
            t1: Scalar,
            tol: Scalar,
            shape: Union[Tuple[int, ...], PyTree[jax.ShapeDtypeStruct]],
            compute_stla: bool,
            key: "jax.random.PRNGKey",
    ):
        self.t0 = jnp.minimum(t0, t1)
        self.t1 = jnp.maximum(t0, t1)
        self._interval_len = self.t1 - self.t0
        self.tol = tol / self._interval_len
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

    @jax.jit
    def normalise_t(self, t: Scalar):
        return (t - self.t0) / self._interval_len

    @jax.jit
    def denormalise_bm_inc(self, x: LevyVal) -> LevyVal:
        sqrt_len = jnp.sqrt(self._interval_len)

        def sqrt_mult(z):
            return z * sqrt_len

        def mult(z):
            return z * self._interval_len

        return LevyVal(h=jtu.tree_map(mult, x.h),
                       W=jtu.tree_map(sqrt_mult, x.W),
                       J=jtu.tree_map(mult, x.J),
                       H=jtu.tree_map(sqrt_mult, x.H))

    @eqx.filter_jit
    def evaluate(
            self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True, use_levy: bool = False
    ) -> LevyVal:
        # TODO: add an option where only W is computed, not H.
        del left

        def is_levy_val(obj):
            return isinstance(obj, LevyVal)

        t0 = eqxi.nondifferentiable(t0, name="t0")
        levy_0 = self._evaluate(t0)
        if t1 is not None:
            t1 = eqxi.nondifferentiable(t1, name="t1")
            levy_1 = self._evaluate(t1)
            wh = jtu.tree_map(levy_diff, levy_0, levy_1, is_leaf=is_levy_val)
        else:
            wh = jtu.tree_map(wj_to_wh_single, levy_0, is_leaf=is_levy_val)

        levy_out: LevyVal = jtu.tree_transpose(
            outer_treedef=jax.tree_structure(self.shape),
            inner_treedef=jax.tree_structure(LevyVal(h=0.0, W=0.0, J=None, H=0.0)),
            pytree_to_transpose=wh
        )
        levy_out = self.denormalise_bm_inc(levy_out)
        return levy_out if use_levy else levy_out.W

    def _evaluate(self, r: Scalar) -> PyTree[LevyVal]:
        """Maps the _evaluate_leaf function at time τ using self.key onto self.shape
        Args:
            r:
        """
        r = self.normalise_t(r)

        eqxi.error_if(
            r,
            r < 0,
            "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1].",
        )
        eqxi.error_if(
            r,
            r > 1,
            "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1].",
        )
        # Clip because otherwise the while loop below won't terminate, and the above
        # errors are only raised after everything has finished executing.
        r = jnp.clip(r, 0.0, 1.0)
        map_func = lambda key, shape: self._evaluate_leaf(key, r, shape)
        return jtu.tree_map(map_func, self.key, self.shape)

    def _brownian_arch(self, s, u, w_s, w_u, key, shape, dtype, J_s=None, J_u=None):
        """For t = (s+u)/2 evaluates w_t and J_t conditional on w_s, w_u, J_s, and J_u
        Args:
            s: start time
            u: end time
            w_s: value of BM at s
            w_u: value of BM at u
            J_s: space-time Levy integral at s
            J_u: space-time Levy integral at u
            key:
            shape:
            dtype:
        """
        # TODO: check for cancellation errors when applying Chen's relation
        h = u - s

        if (not self.compute_stla) or J_s is None or J_u is None:
            mean = w_s + 0.5 * (w_u - w_s)
            std = 0.5 * jnp.sqrt(h)
            w_t = mean + std * jrandom.normal(key, shape, dtype)
            J_t = None
        else:
            n_key, z_key = jrandom.split(key, 2)
            n = jrandom.normal(n_key, shape, dtype) * jnp.sqrt((1 / 12) * h)
            z = jrandom.normal(z_key, shape, dtype) * jnp.sqrt((1 / 16) * h)

            w_t = (3 / (2 * h)) * (J_u - J_s) - (1 / 4) * (w_u + w_s) + z
            J_t = 0.5 * (J_u + J_s) + (h / 8) * (w_s - w_u) + (h / 4) * n
        return w_t, J_t

    def _evaluate_leaf(
            self,
            key,
            r: Scalar,
            shape: jax.ShapeDtypeStruct,
    ) -> LevyVal:
        shape, dtype = shape.shape, shape.dtype

        t0 = jnp.zeros((), dtype)
        t1 = jnp.ones((), dtype)
        r = r.astype(dtype)

        key, init_key_w, init_key_la, midpoint_key = jrandom.split(key, 4)
        thalf = jnp.array(0.5, dtype)
        w_t1 = jrandom.normal(init_key_w, shape, dtype) * jnp.sqrt(t1 - t0)
        J_t1 = None
        J_s = None
        if self.compute_stla:
            J_std = jnp.sqrt((1 / 12) * jnp.power(t1 - t0, 3))
            J_mean = 0.5 * (t1 - t0) * w_t1
            J_t1 = J_std * jrandom.normal(init_key_la, shape, dtype) + J_mean
            J_s = jnp.zeros_like(J_t1)
        w_thalf, J_thalf = self._brownian_arch(t0, t1, 0, w_t1, 0, J_t1, midpoint_key, shape, dtype)
        init_state = _State(
            s=t0,
            t=thalf,
            u=t1,
            w_s=jnp.zeros_like(w_t1),
            w_t=w_thalf,
            w_u=w_t1,
            J_s=J_s,
            J_t=J_thalf,
            J_u=J_t1,
            key=key,
        )

        cancellation_err_tol = self.tol * (2 ** -5)

        def _cond_fun(_state):
            # Slight adaptation on the version of the algorithm given in the
            # above-referenced thesis. There the returned value is snapped to one of
            # the dyadic grid points, so they just stop once
            # jnp.abs(τ - state.s) > self.tol
            # Here, because we use quadratic splines to get better samples, we always
            # iterate down to the level of the spline.
            # return jnp.all(jnp.array([(_state.u - _state.s) > self.tol,
            #                           r < _state.u - cancellation_err_tol,
            #                           r > _state.s + cancellation_err_tol]))
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
            _J_s, _J_u = None, None
            if self.compute_stla:
                _J_s = jnp.where(_cond, _state.J_t, _state.J_s)
                _J_u = jnp.where(_cond, _state.J_u, _state.J_t)
            _key = jnp.where(_cond, _key1, _key2)
            _t = _s + 0.5 * (_u - _s)

            # TODO: added more key splitting for generating the midpoint. Not sure if this is required.
            _key, _midpoint_key = jrandom.split(_key, 2)
            _w_t, _J_t = self._brownian_arch(_s, _u, _w_s, _w_u, _J_s, _J_u, _midpoint_key, shape, dtype)
            return _State(s=_s, t=_t, u=_u,
                          w_s=_w_s, w_t=_w_t, w_u=_w_u,
                          J_s=_J_s, J_t=_J_t, J_u=_J_u,
                          key=_key)

        final_state = lax.while_loop(_cond_fun, _body_fun, init_state)

        # Now we check if r==s or r==u, in which case we just output the value at that endpoint.
        # Otherwise we interpolate in between.

        def _final_interpolation(_state: _State) -> (jax.Array, jax.Array):
            s = _state.s
            t = _state.t
            u = _state.u
            w_s = _state.w_s
            w_t = _state.w_t
            w_u = _state.w_u
            J_s = _state.J_s
            J_t = _state.J_t
            J_u = _state.J_u

            # This is different from original quadratic interpolation
            h = u - s
            w_su = w_u - w_s
            w_st = w_t - w_s
            sr = r - s
            ru = u - r

            if self.compute_stla:
                H_su = (1 / h) * (J_u - J_s) - 0.5 * (w_s + w_u)
                H_st = (2 / h) * (J_t - J_s) - 0.5 * (w_s + w_t)
                x1 = (4 / jnp.sqrt(h)) * (w_st - 0.5 * w_su - 1.5 * H_su)
                x2 = jnp.sqrt(12 / h) * (w_st + 2 * H_st - 0.5 * w_su - 2 * H_su)
                d = jnp.sqrt(jnp.power(sr, 3) + jnp.power(ru, 3))
                d_prime = 1 / (2 * h * d)
                a = d_prime * jnp.power(sr, 3.5) * jnp.sqrt(ru)
                b = d_prime * jnp.power(ru, 3.5) * jnp.sqrt(sr)

                w_sr = (sr / h) * w_su + 6 * sr * ru / jnp.square(h) * H_su + 2 * ((a + b) / h) * x1
                H_sr = (jnp.square(sr / h) * H_su
                        - d_prime * jnp.power(sr, 2.5) * jnp.sqrt(ru) * x1
                        + (1 / (jnp.sqrt(12) * d)) * jnp.sqrt(sr) * jnp.power(ru, 1.5) * x2)

                w_r = w_s + w_sr
                J_r = J_s + sr * H_sr + (sr / 2) * (w_s + w_r)
            else:
                b_t = w_t - 0.5 * (w_u + w_s)  # the brownian bridge is our access to randomness
                w_r = w_s + (2 * jnp.sqrt(sr * ru) / h) * b_t + (sr / h) * w_su
                J_r = None
            return LevyVal(h=r, W=w_r, J=J_r, H=None)

        return _final_interpolation(final_state)

        # def _equal_to_s_cond(_state: _State):
        #     return r < (_state.s + cancellation_err_tol)
        #
        # def _return_s_value(_state: _State) -> LevyVal:
        #     return LevyVal(h=r, W=_state.w_s, J=_state.J_s, H=None)
        #
        # def _equal_to_u_cond(_state: _State):
        #     return r > (_state.u - cancellation_err_tol)
        #
        # def _return_u_value(_state: _State) -> LevyVal:
        #     return LevyVal(h=r, W=_state.w_u, J=_state.J_u, H=None)
        #
        # def _not_equal_to_s_fun(_state: _State) -> LevyVal:
        #     return jax.lax.cond(_equal_to_u_cond(_state), _return_u_value, _final_interpolation, _state)
        #
        # return jax.lax.cond(_equal_to_s_cond(final_state), _return_s_value, _not_equal_to_s_fun, final_state)
