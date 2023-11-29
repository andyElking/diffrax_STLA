import dataclasses
from dataclasses import field
from typing import Optional, Tuple, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu

from ..custom_types import levy_tree_transpose, LevyVal, PyTree, Scalar
from ..misc import is_tuple_of_ints, linear_rescale, split_by_tree
from .base import AbstractBrownianPath


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

# We define
# tH_t = \int_0^t W_u du
# H_{s,t} = 1/(t-s) ( \int_s^t ( W_u - (u-s)/(t-s) W_{s,t} ) du ).
# For more details see Definition 4.2.1 and Theorem 6.1.4 of
#
# Foster, J. M. (2020). Numerical approximations for stochastic
# differential equations [PhD thesis]. University of Oxford.


class _State(eqx.Module):
    s: Scalar
    t: Scalar
    u: Scalar
    w_s: Scalar
    w_t: Scalar
    w_u: Scalar
    key: "jax.random.PRNGKey"
    sH_s: Optional[Scalar]
    tH_t: Optional[Scalar]
    uH_u: Optional[Scalar]


def _levy_diff(x0: LevyVal, x1: LevyVal) -> LevyVal:
    r"""Computes $(W_{s,u}, H_{s,u})$ from $(W_s, sH_s)$ and
    $(W_u, uH_u)$, where $uH_u = \int_0^u W_t dt$

    **Arguments:**

    - `x0`: `LevyVal(W_s, sH_s)`

    - `x1`: `LevyVal$(W_u, uH_u)`

    **Returns:**

    `LevyVal(W_{s,u}, H_{s,u})`
    """
    h = (x1.t - x0.t).astype(x0.W.dtype)
    h = jnp.where(jnp.abs(h) < jnp.finfo(h).eps, jnp.inf, h)
    inverse_h = 1 / h
    uB = x1.t * x0.W - x0.t * x1.W
    w_01 = x1.W - x0.W
    thh_01 = x1.tH_t - x0.tH_t - 0.5 * uB
    hh_01 = inverse_h * thh_01
    return LevyVal(t=h, W=w_01, H=hh_01)


class VirtualBrownianTree(AbstractBrownianPath):
    """Brownian simulation that discretises the interval `[t0, t1]` to tolerance `tol`,
    and is piecewise quadratic at that discretisation.

    If `levy_area=="space_time"`, then it also computes space-time Lévy area `H`.
    This will impact the Brownian path, so even with the same key, the trajectory will
    be different depending on the value of `levy_area`.

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

        (The implementation here is a slight improvement on the reference implementation
        by using an interpolation method which ensures all the 2nd moments are correct.)
    """

    t0: Scalar = field(init=True)
    t1: Scalar = field(init=True)  # override init=False in AbstractPath
    tol: Scalar
    shape: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    levy_area: str = eqx.field(static=True)
    key: "jax.random.PRNGKey"  # noqa: F821

    def __init__(
        self,
        t0: Scalar,
        t1: Scalar,
        tol: Scalar,
        shape: Union[Tuple[int, ...], PyTree[jax.ShapeDtypeStruct]],
        key: "jax.random.PRNGKey",
        levy_area: str = "",
    ):
        if t0 >= t1:
            raise ValueError("t0 must be strictly less than t1.")
        self.t0 = t0
        self.t1 = t1
        self.tol = tol / (self.t1 - self.t0)
        self.levy_area = levy_area
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

    def _denormalise_bm_inc(self, x: LevyVal) -> LevyVal:
        # TODO: demonstrate rescaling actually helps

        # Rescaling back from [0, 1] to the original interval [t0, t1].
        interval_len = self.t1 - self.t0  # can be any dtype
        sqrt_len = jnp.sqrt(jnp.abs(interval_len))

        def sqrt_mult(z):
            # need to cast to dtype of each leaf in PyTree
            dtype = jnp.dtype(z)
            return z * jnp.asarray(sqrt_len, dtype)

        def mult(z):
            dtype = jnp.dtype(z)
            return z * jnp.asarray(interval_len, dtype)

        return LevyVal(
            t=jtu.tree_map(mult, x.t),
            W=jtu.tree_map(sqrt_mult, x.W),
            H=jtu.tree_map(sqrt_mult, x.H),
        )

    @eqx.filter_jit
    def evaluate(
        self,
        t0: Scalar,
        t1: Optional[Scalar] = None,
        left: bool = True,
        use_levy: bool = False,
    ) -> LevyVal:
        def _is_levy_val(obj):
            return isinstance(obj, LevyVal)

        t0 = eqxi.nondifferentiable(t0, name="t0")
        # map the interval [self.t0, self.t1] onto [0,1]
        t0 = linear_rescale(self.t0, t0, self.t1)
        levy_0 = self._evaluate(t0)
        if t1 is None:

            if self.levy_area == "space-time":
                # set tH_t to None
                levy_out = jtu.tree_map(
                    lambda x: dataclasses.replace(x, tH_t=None), levy_0
                )
            else:
                levy_out = levy_0

        else:
            t1 = eqxi.nondifferentiable(t1, name="t1")
            # map the interval [self.t0, self.t1] onto [0,1]
            t1 = linear_rescale(self.t0, t1, self.t1)
            levy_1 = self._evaluate(t1)
            levy_diff = (
                _levy_diff
                if self.levy_area == "space-time"
                else lambda x, y: LevyVal(t=y.t - x.t, W=y.W - x.W)
            )
            levy_out = jtu.tree_map(levy_diff, levy_0, levy_1, is_leaf=_is_levy_val)

        levy_out = levy_tree_transpose(self.shape, self.levy_area, levy_out)
        # now map [0,1] back onto [self.t0, self.t1]
        levy_out = self._denormalise_bm_inc(levy_out)
        assert isinstance(levy_out, LevyVal)
        return levy_out if use_levy else levy_out.W

    def _evaluate(self, r: Scalar) -> PyTree[LevyVal]:
        """Maps the _evaluate_leaf function at time τ using self.key onto self.shape"""
        r = eqxi.error_if(
            r,
            r < 0,
            "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1].",
        )
        r = eqxi.error_if(
            r,
            r > 1,
            "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1].",
        )
        # Clip because otherwise the while loop below won't terminate, and the above
        # errors are only raised after everything has finished executing.
        map_func = lambda key, shape: self._evaluate_leaf(key, r, shape)
        return jtu.tree_map(map_func, self.key, self.shape)

    def _brownian_arch(self, s, u, w_s, w_u, key, shape, dtype, sH_s, uH_u):
        """For `t = (s+u)/2` evaluates `w_t` and `tH_t` conditioned
         on `w_s`, `w_u`, `sH_s`, and `uH_u`.
        **Arguments:**
            - `s`: start time
            - `u`: end time
            - `w_s`: value of BM at s
            - `w_u`: value of BM at u
            - `key`:
            - `shape`:
            - `dtype`:
            - `sH_s`: space-time Lévy integral at s
            - `uH_u`: space-time Lévy integral at u
        """

        h = (u - s).astype(w_s.dtype)
        sqrt_h = jnp.sqrt(h)

        if self.levy_area == "space-time":
            assert uH_u is not None
            assert sH_s is not None
            x1_key, x2_key = jrandom.split(key, 2)
            x1 = jrandom.normal(x1_key, shape, dtype) * sqrt_h
            x2 = jrandom.normal(x2_key, shape, dtype) * sqrt_h

            uB = u * w_s - s * w_u  # B = brownian bridge on [0,u] evaluated at s
            hH_su = uH_u - sH_s - 0.5 * uB
            w_t = w_s + 0.5 * (w_u - w_s) + 3 / (2 * h) * hH_su + 0.25 * x1
            t = s + (u - s) / 2
            stH_st = 0.125 * hH_su - h / 16 * x1 + h / (8 * jnp.sqrt(3)) * x2

            tH_t = sH_s + stH_st + 0.5 * (t * w_s - s * w_t)

        else:
            assert uH_u is None
            assert sH_s is None
            mean = w_s + 0.5 * (w_u - w_s)
            std = 0.5 * jnp.sqrt(h)
            w_t = mean + std * jrandom.normal(key, shape, dtype)
            tH_t = None
        return w_t, tH_t

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

        thalf = jnp.array(0.5, dtype)
        w_s = jnp.zeros(shape, dtype)
        if self.levy_area == "space-time":
            key, init_key_w, init_key_la, midpoint_key = jrandom.split(key, 4)
            w_1 = jrandom.normal(init_key_w, shape, dtype)
            tH_1 = jnp.sqrt(1 / 12) * jrandom.normal(init_key_la, shape, dtype)
            tH_0 = jnp.zeros_like(tH_1)

        else:
            key, init_key_w, midpoint_key = jrandom.split(key, 3)
            w_1 = jrandom.normal(init_key_w, shape, dtype)
            tH_1, tH_0 = None, None

        w_thalf, tH_thalf = self._brownian_arch(
            t0, t1, w_s, w_1, midpoint_key, shape, dtype, tH_0, tH_1
        )
        init_state = _State(
            s=t0,
            t=thalf,
            u=t1,
            w_s=w_s,
            w_t=w_thalf,
            w_u=w_1,
            sH_s=tH_0,
            tH_t=tH_thalf,
            uH_u=tH_1,
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
            """Single-step of binary search for τ."""
            _key1, _key2 = jrandom.split(_state.key, 2)
            _cond = r > _state.t
            _s = jnp.where(_cond, _state.t, _state.s)
            _u = jnp.where(_cond, _state.u, _state.t)
            _w_s = jnp.where(_cond, _state.w_t, _state.w_s)
            _w_u = jnp.where(_cond, _state.w_u, _state.w_t)
            if self.levy_area == "space-time":
                _sH_s = jnp.where(_cond, _state.tH_t, _state.sH_s)
                _uH_u = jnp.where(_cond, _state.uH_u, _state.tH_t)
            else:
                _sH_s, _uH_u = None, None
            _key = jnp.where(_cond, _key1, _key2)
            _t = _s + 0.5 * (_u - _s)

            _key, _midpoint_key = jrandom.split(_key, 2)
            _w_t, _tH_t = self._brownian_arch(
                _s, _u, _w_s, _w_u, _midpoint_key, shape, dtype, _sH_s, _uH_u
            )
            return _State(
                s=_s,
                t=_t,
                u=_u,
                w_s=_w_s,
                w_t=_w_t,
                w_u=_w_u,
                sH_s=_sH_s,
                tH_t=_tH_t,
                uH_u=_uH_u,
                key=_key,
            )

        final_state = lax.while_loop(_cond_fun, _body_fun, init_state)

        # Based on the values of (W, J) at s<t<u (where t = (s+u)/2), we interpolate
        # to obtain approximate values of (W_r, J_r) for all r ∈ [s,u]. This is done
        # in a way that gives (W_r, J_r) all the correct first and second moments
        # conditional on (W_s, tH_0), and (W_u, uH_u), where (W_t, tH_t) is treated as
        # the source of randomness.
        # NOTE: this gives a different result than the original implementation of the
        # VirtualBrownianTree by Patrick Kidger.

        s = final_state.s
        t = final_state.t
        u = final_state.u
        w_s = final_state.w_s
        w_t = final_state.w_t
        w_u = final_state.w_u
        sH_s = final_state.sH_s
        tH_t = final_state.tH_t
        uH_u = final_state.uH_u

        h = u - s
        w_su = w_u - w_s
        sr = r - s
        ru = u - r

        if self.levy_area == "space-time":
            # reverse _brownian_arch to get x1, x2 from
            # w_s, w_t, w_u, sH_s, tH_t, uH_u

            uB = u * w_s - s * w_u  # B = brownian bridge on [0,u] evaluated at s
            hH_su = uH_u - sH_s - 0.5 * uB
            x1 = 4 * (w_t - 0.5 * w_s - 0.5 * w_u) - 6 / h * hH_su
            stH_st = tH_t - sH_s - 0.5 * (t * w_s - s * w_t)
            x2 = jnp.sqrt(3) * (1 / h * (8 * stH_st - hH_su) + 0.5 * x1)
            # note that x1, x2 are Normal(0, h), unlike in thm 6.1.4 of
            # Foster's thesis, where they are Normal(0, 1), so there is
            # an extra factor of sqrt(h) in the formula for c and d_prime.
            root_h = jnp.sqrt(h)
            x1 = x1 / root_h
            x2 = x2 / root_h

            sr3 = jnp.power(sr, 3)
            ru3 = jnp.power(ru, 3)
            h3 = jnp.power(h, 3)
            sr_ru_half = jnp.sqrt(sr * ru)
            d = jnp.sqrt(sr3 + ru3)
            d_prime = 1 / (2 * h * d)
            a = d_prime * sr3 * sr_ru_half
            b = d_prime * ru3 * sr_ru_half

            w_sr = sr / h * w_su + 6 * sr * ru / h3 * hH_su + 2 * (a + b) / h * x1
            w_r = w_s + w_sr
            c = jnp.sqrt(3 * sr3 * ru3) / (6 * d)
            srH_sr = sr3 / h3 * hH_su - a * x1 + c * x2
            rH_r = sH_s + srH_sr + 0.5 * (r * w_s - s * w_r)
            H_r = 1 / r * rH_r

        else:
            # the brownian bridge b_t is our access to randomness
            b_t = w_t - 0.5 * (w_u + w_s)
            w_r = w_s + (2 * jnp.sqrt(sr * ru) / h) * b_t + (sr / h) * w_su
            H_r = None
            rH_r = None
        return LevyVal(t=r, W=w_r, tH_t=rH_r, H=H_r)


# Coefficients for space-time-time Levy area generation
# B_1 = jnp.sqrt((st * tu * ((st - tu)**4 + 4 * (st**2 * tu**2))) / h**5)
# H_1 = -(st**3 * tu * (st**2 - 3 * st * tu + 6 * tu**2)) /
# (2 * jnp.sqrt((st * tu * ((st - tu)**4 + 4 * (st**2 * tu**2))) * h**5))
# H_2 = jnp.sqrt((st * tu**3 * (st**5 + tu**5)) /
# (12 * ((st - tu)**4 + 4 * (st**2 * tu**2)))) / h**2
# K_1 = (st**4 * tu * (st - tu)) /
# (12 * jnp.sqrt(st * tu * h**5 * ((st - tu)**4 + 4 * (st**2 * tu**2))))
# K_2 = -(st**5 * tu**2) / (4 * h *
# jnp.sqrt(3 * st * tu * (4 * st**2 * tu**2 + (st - tu)**4) * (st**5 + tu**5)))
# K_3 = jnp.sqrt((st * tu**5) / (720 * (st**5 + tu**5)))
