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
# H_{s,t} = 1/(t-s) ( \int_s^t ( W_u - (u-s)/(t-s) W_{s,t} ) du ).
# bhh_t = t * H_{0,t}
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
    bhh_s: Optional[Scalar]
    bhh_t: Optional[Scalar]
    bhh_u: Optional[Scalar]
    bkk_s: Optional[Scalar]
    bkk_t: Optional[Scalar]
    bkk_u: Optional[Scalar]


def _levy_diff(x0: LevyVal, x1: LevyVal) -> LevyVal:
    r"""Computes $(W_{s,u}, H_{s,u})$ from $(W_s, \bar{H}_{s,u})$ and
    $(W_u, \bar{H}_u)$, where $\bar{H}_u = u * H_u$.
    alternatively, if `levy_area=="space-time-time"`, then computes
    $(W_{s,u}, H_{s,u}, K_{s,u})$ from $(W_s, \bar{H}_s, \bar{K}_s)$ and
    $(W_u, \bar{H}_u, \bar{K}_u)$, where $\bar{K}_u = u**2 * K_u$.

    **Arguments:**

    - `x0`: `LevyVal` at time `s`

    - `x1`: `LevyVal` at time `u`

    **Returns:**

    `LevyVal(W_su, H_su)` or `LevyVal(W_su, H_su, K_su)`
    """
    h = (x1.t - x0.t).astype(x0.W.dtype)
    h = jnp.where(jnp.abs(h) < jnp.finfo(h).eps, jnp.inf, h)
    inverse_h = 1 / h
    u_bb_s = x1.t * x0.W - x0.t * x1.W
    w_01 = x1.W - x0.W
    bhh_01 = x1.bar_H - x0.bar_H - 0.5 * u_bb_s  # bhh_01 = H_{s,u} * (u-s)
    hh_01 = inverse_h * bhh_01

    if x0.K is not None:
        assert x1.K is not None
        # bkk_01 = K_{s,u} * (u-s)**2
        bkk_01 = (
            x1.bar_K
            - x0.bar_K
            - h / 2 * x0.bar_H
            + x0.t / 2 * bhh_01
            - (x1.t - 2 * x0.t) / 12 * u_bb_s
        )
        kk_01 = jnp.square(inverse_h) * bkk_01
    else:
        kk_01 = None

    return LevyVal(t=h, W=w_01, H=hh_01, bar_H=None, K=kk_01, bar_K=None)


class VirtualBrownianTree(AbstractBrownianPath):
    """Brownian simulation that discretises the interval `[t0, t1]` to tolerance `tol`,
    and is piecewise quadratic at that discretisation.

    Can be initialised with `levy_area` set to `""`, or `"space-time"`.
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
        if levy_area not in ["", "space-time", "space-time-time"]:
            raise ValueError(
                f"levy_area must be one of '', 'space-time', but got {levy_area}."
            )
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
            bar_H=None,
            K=jtu.tree_map(sqrt_mult, x.K),
            bar_K=None,
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

            if self.levy_area in ["space-time", "space-time-time"]:
                # set bhh_t and bkk_t to None
                levy_out = jtu.tree_map(
                    lambda x: dataclasses.replace(x, bhh_t=None, bkk_t=None), levy_0
                )
            else:
                levy_out = levy_0

        else:
            t1 = eqxi.nondifferentiable(t1, name="t1")
            # map the interval [self.t0, self.t1] onto [0,1]
            t1 = linear_rescale(self.t0, t1, self.t1)
            levy_1 = self._evaluate(t1)

            if self.levy_area in ["space-time", "space-time-time"]:
                levy_diff = _levy_diff
            else:
                levy_diff = lambda x, y: LevyVal(
                    t=y.t - x.t, W=y.W - x.W, H=None, bar_H=None, K=None, bar_K=None
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

    def _brownian_arch(
        self, s, u, w_s, w_u, key, shape, dtype, bhh_s, bhh_u, bkk_s, bkk_u
    ):
        """For `t = (s+u)/2` evaluates `w_t` and `bhh_t` conditioned
         on `w_s`, `w_u`, `bhh_s`, and `bhh_u`.
        **Arguments:**
            - `s`: start time
            - `u`: end time
            - `w_s`: value of BM at s
            - `w_u`: value of BM at u
            - `key`:
            - `shape`:
            - `dtype`:
            - `bhh_s`: space-time Lévy integral at s
            - `bhh_u`: space-time Lévy integral at u
        """

        su = u - s
        t = s + su / 2
        root_su = jnp.sqrt(su)

        if self.levy_area == "space-time-time":

            # TODO: check if this is correct

            assert bhh_u is not None
            assert bhh_s is not None
            assert bkk_u is not None
            assert bkk_s is not None
            z_key, x1_key, x2_key = jrandom.split(key, 3)
            z = jrandom.normal(z_key, shape, dtype) * (root_su / 4)
            x1 = jrandom.normal(x1_key, shape, dtype) * (root_su / jnp.sqrt(768))
            x2 = jrandom.normal(x2_key, shape, dtype) * (root_su / jnp.sqrt(2880))

            su2 = jnp.square(su)
            u_bb_s = u * w_s - s * w_u
            w_su = w_u - w_s
            bhh_su = bhh_u - bhh_s - 0.5 * u_bb_s
            bkk_su = (
                bkk_u
                - bkk_s
                - su / 2 * bhh_s
                + s / 2 * bhh_su
                - (u - 2 * s) / 12 * u_bb_s
            )

            w_st = 0.5 * w_su + 1.5 / su * bhh_su + z
            bhh_st = 0.125 * bhh_su + 15 / (8 * su) * bkk_su - su / 4 * z + su / 2 * x1
            bkk_st = 1 / 32 * bkk_su - su2 / 8 * x1 + su2 / 4 * x2

            # jax.debug.print("w_st {w_st}, bhh_st {bhh_st}, bkk_st {bkk_st}",
            #                 w_st=w_st, bhh_st=bhh_st, bkk_st=bkk_st)

            w_t = w_s + w_st
            t_bb_s = t * w_s - s * w_t
            bhh_t = bhh_s + bhh_st + 0.5 * t_bb_s
            bkk_t = (
                bkk_s
                + bkk_st
                + su / 4 * bhh_s
                - s / 2 * bhh_st
                + (t - 2 * s) / 12 * t_bb_s
            )

            # jax.debug.print("t {t}, w_t {w_t}, bhh_t {bhh_t}, bkk_t {bkk_t}",
            #                 t=t, w_t=w_t, bhh_t=bhh_t, bkk_t=bkk_t)

        elif self.levy_area == "space-time":
            assert bhh_u is not None
            assert bhh_s is not None
            x1_key, x2_key = jrandom.split(key, 2)
            x1 = jrandom.normal(x1_key, shape, dtype) * root_su
            x2 = jrandom.normal(x2_key, shape, dtype) * root_su

            u_bb_s = u * w_s - s * w_u  # = u * brownian bridge on [0,u] evaluated at s
            bhh_su = bhh_u - bhh_s - 0.5 * u_bb_s
            w_t = w_s + 0.5 * (w_u - w_s) + 3 / (2 * su) * bhh_su + 0.25 * x1
            bhh_st = 0.125 * bhh_su - su / 16 * x1 + su / (8 * jnp.sqrt(3)) * x2
            bhh_t = bhh_s + bhh_st + 0.5 * (t * w_s - s * w_t)
            bkk_t = None

        else:
            assert bhh_u is None
            assert bhh_s is None
            mean = w_s + 0.5 * (w_u - w_s)
            std = 0.5 * jnp.sqrt(su)
            w_t = mean + std * jrandom.normal(key, shape, dtype)
            bhh_t, bkk_t = None, None
        return w_t, bhh_t, bkk_t

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
        if self.levy_area == "space-time-time":
            (
                key,
                init_key_w,
                init_key_stla,
                init_key_sttla,
                midpoint_key,
            ) = jrandom.split(key, 5)
            w_1 = jrandom.normal(init_key_w, shape, dtype)
            bhh_1 = jnp.sqrt(1 / 12) * jrandom.normal(init_key_stla, shape, dtype)
            bhh_0 = jnp.zeros_like(bhh_1)
            bkk_1 = jnp.sqrt(1 / 720) * jrandom.normal(init_key_sttla, shape, dtype)
            bkk_0 = jnp.zeros_like(bkk_1)

        elif self.levy_area == "space-time":
            key, init_key_w, init_key_la, midpoint_key = jrandom.split(key, 4)
            w_1 = jrandom.normal(init_key_w, shape, dtype)
            bhh_1 = jnp.sqrt(1 / 12) * jrandom.normal(init_key_la, shape, dtype)
            bhh_0 = jnp.zeros_like(bhh_1)
            bkk_0, bkk_1 = None, None

        else:
            key, init_key_w, midpoint_key = jrandom.split(key, 3)
            w_1 = jrandom.normal(init_key_w, shape, dtype)
            bhh_1, bhh_0 = None, None
            bkk_0, bkk_1 = None, None

        w_thalf, bhh_thalf, bkk_half = self._brownian_arch(
            t0, t1, w_s, w_1, midpoint_key, shape, dtype, bhh_0, bhh_1, bkk_0, bkk_1
        )
        init_state = _State(
            s=t0,
            t=thalf,
            u=t1,
            w_s=w_s,
            w_t=w_thalf,
            w_u=w_1,
            key=key,
            bhh_s=bhh_0,
            bhh_t=bhh_thalf,
            bhh_u=bhh_1,
            bkk_s=bkk_0,
            bkk_t=bkk_half,
            bkk_u=bkk_1,
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

            if self.levy_area in ["space-time", "space-time-time"]:
                _bhh_s = jnp.where(_cond, _state.bhh_t, _state.bhh_s)
                _bhh_u = jnp.where(_cond, _state.bhh_u, _state.bhh_t)
                if self.levy_area == "space-time-time":
                    _bkk_s = jnp.where(_cond, _state.bkk_t, _state.bkk_s)
                    _bkk_u = jnp.where(_cond, _state.bkk_u, _state.bkk_t)
                else:
                    _bkk_s, _bkk_u = None, None
            else:
                _bhh_s, _bhh_u = None, None
                _bkk_s, _bkk_u = None, None
            _key = jnp.where(_cond, _key1, _key2)
            _t = _s + 0.5 * (_u - _s)

            _key, _midpoint_key = jrandom.split(_key, 2)
            _w_t, _bhh_t, _bkk_t = self._brownian_arch(
                _s,
                _u,
                _w_s,
                _w_u,
                _midpoint_key,
                shape,
                dtype,
                _bhh_s,
                _bhh_u,
                _bkk_s,
                _bkk_u,
            )
            return _State(
                s=_s,
                t=_t,
                u=_u,
                w_s=_w_s,
                w_t=_w_t,
                w_u=_w_u,
                bhh_s=_bhh_s,
                bhh_t=_bhh_t,
                bhh_u=_bhh_u,
                key=_key,
                bkk_s=_bkk_s,
                bkk_t=_bkk_t,
                bkk_u=_bkk_u,
            )

        final_state = lax.while_loop(_cond_fun, _body_fun, init_state)

        # Based on the values of (W, H) at s<t<u (where t = (s+u)/2), we interpolate
        # to obtain approximate values of (W_r, hh_r) for all r ∈ [s,u]. This is done
        # in a way that gives (W_r, hh_r) all the correct first and second moments
        # conditional on (W_s, H_s), and (W_u, H_u), where (W_t, H_t) is treated as
        # the source of randomness.
        # NOTE: this gives a different result than the original implementation of the
        # VirtualBrownianTree by Patrick Kidger.

        s = final_state.s
        t = final_state.t
        u = final_state.u
        w_s = final_state.w_s
        w_t = final_state.w_t
        w_u = final_state.w_u
        bhh_s = final_state.bhh_s
        bhh_t = final_state.bhh_t
        bhh_u = final_state.bhh_u
        # bkk_s = final_state.bkk_s
        bkk_t = final_state.bkk_t
        # bkk_u = final_state.bkk_u

        su = u - s
        w_su = w_u - w_s
        sr = r - s
        ru = u - r

        if self.levy_area == "space-time-time":

            w_r = w_t
            hh_r = bhh_t / t
            bhh_r = bhh_t
            kk_r = bkk_t / t**2
            bkk_r = bkk_t

        elif self.levy_area == "space-time":
            # reverse _brownian_arch to get x1, x2 from
            # w_s, w_t, w_u, bhh_s, bhh_t, bhh_u

            uB = u * w_s - s * w_u  # B = brownian bridge on [0,u] evaluated at s
            bhh_su = bhh_u - bhh_s - 0.5 * uB
            x1 = 4 * (w_t - 0.5 * w_s - 0.5 * w_u) - 6 / su * bhh_su
            bhh_st = bhh_t - bhh_s - 0.5 * (t * w_s - s * w_t)
            x2 = jnp.sqrt(3) * (1 / su * (8 * bhh_st - bhh_su) + 0.5 * x1)
            # note that x1, x2 are Normal(0, su), unlike in thm 6.1.4 of
            # Foster's thesis, where they are Normal(0, 1), so there is
            # an extra factor of sqrt(su) in the formula for c and d_prime.
            root_su = jnp.sqrt(su)
            x1 = x1 / root_su
            x2 = x2 / root_su

            sr3 = jnp.power(sr, 3)
            ru3 = jnp.power(ru, 3)
            su3 = jnp.power(su, 3)
            sr_ru_half = jnp.sqrt(sr * ru)
            d = jnp.sqrt(sr3 + ru3)
            d_prime = 1 / (2 * su * d)
            a = d_prime * sr3 * sr_ru_half
            b = d_prime * ru3 * sr_ru_half

            w_sr = sr / su * w_su + 6 * sr * ru / su3 * bhh_su + 2 * (a + b) / su * x1
            w_r = w_s + w_sr
            c = jnp.sqrt(3 * sr3 * ru3) / (6 * d)
            bhh_sr = sr3 / su3 * bhh_su - a * x1 + c * x2
            bhh_r = bhh_s + bhh_sr + 0.5 * (r * w_s - s * w_r)

            r = jnp.where(jnp.abs(r) < jnp.finfo(r).eps, jnp.inf, r)
            inverse_r = 1 / r
            hh_r = inverse_r * bhh_r
            kk_r, bkk_r = None, None

        else:
            # the brownian bridge b_t is our access to randomness
            b_t = w_t - 0.5 * (w_u + w_s)
            w_r = w_s + (2 * jnp.sqrt(sr * ru) / su) * b_t + (sr / su) * w_su
            hh_r, bhh_r = None, None
            kk_r, bkk_r = None, None
        return LevyVal(t=r, W=w_r, bar_H=bhh_r, H=hh_r, K=kk_r, bar_K=bkk_r)
