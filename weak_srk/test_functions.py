import math
from functools import partial, reduce
from operator import mul
from test.test_brownian import nth_moment_indices

import jax
import jax.numpy as jnp
from jax import Array


# Write some test functions for estimating weak order


def f1(ys):
    out = 1000 * jnp.sqrt(jnp.mean((ys[:, 2] - ys[:, 3]) ** 2, axis=0))
    return jnp.expand_dims(out, axis=0)


def f2(ys):
    out = jnp.pow(jnp.mean((ys[:, 2] - ys[:, 3]) ** 4, axis=0), 1 / 4)
    return jnp.expand_dims(out, axis=0)


def std(ys):
    return jnp.std(ys, axis=0)


def empirical_nth_moments(n: int, ys: Array) -> Array:
    assert ys.ndim == 2, f"Expected 2D array, got shape {ys.shape}"
    d = ys.shape[1]
    indices = nth_moment_indices(d, n)
    num_moments = indices[0].shape[0]
    num_samples = ys.shape[0]

    @jax.jit
    def compute_fast(ys, indices):
        init = jnp.ones((num_samples, num_moments), dtype=ys.dtype)
        indices_arr = jnp.stack(indices, axis=0)

        def step(carry, idx):
            return carry * ys[:, idx], None

        result, _ = jax.lax.scan(step, init, indices_arr)
        return jnp.mean(result, axis=0)

    return compute_fast(ys, indices)


def normalise(ys: Array) -> Array:
    return (ys - jnp.mean(ys, axis=0)) / jnp.std(ys, axis=0)


def third(ys):
    ys = normalise(ys)
    return jnp.pow(empirical_nth_moments(3, ys), 1 / 3)


def fourth(ys):
    ys = normalise(ys)
    return jnp.pow(empirical_nth_moments(4, ys), 1 / 4)


fun_list = [f1, f2, std, third, fourth]


@jax.jit
def compute_test_functions(ys: Array) -> Array:
    if ys.shape[1] == 1:
        ys = ys[:, 0]
    fun_results = [fun(ys) for fun in fun_list]
    return jnp.concatenate(fun_results, axis=0)


@jax.jit
def comp_fun_dist(ys1, ref_funs):
    fun1 = compute_test_functions(ys1)
    dist = jnp.sum(jnp.abs(fun1 - ref_funs))
    return dist


@partial(jax.jit, static_argnames=("max_len",))
def energy_distance(x: Array, y: Array, max_len: int = 2**16):
    assert y.ndim == x.ndim
    assert x.shape[1:] == y.shape[1:]
    prod = reduce(mul, x.shape[1:], 1)
    if prod >= 4:
        max_len = int(max_len / math.sqrt(prod))

    if x.shape[0] > max_len:
        x = x[:max_len]
    if y.shape[0] > max_len:
        y = y[:max_len]

    @partial(jax.vmap, in_axes=(None, 0))
    def _dist_single(_x, _y_single):
        assert _x.ndim == _y_single.ndim + 1, f"{_x.ndim} != {_y_single.ndim + 1}"
        diff = _x - _y_single
        if x.ndim > 1:
            # take the norm over all axes except the first one
            diff = jnp.sqrt(jnp.sum(diff**2, axis=tuple(range(1, diff.ndim))))
        return jnp.mean(jnp.abs(diff))

    def dist(_x, _y):
        assert _x.ndim == _y.ndim
        return jnp.mean(_dist_single(_x, _y))

    return 2 * dist(x, y) - dist(x, x) - dist(y, y)
