import math
import os
from functools import partial, reduce
from operator import mul
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import ot  # pyright: ignore
from jax import Array
from numpyro.infer import MCMC, NUTS  # pyright: ignore


def dict_to_array(dct: dict):
    b = dct["b"]
    lst = [b, dct["W"]]
    if "alpha" in dct:
        alpha = dct["alpha"]
        alpha = jnp.expand_dims(alpha, alpha.ndim)
        lst = [alpha] + lst
    return jnp.concatenate(lst, axis=-1)


vec_dict_to_array = jax.jit(jax.vmap(dict_to_array, in_axes=0, out_axes=0))


def flatten_samples(samples):
    if isinstance(samples, dict):
        samples = vec_dict_to_array(samples)
    # remove alpha
    samples = samples[..., 1:]
    return jnp.reshape(samples, (-1, samples.shape[-1]))


def predict(x, samples):
    b = samples[:, 0]
    w = samples[:, 1:]
    logits = jnp.sum(w * x, axis=-1) + b
    # apply sigmoid
    return 1.0 / (1.0 + jnp.exp(-logits))


def adjust_max_len(max_len, data_dim):
    if data_dim >= 4:
        exponent = math.ceil(math.log2(max_len / math.sqrt(data_dim)))
        max_len = 2 ** int(exponent)
    return max_len


def truncate_samples(x, max_len: Optional[int]):
    if max_len is None:
        return x

    data_dim = reduce(mul, x.shape[1:], 1)
    max_len = adjust_max_len(max_len, data_dim)
    if x.shape[0] > max_len:
        x = x[:max_len]
    return x


def compute_w2(x, y, num_iters, max_len: Optional[int] = 2**11):
    x = truncate_samples(x, max_len)
    y = truncate_samples(y, max_len)
    source_samples = np.array(x)
    target_samples = np.array(y)
    source_weights = np.ones(source_samples.shape[0]) / source_samples.shape[0]
    target_weights = np.ones(target_samples.shape[0]) / target_samples.shape[0]
    mm = ot.dist(source_samples, target_samples)
    return ot.emd2(source_weights, target_weights, mm, numItermax=num_iters)


@partial(jax.jit, static_argnames=("max_len_x", "max_len_y"))
def energy_distance(
    x: Array,
    y: Array,
    max_len_x: Optional[int] = 2**15,
    max_len_y: Optional[int] = 2**15,
):
    assert y.ndim == x.ndim
    x = truncate_samples(x, max_len_x)
    y = truncate_samples(y, max_len_y)

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


def test_accuracy(x_test, labels_test, samples):
    if isinstance(samples, dict):
        samples = vec_dict_to_array(samples)
    assert x_test.shape[1] + 1 == samples.shape[-1], (
        f"The last dim of {x_test.shape} should be the"
        f" last dim of {samples.shape} minus 1"
    )
    sample_dim = samples.shape[-1]
    samples = jnp.reshape(samples, (-1, sample_dim))
    if samples.shape[0] > 2**10:
        samples = samples[: 2**10]

    func = jax.jit(jax.vmap(lambda x: predict(x, samples), in_axes=0, out_axes=0))
    predictions = func(x_test)
    assert predictions.shape == (
        labels_test.shape[0],
        samples.shape[0],
    ), f"{predictions.shape} != {(labels_test.shape[0], samples.shape[0])}"

    labels_test = jnp.reshape(labels_test, (labels_test.shape[0], 1))
    is_correct = jnp.abs(predictions - labels_test) < 0.5
    accuracy_per_sample = jnp.mean(is_correct, axis=0)

    avg_accuracy = jnp.mean(accuracy_per_sample)

    len10 = int(0.1 * accuracy_per_sample.shape[0])
    best_sorted = jnp.sort(accuracy_per_sample)[len10:]
    accuracy_best90 = jnp.mean(best_sorted)
    return avg_accuracy, accuracy_best90


def get_ground_truth(model, filename, x_train, labels_train):
    # if ground_truth is not computed, compute it
    if not os.path.exists(filename):
        gt_nuts = MCMC(
            NUTS(model, step_size=1.0),
            num_warmup=2**10,
            num_samples=2**13,
            num_chains=2**3,
            chain_method="vectorized",
        )
        gt_nuts.run(jr.PRNGKey(0), x_train, labels_train)
        gt_logreg = vec_dict_to_array(gt_nuts.get_samples())
        # shuffle the ground truth samples
        gt_logreg = jr.permutation(jr.key(0), gt_logreg, axis=0)
        np.save(filename, gt_logreg)
    else:
        gt_logreg = np.load(filename)
    return gt_logreg
