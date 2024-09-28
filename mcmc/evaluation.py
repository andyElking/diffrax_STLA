import math
from functools import partial, reduce
from operator import mul

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import ot  # pyright: ignore
from jax import Array
from matplotlib import pyplot as plt  # pyright: ignore
from numpyro import diagnostics  # pyright: ignore


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


def truncate_samples(x, y, max_len: int = 2**16):
    assert x.shape[1:] == y.shape[1:]
    prod = reduce(mul, x.shape[1:], 1)
    if prod >= 4:
        max_len = int(max_len / math.sqrt(prod))

    if x.shape[0] > max_len:
        x = x[:max_len]
    if y.shape[0] > max_len:
        y = y[:max_len]
    return x, y


def compute_w2(x, y, num_iters, max_len: int = 2**11):
    x, y = truncate_samples(x, y, max_len)
    source_samples = np.array(x)
    target_samples = np.array(y)
    source_weights = np.ones(source_samples.shape[0]) / source_samples.shape[0]
    target_weights = np.ones(target_samples.shape[0]) / target_samples.shape[0]
    mm = ot.dist(source_samples, target_samples)
    return ot.emd2(source_weights, target_weights, mm, numItermax=num_iters)


@partial(jax.jit, static_argnames=("max_len",))
def energy_distance(x: Array, y: Array, max_len: int = 2**14):
    assert y.ndim == x.ndim
    x, y = truncate_samples(x, y, max_len)

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


def eval_logreg(
    samples,
    evals_per_sample=None,
    ground_truth=None,
    num_iters_w2=0,
    x_test=None,
    labels_test=None,
    has_alpha=False,
):
    if isinstance(samples, dict):
        samples = vec_dict_to_array(samples)

    if has_alpha:
        samples = samples[..., 1:]

    sample_dim = samples.shape[-1]
    reshaped = jnp.reshape(samples, (-1, sample_dim))
    result_str = ""

    ess = diagnostics.effective_sample_size(samples)
    avg_ess = 1 / jnp.mean(1 / jnp.stack(jtu.tree_leaves(ess)))
    ess_per_sample = avg_ess / reshaped.shape[0]
    result_str += (
        f"\nEffective sample size: {avg_ess:.4},"
        f" ess per sample: {ess_per_sample:.4}"
    )
    if evals_per_sample is not None:
        avg_evals = jnp.mean(evals_per_sample)
        result_str += f", grad evals per sample: {avg_evals:.4}"

    half_len = reshaped.shape[0] // 2
    energy_self = energy_distance(reshaped[:half_len], reshaped[half_len:])
    result_str += f"\nEnergy dist v self: {energy_self:.4}"

    if ground_truth is not None:
        energy_gt = energy_distance(reshaped, ground_truth)
        result_str += f", energy dist vs ground truth: {energy_gt:.4}"
    else:
        energy_gt = None

    if num_iters_w2 > 0 and ground_truth is not None:
        w2 = compute_w2(reshaped, ground_truth, num_iters_w2)
        result_str += f", Wasserstein-2: {w2:.4}"
    else:
        w2 = None

    if x_test is not None and labels_test is not None:
        test_acc, test_acc_best90 = test_accuracy(x_test, labels_test, samples)
        result_str += (
            f"\nTest_accuracy: {test_acc:.4}, top 90% accuracy: {test_acc_best90:.4}"
        )
    else:
        test_acc, test_acc_best90 = None, None

    print(result_str)

    result_dict = {
        "ess": avg_ess,
        "ess_per_sample": ess_per_sample,
        "energy_v_self": energy_self,
        "grad_evals_per_sample": evals_per_sample,
        "test_accuracy": test_acc,
        "top90_accuracy": test_acc_best90,
        "w2": w2,
        "energy_gt": energy_gt,
    }

    return result_str, result_dict


def compute_metrics(sample_slice, ground_truth, x_test, labels_test):
    energy_err = energy_distance(sample_slice, ground_truth, max_len=2**14)

    if x_test is not None and labels_test is not None:
        test_acc, test_acc_best90 = test_accuracy(x_test, labels_test, sample_slice)
    else:
        test_acc, test_acc_best90 = None, None

    return energy_err, test_acc, test_acc_best90


def eval_progressive_logreg(
    samples,
    ground_truth,
    evals_per_sample,
    x_test,
    labels_test,
    num_iters_w2=100000,
    max_samples_w2=2**11,
    metric_eval_interval=1,
):
    if isinstance(samples, dict):
        samples = vec_dict_to_array(samples)

    num_chains, chain_len, sample_dim = samples.shape

    if jnp.shape(evals_per_sample) == (num_chains * chain_len,):
        evals_per_sample = jnp.reshape(evals_per_sample, (num_chains, chain_len))
        evals_per_sample = jnp.mean(evals_per_sample, axis=0)
    elif jnp.size(evals_per_sample) == 1:
        evals_per_sample = jnp.broadcast_to(evals_per_sample, (chain_len,))
    else:
        assert False, f"evals_per_sample shape: {evals_per_sample.shape}"

    assert jnp.shape(evals_per_sample) == (
        chain_len,
    ), f"{evals_per_sample.shape} != {(chain_len,)}"

    cumulative_evals = jnp.cumsum(evals_per_sample)[::metric_eval_interval]

    samples_for_eval = samples[:, ::metric_eval_interval]

    # now we go along chain_len and compute the metrics for each step
    partial_metrics = partial(
        compute_metrics,
        ground_truth=ground_truth,
        x_test=x_test,
        labels_test=labels_test,
    )
    # vectorize over the chain_len dimension
    vec_metrics = jax.vmap(partial_metrics, in_axes=1)
    energy_err, test_acc, test_acc_best90 = vec_metrics(samples_for_eval)

    if num_iters_w2 > 0:
        # wasserstein-2 distance is done via numpy, so cannot be vectorised
        w2_list = []
        for i in range(jnp.shape(samples_for_eval)[1]):
            w2_single = compute_w2(
                samples_for_eval[:, i], ground_truth, num_iters_w2, max_samples_w2
            )
            w2_list.append(w2_single)
        w2 = jnp.array(w2_list)
    else:
        w2 = None

    result_dict = {
        "energy_err": energy_err,
        "test_acc": test_acc,
        "test_acc_best90": test_acc_best90,
        "cumulative_evals": cumulative_evals,
        "w2": w2,
    }
    return result_dict


def plot_progressive_results(result_dict):
    energy_err = result_dict["energy_err"]
    test_acc = result_dict["test_acc"]
    test_acc_best90 = result_dict["test_acc_best90"]
    cumulative_evals = result_dict["cumulative_evals"]
    w2 = result_dict["w2"]

    num_subplots = 2 if w2 is None else 3
    fig, axs = plt.subplots(num_subplots, 1, figsize=(8, 15))
    axs[0].plot(cumulative_evals, energy_err)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Energy distance")

    axs[1].plot(cumulative_evals, test_acc, label="Test accuracy")
    axs[1].plot(cumulative_evals, test_acc_best90, label="Top 90% accuracy")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    if w2 is not None:
        axs[2].plot(cumulative_evals, w2)
        axs[2].set_yscale("log")
        axs[2].set_ylabel("Wasserstein-2 distance")
        axs[2].set_xlabel("Cumulative gradient evaluations")
    else:
        axs[1].set_xlabel("Cumulative gradient evaluations")

    return fig
