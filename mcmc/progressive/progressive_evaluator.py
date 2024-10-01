from functools import partial

import jax
from jax import numpy as jnp

from ..evaluation import AbstractEvaluator, compute_energy, compute_w2, test_accuracy
from ..utils import vec_dict_to_array


def compute_metrics(sample_slice, ground_truth, x_test, labels_test):
    energy_err = compute_energy(
        sample_slice, ground_truth, max_len_x=2**14, max_len_y=2**15
    )

    if x_test is not None and labels_test is not None:
        test_acc, test_acc_best90 = test_accuracy(x_test, labels_test, sample_slice)
    else:
        test_acc, test_acc_best90 = None, None

    return energy_err, test_acc, test_acc_best90


def eval_progressive_logreg(
    samples,
    ground_truth,
    evals_per_sample,
    wall_time,
    x_test,
    labels_test,
    num_iters_w2=100000,
    max_samples_w2=2**11,
    num_eval_points: int = 32,
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

    assert chain_len >= num_eval_points and chain_len % num_eval_points == 0
    metric_eval_interval = chain_len // num_eval_points
    cumulative_evals = jnp.cumsum(evals_per_sample)[::metric_eval_interval]
    samples_for_eval = samples[:, ::metric_eval_interval]

    # now we go along chain_len and compute the metrics for each step
    partial_metrics = partial(
        compute_metrics,
        ground_truth=ground_truth[: 2**14],
        x_test=x_test,
        labels_test=labels_test,
    )
    # vectorize over the chain_len dimension
    vec_metrics = jax.jit(jax.vmap(partial_metrics, in_axes=1))
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
        "wall_time": wall_time,
    }
    return result_dict


class ProgressiveEvaluator(AbstractEvaluator):
    def __init__(self, num_iters_w2=100000, max_samples_w2=2**11, num_eval_points=32):
        self.num_iters_w2 = num_iters_w2
        self.max_samples_w2 = max_samples_w2
        self.num_eval_points = num_eval_points

    def eval(self, samples, aux_output, ground_truth, config):
        x_test = config["x_test"]
        labels_test = config["labels_test"]
        evals_per_sample = aux_output["evals_per_sample"]
        wall_time = aux_output["wall_time"]
        return eval_progressive_logreg(
            samples,
            ground_truth,
            evals_per_sample,
            wall_time,
            x_test,
            labels_test,
            self.num_iters_w2,
            self.max_samples_w2,
            self.num_eval_points,
        )
