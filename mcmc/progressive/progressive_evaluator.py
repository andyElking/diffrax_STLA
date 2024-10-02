from abc import abstractmethod
from functools import partial

import jax
from jax import Array, numpy as jnp

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

    return {
        "energy_err": energy_err,
        "test_acc": test_acc,
        "test_acc_best90": test_acc_best90,
    }


class AbstractProgressiveEvaluator(AbstractEvaluator):
    def __init__(self, num_points=32):
        self.num_points = num_points

    @abstractmethod
    def vectorisable_metrics(
        self, sample_slice, ground_truth, test_args, model, key
    ) -> dict[str, Array]:
        raise NotImplementedError

    @abstractmethod
    def sequential_metrics(
        self, sample_slice, ground_truth, test_args, model, key
    ) -> dict:
        raise NotImplementedError

    def eval(self, samples, aux_output, ground_truth, config, model, key):
        test_args = config["test_args"]
        cumulative_evals = aux_output["cumulative_evals"]
        wall_time = aux_output["wall_time"]
        if isinstance(samples, dict):
            samples = vec_dict_to_array(samples)

        num_chains, chain_len, sample_dim = samples.shape

        assert jnp.shape(cumulative_evals) == (
            chain_len,
        ), f"{cumulative_evals.shape} != {(chain_len,)}"

        assert chain_len >= self.num_points and chain_len % self.num_points == 0
        metric_eval_interval = chain_len // self.num_points
        cumulative_evals = cumulative_evals[::metric_eval_interval]
        samples_for_eval = samples[:, ::metric_eval_interval]
        del samples
        assert jnp.shape(samples_for_eval) == (num_chains, self.num_points, sample_dim)
        assert jnp.shape(cumulative_evals) == (self.num_points,)

        # now we go along chain_len and compute the metrics for each step
        partial_metrics = partial(
            self.vectorisable_metrics,
            ground_truth=ground_truth[: 2**14],
            test_args=test_args,
            model=model,
            key=key,
        )
        # vectorize over the chain_len dimension
        vec_metrics = jax.jit(jax.vmap(partial_metrics, in_axes=1))
        vec_dict = vec_metrics(samples_for_eval)

        # compute metrics which cannot be vectorised (like W2)
        seq_dict: dict[str, list] = {}
        for i in range(jnp.shape(samples_for_eval)[1]):
            seq_out = self.sequential_metrics(
                samples_for_eval[:, i], ground_truth, test_args, model, key
            )
            for key, value in seq_out.items():
                if key not in seq_dict:
                    seq_dict[key] = []
                seq_dict[key].append(value)

        result_dict = {
            "cumulative_evals": cumulative_evals,
            "wall_time": wall_time,
        }
        for key, value in vec_dict.items():
            result_dict[key] = value
        for key, value in seq_dict.items():
            result_dict[key] = jnp.array(value)

        return result_dict


class ProgressiveEvaluator(AbstractProgressiveEvaluator):
    def __init__(self, num_iters_w2=100000, max_samples_w2=2**11, num_points=32):
        self.num_iters_w2 = num_iters_w2
        self.max_samples_w2 = max_samples_w2
        super().__init__(num_points=num_points)

    def vectorisable_metrics(
        self, sample_slice, ground_truth, test_args, model, key
    ) -> dict[str, Array]:
        return compute_metrics(sample_slice, ground_truth, *test_args)

    def sequential_metrics(
        self, sample_slice, ground_truth, test_args, model, key
    ) -> dict:
        if self.num_iters_w2 > 0:
            w2 = compute_w2(
                sample_slice,
                ground_truth,
                self.num_iters_w2,
                self.max_samples_w2,
            )
            return {"w2": w2}
        else:
            return {}
