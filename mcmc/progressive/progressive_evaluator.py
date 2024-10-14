from abc import abstractmethod
from functools import partial
from typing import Optional

import jax
import jax.tree_util as jtu
from jax import Array, numpy as jnp

from ..evaluation import AbstractEvaluator
from ..logreg_utils import test_accuracy, vec_dict_to_array
from ..metrics import compute_energy, compute_w2


def compute_metrics(sample_slice, ground_truth, x_test, labels_test):
    if x_test is not None and labels_test is not None:
        test_acc, test_acc_best80 = test_accuracy(x_test, labels_test, sample_slice)
    else:
        test_acc, test_acc_best80 = None, None

    return {
        "test_acc": test_acc,
        "test_acc_best80": test_acc_best80,
    }


class AbstractProgressiveEvaluator(AbstractEvaluator):
    def __init__(self, num_points=32):
        self.num_points = num_points

    @abstractmethod
    def vectorisable_metrics(
        self, sample_slice, ground_truth, config, model, key
    ) -> dict[str, Optional[Array]]:
        raise NotImplementedError

    @abstractmethod
    def sequential_metrics(
        self, sample_slice, ground_truth, config, model, key
    ) -> dict:
        raise NotImplementedError

    @abstractmethod
    def preprocess_samples(self, samples, config):
        raise NotImplementedError

    def eval(self, samples, aux_output, ground_truth, config, model, key):
        cumulative_evals = aux_output["cumulative_evals"]
        wall_time = aux_output["wall_time"]
        samples = self.preprocess_samples(samples, config)

        num_chains, chain_len = jtu.tree_leaves(samples)[0].shape[:2]

        assert jnp.shape(cumulative_evals) == (
            chain_len,
        ), f"{cumulative_evals.shape} != {(chain_len,)}"

        assert chain_len >= self.num_points and chain_len % self.num_points == 0
        eval_interval = chain_len // self.num_points
        cumulative_evals = cumulative_evals[::eval_interval]
        samples = jtu.tree_map(lambda x: x[:, ::eval_interval], samples)
        assert jtu.tree_all(
            jtu.tree_map(
                lambda x: x.shape[:2] == (num_chains, self.num_points), samples
            )
        ), (
            f"expected shapes prefixed by {(num_chains, self.num_points)}"
            f" but got {jtu.tree_map(lambda x: x.shape, samples)}"
        )
        assert jnp.shape(cumulative_evals) == (self.num_points,)

        # now we go along chain_len and compute the metrics for each step
        partial_metrics = partial(
            self.vectorisable_metrics,
            ground_truth=ground_truth,
            config=config,
            model=model,
            key=key,
        )
        # vectorize over the chain_len dimension
        vec_metrics = jax.jit(jax.vmap(partial_metrics, in_axes=1))
        vec_dict = vec_metrics(samples)

        def get_slice(samples, i):
            return jtu.tree_map(lambda x: x[:, i], samples)

        # compute metrics which cannot be vectorised (like W2)
        seq_dict: dict[str, list] = {}
        for i in range(self.num_points):
            seq_out = self.sequential_metrics(
                get_slice(samples, i), ground_truth, config, model, key
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
        self, sample_slice, ground_truth, config, model, key
    ) -> dict[str, Optional[Array]]:
        return compute_metrics(sample_slice, ground_truth, *config["test_args"])

    def sequential_metrics(
        self, sample_slice, ground_truth, config, model, key
    ) -> dict[str, Optional[Array]]:
        energy_err = compute_energy(
            sample_slice, ground_truth, max_len_x=2**14, max_len_y=2**15
        )
        result = {"energy_err": energy_err}

        if self.num_iters_w2 > 0:
            w2 = compute_w2(
                sample_slice,
                ground_truth,
                self.num_iters_w2,
                self.max_samples_w2,
            )
            result["w2"] = w2

        return result

    def preprocess_samples(self, samples, config):
        if isinstance(samples, dict):
            samples = vec_dict_to_array(samples)
        return samples
