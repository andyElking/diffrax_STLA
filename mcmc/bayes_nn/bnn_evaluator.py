import jax.numpy as jnp
from jax import Array
from numpyro import handlers

from ..evaluation import compute_energy, compute_w2
from ..progressive import AbstractProgressiveEvaluator


def predict(model, key, samples, X, D_H):
    model = handlers.substitute(handlers.seed(model, key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None, D_H=D_H)
    return model_trace["Y"]["value"]


def bnn_prediction_accuracy(model, key, samples, test_args, gt_pred):
    X = test_args["X"]
    Y_true = test_args["Y_true"]
    D_H = test_args["D_H"]
    linspace_len = X.shape[0]
    assert Y_true.shape == (
        linspace_len,
    ), f"Expected{(linspace_len,)}, got {Y_true.shape}"
    Y_pred = predict(model, key, samples, X, D_H)
    n_samples = samples.shape[0]
    assert Y_pred.shape == (n_samples, linspace_len)
    diff = Y_pred - Y_true
    # diff should be a Guassian with mean 0
    mean_diff = jnp.mean(diff, axis=0)
    mean_err = jnp.mean(jnp.abs(mean_diff))
    # gt_pred are predictions from the ground truth samples
    pred_energy_err = compute_energy(
        diff, gt_pred - Y_true, max_len_x=2**14, max_len_y=2**15
    )

    return mean_err, pred_energy_err


class ProgBNNEvaluator(AbstractProgressiveEvaluator):
    def __init__(self, num_iters_w2=100000, max_samples_w2=2**11, num_points=32):
        self.num_iters_w2 = num_iters_w2
        self.max_samples_w2 = max_samples_w2
        super().__init__(num_points=num_points)

    def vectorisable_metrics(
        self, sample_slice, ground_truth, test_args, model, key
    ) -> dict[str, Array]:
        gt_samples, gt_predictions = ground_truth

        energy_err = compute_energy(
            sample_slice, gt_samples, max_len_x=2**14, max_len_y=2**15
        )
        mean_err, pred_energy_err = bnn_prediction_accuracy(
            model, key, sample_slice, test_args, gt_predictions
        )

        return {
            "energy_err": energy_err,
            "mean_err": mean_err,
            "pred_energy_err": pred_energy_err,
        }

    def sequential_metrics(
        self, sample_slice, ground_truth, test_args, model, key
    ) -> dict:
        gt_samples, gt_predictions = ground_truth

        if self.num_iters_w2 > 0:
            w2 = compute_w2(
                sample_slice,
                gt_samples,
                self.num_iters_w2,
                self.max_samples_w2,
            )
            return {"w2": w2}
        else:
            return {}
