import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array, numpy as jnp, tree_util as jtu
from numpyro import handlers

from ..logging import AbstractLogger
from ..metrics import compute_energy, compute_w2
from ..progressive import AbstractProgressiveEvaluator


def predict(model, key, samples, X, D_H):
    print(
        f"Sample shape: {jtu.tree_map(lambda x: x.shape, samples)}, X shape: {X.shape}"
    )
    model = handlers.substitute(handlers.seed(model, key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None, D_H=D_H)
    return model_trace["Y"]["value"]


def vec_predict(model, key, samples, test_args):
    X, Y_true, D_H = test_args
    linspace_len = X.shape[0]
    assert Y_true.shape == (
        linspace_len,
    ), f"Expected{(linspace_len,)}, got {Y_true.shape}"
    num_samples = jtu.tree_leaves(samples)[0].shape[0]
    keys = jax.random.split(key, num_samples)

    def pred_fun(key, samples):
        return predict(model, key, samples, X, D_H)

    _vec_fun = eqx.filter_jit(jax.vmap(pred_fun, in_axes=0))
    y_pred = _vec_fun(keys, samples)
    assert y_pred.shape == (
        num_samples,
        linspace_len,
        1,
    ), f"Expected {(num_samples, linspace_len)}, got {y_pred.shape}"
    return y_pred[:, :, 0]


def bnn_pred_error(y, y_gt, y_true):
    diff = y - y_true
    # diff should be a Guassian with mean 0
    mean_diff = jnp.mean(diff, axis=0)
    mean_err = jnp.mean(jnp.abs(mean_diff))
    # gt_pred are predictions from the ground truth samples
    diff_sparse = diff[::50]
    diff_gt_sparse = (y_gt - y_true)[::50]
    pred_energy_err = compute_energy(
        diff_sparse, diff_gt_sparse, max_len_x=2**14, max_len_y=2**15
    )
    return mean_err, pred_energy_err


class ProgBNNEvaluator(AbstractProgressiveEvaluator):
    def __init__(self, num_iters_w2=100000, max_samples_w2=2**11, num_points=32):
        self.num_iters_w2 = num_iters_w2
        self.max_samples_w2 = max_samples_w2
        super().__init__(num_points=num_points)

    def vectorisable_metrics(
        self, sample_slice, ground_truth, config, model, key
    ) -> dict[str, Array]:
        gt_samples, gt_predictions = ground_truth

        x, y_true, d_h = config["test_args"]
        y_pred = vec_predict(model, key, sample_slice, config["test_args"])
        mean_err, pred_energy_err = bnn_pred_error(y_pred, gt_predictions, y_true)
        flat_slice = flatten_bnn_samples(sample_slice)
        energy_err = compute_energy(
            flat_slice, gt_samples, max_len_x=2**14, max_len_y=2**15
        )
        return {
            "energy_err": energy_err,
            "mean_err": mean_err,
            "pred_energy_err": pred_energy_err,
        }

    def sequential_metrics(
        self, sample_slice, ground_truth, config, model, key
    ) -> dict:
        gt_samples, gt_predictions = ground_truth
        results = {}

        if self.num_iters_w2 > 0:
            flat_slice = flatten_bnn_samples(sample_slice)
            w2 = compute_w2(
                flat_slice,
                gt_samples,
                self.num_iters_w2,
                self.max_samples_w2,
            )
            results["w2"] = w2
        return results

    def preprocess_samples(self, samples, config):
        return samples


class BNNLogger(AbstractLogger):
    def make_log_string(self, method_name: str, method_dict: dict) -> str:
        best_mean_err = jnp.max(method_dict["mean_err"])
        best_pred_energy = jnp.max(method_dict["pred_energy_err"])
        best_energy = jnp.min(method_dict["energy_err"])
        best_w2 = jnp.min(method_dict["w2"])
        str_out = (
            f"{method_name}: mean_err: {best_mean_err:.4}, pred_energy: {best_pred_energy:.4},"
            f" energy: {best_energy:.3e}, w2: {best_w2:.3e}"
        )
        return str_out


def flatten_bnn_samples(samples):
    leaves = jtu.tree_leaves(samples)
    target_shape = samples["prec_obs"].shape + (-1,)

    def flatten(x):
        return jnp.reshape(x, target_shape)

    out = jnp.concat([flatten(x) for x in leaves], axis=-1)
    assert out.shape[:-1] == target_shape[:-1]
    return out
