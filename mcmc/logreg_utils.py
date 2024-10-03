import os

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
from jax import random as jr, tree_util as jtu, numpy as jnp
from numpyro import distributions as dist, handlers
from numpyro.infer import MCMC, NUTS

from .metrics import compute_energy
from .utils import get_prior_samples


def get_model(data_dim):
    def model(x, labels):
        x_var = jnp.var(x, axis=0)
        W = numpyro.sample(
            "W",
            dist.Normal(jnp.zeros(data_dim), 0.5 / x_var),  # pyright: ignore
        )
        b = numpyro.sample("b", dist.Normal(jnp.zeros((1,)), 1))  # pyright: ignore
        logits = jnp.sum(W * x, axis=-1) + b
        obs = numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)
        return {"W": W, "b": b, "obs": obs}
    return model


def get_model_and_data(data, name):
    if name == "tbp":
        return get_taiwanese_bankruptcy_prediction()

    dset = data[name][0, 0]
    x = dset["x"]
    labels = jnp.squeeze(dset["t"])
    # labels are -1 and 1, convert to 0 and 1
    labels = (labels + 1) / 2
    n, data_dim = x.shape
    print(f"Data shape: {x.shape}")

    # randomly shuffle the data
    perm = jax.random.permutation(jr.PRNGKey(0), n)
    x = x[perm]
    labels = labels[perm]

    n_train = min(int(n * 0.8), 1000)
    x_train = x[:n_train]
    labels_train = labels[:n_train]
    x_test = x[n_train:]
    labels_test = labels[n_train:]
    print(x_train[:5, :10])
    print(labels_train[:5])
    print(f"x_train shape: {x_train.shape}, labels_train shape: {labels_train.shape}"
          f"x_train dtype: {x_train.dtype}, labels_train dtype: {labels_train.dtype}")

    return get_model(data_dim), (x_train, labels_train), (x_test, labels_test)


def get_taiwanese_bankruptcy_prediction():
    x = np.load("tbp_data/tbp_x.npy")
    labels = np.load("tbp_data/tbp_y.npy")
    labels = jnp.array(labels, dtype=jnp.float64)
    x = jnp.array(x, dtype=jnp.float64)
    x_train = x[:1000]
    labels_train = labels[:1000]
    x_test = x[1000:]
    labels_test = labels[1000:]
    data_dim = x.shape[1]
    assert data_dim == 95, f"Data dim should be 95, but is {data_dim}"
    print(x_train[:5, :10])
    print(labels_train[:5])
    print(f"x_train shape: {x_train.shape}, labels_train shape: {labels_train.shape}"
          f"x_train dtype: {x_train.dtype}, labels_train dtype: {labels_train.dtype}")
    return get_model(data_dim), (x_train, labels_train), (x_test, labels_test)


def get_gt_logreg(model, model_name, model_args, config, key):
    filename = f"ground_truth/{model_name}_ground_truth.npy"
    # if ground_truth is not computed, compute it
    if not os.path.exists(filename):
        num_chains = 2 ** 3
        x0 = get_prior_samples(key, model, model_args, num_chains)
        x0.pop("obs", None)
        x0.pop("Y", None)
        print(jtu.tree_map(lambda x: x.shape, x0))
        gt_nuts = MCMC(
            NUTS(model, step_size=1.0),
            num_warmup=2 ** 10,
            num_samples=2 ** 13,
            num_chains=num_chains,
            chain_method="vectorized",
        )
        gt_nuts.run(jr.PRNGKey(0), *model_args, init_params=x0)
        gt = vec_dict_to_array(gt_nuts.get_samples())
        # shuffle the ground truth samples
        permute = jax.jit(lambda x: jr.permutation(jr.key(0), x, axis=0))
        gt = jtu.tree_map(permute, gt)
        np.save(filename, gt)
    else:
        gt = np.load(filename)
    return gt


def dict_to_array(dct: dict):
    b = dct["b"]
    lst = [b, dct["W"]]
    if "alpha" in dct:
        alpha = dct["alpha"]
        alpha = jnp.expand_dims(alpha, alpha.ndim)
        lst = [alpha] + lst
    return jnp.concatenate(lst, axis=-1)


vec_dict_to_array = jax.jit(jax.vmap(dict_to_array, in_axes=0, out_axes=0))


def eval_gt_logreg(gt, config):
    x_test, labels_test = config["test_args"]
    size_gt_half = int(gt.shape[0] // 2)
    gt_energy_bias = compute_energy(gt[:size_gt_half], gt[size_gt_half:])
    gt_test_acc, gt_test_acc_best90 = test_accuracy(x_test, labels_test, gt)
    str_gt = (
        f"GT energy bias: {gt_energy_bias:.3e}, test acc: {gt_test_acc:.4},"
        f" test acc top 90%: {gt_test_acc_best90:.4}"
    )
    return str_gt


def predict(x, samples):
    b = samples[:, 0]
    w = samples[:, 1:]
    logits = jnp.sum(w * x, axis=-1) + b
    # apply sigmoid
    return 1.0 / (1.0 + jnp.exp(-logits))


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
