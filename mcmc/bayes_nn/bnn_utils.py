import os

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random as jr
from numpyro.infer import MCMC, NUTS

from ..metrics import compute_energy
from .bnn_evaluator import bnn_pred_error, flatten_bnn_samples, vec_predict


# create artificial regression dataset
def get_data(N=50, D_X=3, sigma_obs=0.05, N_test=500):
    D_Y = 1  # create 1d outputs
    np.random.seed(0)
    X = jnp.linspace(-1, 1, N)
    X = jnp.power(X[:, np.newaxis], jnp.arange(D_X))
    W = 0.5 * np.random.randn(D_X)

    def get_y(x):
        return jnp.dot(x, W) + 0.5 * jnp.power(0.5 + x[:, 1], 2.0) * jnp.sin(
            4.0 * x[:, 1]
        )

    Y = get_y(X)
    Y += sigma_obs * np.random.randn(N)
    Y = Y[:, np.newaxis]
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    assert X.shape == (N, D_X)
    assert Y.shape == (N, D_Y)

    X_test = jnp.linspace(-1.3, 1.3, N_test)
    X_test = jnp.power(X_test[:, np.newaxis], jnp.arange(D_X))
    Y_test = get_y(X_test)
    assert X_test.shape == (N_test, D_X)
    assert Y_test.shape == (N_test,), f"Expected {(N_test,)}, got {Y_test.shape}"

    return X, Y, X_test, Y_test


def model(X, Y, D_H, D_Y=1):
    N, D_X = X.shape

    # sample first layer (we put unit normal priors on all weights)
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H))))
    assert w1.shape == (D_X, D_H), f"Expected shape {(D_X, D_H)}, got {w1.shape}"
    z1 = jnp.tanh(jnp.matmul(X, w1))  # <= first layer of activations
    assert z1.shape == (N, D_H)

    # sample second layer
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    assert w2.shape == (D_H, D_H)
    z2 = jnp.tanh(jnp.matmul(z1, w2))  # <= second layer of activations
    assert z2.shape == (N, D_H)

    # sample final layer of weights and neural network output
    w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y))))
    assert w3.shape == (D_H, D_Y)
    z3 = jnp.matmul(z2, w3)  # <= output of the neural network
    assert z3.shape == (N, D_Y)

    if Y is not None:
        assert z3.shape == Y.shape

    # we put a prior on the observation noise
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # observe data
    with numpyro.plate("data", N):
        # note we use to_event(1) because each observation has shape (1,)
        numpyro.sample("Y", dist.Normal(z3, sigma_obs).to_event(1), obs=Y)


def get_model_and_data(N=50, D_X=3, sigma_obs=0.05, N_test=500):
    X, Y, X_test, Y_test = get_data(N, D_X, sigma_obs, N_test)
    D_H = 4
    model_args = X, Y, D_H
    test_args = X_test, Y_test, D_H
    return model, model_args, test_args


def get_gt_bnn(model, model_name, model_args, config, key):
    filename_samples = f"ground_truth/{model_name}_gt_samples.npy"
    filename_pred = f"ground_truth/{model_name}_gt_pred.npy"
    # if ground_truth is not computed, compute it
    if not os.path.exists(filename_samples):
        gt_nuts = MCMC(
            NUTS(model),
            num_warmup=2**2,
            num_samples=2**4,
            num_chains=2**1,
            chain_method="vectorized",
        )
        gt_nuts.run_sde(jr.PRNGKey(0), *model_args)
        gt_samples = gt_nuts.get_samples()
        print(
            f"gt_samples before flatten: {jtu.tree_map(lambda x: x.shape, gt_samples)}"
        )
        # Now use test_args and gt_bnn samples to compute predictions
        gt_pred = vec_predict(model, key, gt_samples, config["test_args"])
        print(f"gt_pred shape: {gt_pred.shape}")
        gt_samples = flatten_bnn_samples(gt_samples)
        # shuffle the ground truth samples
        permute = jax.jit(lambda x: jr.permutation(jr.key(0), x, axis=0))
        gt_samples = permute(gt_samples)
        gt_pred = permute(gt_pred)
        np.save(filename_samples, gt_samples)
        np.save(filename_pred, gt_pred)
    else:
        assert os.path.exists(filename_pred)
        gt_samples = np.load(filename_samples)
        gt_pred = np.load(filename_pred)
    return gt_samples, gt_pred


def eval_gt_bnn(gt, config):
    gt_samples, gt_pred = gt
    size_gt_half = int(gt_samples.shape[0] // 2)
    smp_energy_bias = compute_energy(
        gt_samples[:size_gt_half], gt_samples[size_gt_half:]
    )
    y_true = config["test_args"][1]
    y1 = gt_pred[:size_gt_half]
    y2 = gt_pred[size_gt_half:]
    mean_err, pred_energy_err = bnn_pred_error(y1, y2, y_true)
    str_gt = (
        f"sample energy bias: {smp_energy_bias:.3e},"
        f" mean_err: {mean_err:.4}, pred_energy_err: {pred_energy_err:.4}"
    )
    return str_gt
