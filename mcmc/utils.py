import os

import jax
import jax.tree_util as jtu
import numpy as np
from jax import numpy as jnp, random as jr
from numpyro.infer import MCMC, NUTS


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


def compute_gt_logreg(model, model_args):
    gt_nuts = MCMC(
        NUTS(model, step_size=1.0),
        num_warmup=2**10,
        num_samples=2**13,
        num_chains=2**3,
        chain_method="vectorized",
    )
    gt_nuts.run(jr.PRNGKey(0), *model_args)
    gt_logreg = vec_dict_to_array(gt_nuts.get_samples())
    return gt_logreg


def get_bnn_gt_fun(test_args):
    def compute_gt_bnn(model, model_args):
        gt_nuts = MCMC(
            NUTS(model),
            num_warmup=2**10,
            num_samples=2**13,
            num_chains=2**3,
            chain_method="vectorized",
        )
        gt_nuts.run(jr.PRNGKey(0), *model_args)
        gt_bnn = gt_nuts.get_samples()
        return gt_bnn

    return compute_gt_bnn


def get_ground_truth(model, filename, model_args, compute_gt_fun):
    # if ground_truth is not computed, compute it
    if not os.path.exists(filename):
        gt = compute_gt_fun(model, model_args)
        # shuffle the ground truth samples
        permute = jax.jit(lambda x: jr.permutation(jr.key(0), x, axis=0))
        gt = jtu.tree_map(permute, gt)
        np.save(filename, gt)
    else:
        gt = np.load(filename, allow_pickle=True)
    return gt
