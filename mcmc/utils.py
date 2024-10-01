import os

import jax
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
