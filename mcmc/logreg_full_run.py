import os
import pickle

import jax.random as jr
import numpy as np
import scipy
from evaluation import (
    energy_distance,
    eval_logreg,
    flatten_samples,
    test_accuracy,
    vec_dict_to_array,
)
from get_model import get_model_and_data
from numpyro.infer import MCMC, NUTS, Predictive  # pyright: ignore

from mcmc import run_lmc_numpyro


def run_logreg_dataset(name, log_filename, results_dict_filename=None):
    dataset = scipy.io.loadmat("mcmc_data/benchmarks.mat")
    model, data_split = get_model_and_data(dataset, name)
    x_train, labels_train, x_test, labels_test = data_split

    gt_filename = f"mcmc_data/{name}_ground_truth.npy"

    # if ground_truth is not computed, compute it
    if not os.path.exists(gt_filename):
        gt_nuts = MCMC(
            NUTS(model, step_size=1.0),
            num_warmup=2**10,
            num_samples=2**13,
            num_chains=2**3,
            chain_method="vectorized",
        )
        gt_nuts.run(jr.PRNGKey(0), x_train, labels_train)
        gt_logreg = vec_dict_to_array(gt_nuts.get_samples())
        np.save(gt_filename, gt_logreg)
    else:
        gt_logreg = np.load(gt_filename)

    size_gt_half = int(gt_logreg.shape[0] // 2)
    gt_energy_bias = energy_distance(gt_logreg[:size_gt_half], gt_logreg[size_gt_half:])
    gt_test_acc, gt_test_acc_best90 = test_accuracy(x_test, labels_test, gt_logreg)
    str_gt = (
        f"GT energy bias: {gt_energy_bias:.3e}, test acc: {gt_test_acc:.4},"
        f" test acc top 90%: {gt_test_acc_best90:.4}"
    )
    print(str_gt)
    with open(log_filename, "a") as f:
        f.write(f"======= {name} =======\n" f"{str_gt}\n")

    num_chains = 2**7
    num_samples_per_chain = 2**8
    warmup_len = 2**7

    x0 = Predictive(model, num_samples=num_chains)(jr.key(0), x_train, labels_train)
    del x0["obs"]

    nuts = MCMC(
        NUTS(model),
        num_warmup=warmup_len,
        num_samples=num_samples_per_chain,
        num_chains=num_chains,
        chain_method="vectorized",
    )
    nuts.warmup(
        jr.PRNGKey(2),
        x_train,
        labels_train,
        extra_fields=("num_steps",),
        collect_warmup=True,
    )
    warmup_steps = sum(nuts.get_extra_fields()["num_steps"])
    nuts.run(jr.PRNGKey(2), x_train, labels_train, extra_fields=("num_steps",))
    out_logreg_nuts = nuts.get_samples(group_by_chain=True)
    num_steps_nuts = sum(nuts.get_extra_fields()["num_steps"]) + warmup_steps
    geps_nuts = num_steps_nuts / (num_chains * num_samples_per_chain)
    print("NUTS:")
    eval_nuts_str, eval_nuts_dict = eval_logreg(
        out_logreg_nuts,
        geps_nuts,
        ground_truth=gt_logreg,
        x_test=x_test,
        labels_test=labels_test,
        num_iters_w2=100000,
    )

    with open(log_filename, "a") as f:
        f.write(f"NUTS: {eval_nuts_str}\n\n")

    # Run LMC
    print("LMC:")
    lmc_tol = 0.02
    warmup_tol_mult = 4
    # Adapt chain_sep so that the total number of steps is similar to NUTS
    chain_sep = (0.4 * num_steps_nuts / num_chains) * (
        lmc_tol / (num_samples_per_chain + 4 + warmup_len / warmup_tol_mult)
    )
    print(f"Target chain separation: {chain_sep:.4}")
    if chain_sep < 0.1:
        chain_sep = 0.1

    out_logreg_lmc, geps_lmc = run_lmc_numpyro(
        jr.PRNGKey(3),
        model,
        (x_train, labels_train),
        num_chains,
        num_samples_per_chain,
        chain_sep=chain_sep,
        tol=lmc_tol,
        warmup_mult=warmup_len,
        warmup_tol_mult=warmup_tol_mult,
        use_adaptive=False,
    )

    eval_lmc_str, eval_lmc_dict = eval_logreg(
        out_logreg_lmc,
        geps_lmc,
        ground_truth=gt_logreg,
        x_test=x_test,
        labels_test=labels_test,
        num_iters_w2=100000,
    )

    with open(log_filename, "a") as f:
        f.write(f"LMC: {eval_lmc_str}\n\n")

    # Compute energy distance between the two methods
    lmc_flat = flatten_samples(out_logreg_lmc)
    nuts_flat = flatten_samples(out_logreg_nuts)
    energy_dist = energy_distance(lmc_flat, nuts_flat)
    print(f"Energy distance between LMC and NUTS: {energy_dist:.5}")

    with open(log_filename, "a") as f:
        f.write(f"Energy distance: {energy_dist:.5}\n\n\n")

    results_dict = {
        "dataset_name": name,
        "LMC": eval_lmc_dict,
        "NUTS": eval_nuts_dict,
        "Energy distance": energy_dist,
    }

    if results_dict_filename is not None:
        with open(results_dict_filename, "wb") as f:
            pickle.dump(results_dict, f)
