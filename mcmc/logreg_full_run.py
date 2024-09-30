import os
import pickle

import diffrax
import jax.random as jr
import numpy as np
import scipy
from evaluation import (
    compute_w2,
    energy_distance,
    test_accuracy,
    vec_dict_to_array,
)
from get_model import get_model_and_data
from jax import numpy as jnp, tree_util as jtu
from numpyro import diagnostics  # pyright: ignore
from numpyro.infer import MCMC, NUTS, Predictive  # pyright: ignore

from mcmc import run_lmc_numpyro
from mcmc.parse_results import result_dict_to_string


def run_logreg_dataset(
    name,
    log_filename,
    timestamp,
    nuts_result_filename=None,
    quic_result_filename=None,
    euler_result_filename=None,
):
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

    num_chains = 2**8
    num_samples_per_chain = 2**8
    warmup_len = 2**7

    print("NUTS:")
    if nuts_result_filename is None:
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
        eval_nuts_str, eval_nuts_dict = eval_logreg(
            out_logreg_nuts,
            geps_nuts,
            ground_truth=gt_logreg,
            x_test=x_test,
            labels_test=labels_test,
            num_iters_w2=100000,
        )
        del out_logreg_nuts
    else:
        with open(nuts_result_filename, "rb") as f:
            eval_nuts_dict = pickle.load(f)
            geps_nuts = eval_nuts_dict["grad_evals_per_sample"]
            eval_nuts_str = result_dict_to_string(eval_nuts_dict)

    with open(log_filename, "a") as f:
        f.write(f"NUTS: {eval_nuts_str}\n\n")

    print("LMC:")
    lmc_tol = 0.04
    if name == "splice":
        lmc_tol = 0.01
    warmup_tol_mult = 2
    # Adapt chain_sep so that the total number of steps is similar to NUTS
    chain_sep = (0.4 * geps_nuts * num_samples_per_chain) * (
        lmc_tol / (num_samples_per_chain + 4 + warmup_len / warmup_tol_mult)
    )
    print(f"Target time-interval between samples for LMC: {chain_sep:.4}")
    if chain_sep < 0.1:
        chain_sep = 0.1

    # Run LMC with QUICSORT
    print("QUICSORT:")
    if quic_result_filename is None:
        out_logreg_quic, geps_quic = run_lmc_numpyro(
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
            solver=diffrax.QUICSORT(0.1),
        )

        result_str_quic, result_dict_quic = eval_logreg(
            out_logreg_quic,
            geps_quic,
            ground_truth=gt_logreg,
            x_test=x_test,
            labels_test=labels_test,
            num_iters_w2=100000,
        )
        del out_logreg_quic
    else:
        with open(quic_result_filename, "rb") as f:
            result_dict_quic = pickle.load(f)
            result_str_quic = result_dict_to_string(result_dict_quic)

    with open(log_filename, "a") as f:
        f.write(f"LMC QUICSORT: {result_str_quic}\n\n")

    # Run LMC with Euler
    print("Euler:")
    if euler_result_filename is None:
        # Euler is far less stable than QUICSORT, so we use a larger tolerance
        out_logreg_euler, geps_euler = run_lmc_numpyro(
            jr.PRNGKey(3),
            model,
            (x_train, labels_train),
            num_chains,
            num_samples_per_chain,
            chain_sep=chain_sep,
            tol=0.2 * lmc_tol,
            warmup_mult=warmup_len,
            warmup_tol_mult=warmup_tol_mult,
            use_adaptive=False,
            solver=diffrax.Euler(),
        )

        result_str_euler, result_dict_euler = eval_logreg(
            out_logreg_euler,
            geps_euler,
            ground_truth=gt_logreg,
            x_test=x_test,
            labels_test=labels_test,
            num_iters_w2=100000,
        )
        del out_logreg_euler
    else:
        with open(euler_result_filename, "rb") as f:
            result_dict_euler = pickle.load(f)
            result_str_euler = result_dict_to_string(result_dict_euler)

    with open(log_filename, "a") as f:
        f.write(f"LMC Euler: {result_str_euler}\n\n")

    results_dict = {
        "dataset_name": name,
        "QUICSORT": result_dict_quic,
        "Euler": result_dict_euler,
        "NUTS": eval_nuts_dict,
    }

    with open(f"mcmc_data/results_dict_{name}_{timestamp}.pkl", "wb") as f:
        pickle.dump(results_dict, f)


def eval_logreg(
    samples,
    evals_per_sample=None,
    ground_truth=None,
    num_iters_w2=0,
    x_test=None,
    labels_test=None,
    has_alpha=False,
):
    if isinstance(samples, dict):
        samples = vec_dict_to_array(samples)

    if has_alpha:
        samples = samples[..., 1:]

    sample_dim = samples.shape[-1]
    reshaped = jnp.reshape(samples, (-1, sample_dim))
    result_str = ""

    ess = diagnostics.effective_sample_size(samples)
    avg_ess = 1 / jnp.mean(1 / jnp.stack(jtu.tree_leaves(ess)))
    ess_per_sample = avg_ess / reshaped.shape[0]
    result_str += f"\nESS per sample: {ess_per_sample:.4}"
    if evals_per_sample is not None:
        avg_evals = jnp.mean(evals_per_sample)
        result_str += f", grad evals per sample: {avg_evals:.4}"
        # grad evals per effective sample
        gepes = avg_evals / ess_per_sample
        result_str += f", GEPS/ESS: {gepes:.4}"

    half_len = reshaped.shape[0] // 2
    energy_self = energy_distance(reshaped[:half_len], reshaped[half_len:])

    if ground_truth is not None:
        energy_gt = energy_distance(reshaped, ground_truth)
        result_str += f"\nEnergy dist vs ground truth: {energy_gt:.4}"
    else:
        energy_gt = None

    if num_iters_w2 > 0 and ground_truth is not None:
        w2 = compute_w2(reshaped, ground_truth, num_iters_w2)
        result_str += f", Wasserstein-2 error: {w2:.4}"
    else:
        w2 = None

    if x_test is not None and labels_test is not None:
        test_acc, test_acc_best90 = test_accuracy(x_test, labels_test, samples)
        result_str += (
            f"\nTest_accuracy: {test_acc:.4}, top 90% accuracy: {test_acc_best90:.4}"
        )
    else:
        test_acc, test_acc_best90 = None, None

    print(result_str)

    result_dict = {
        "ess": ess,
        "ess_per_sample": ess_per_sample,
        "energy_v_self": energy_self,
        "grad_evals_per_sample": evals_per_sample,
        "test_accuracy": test_acc,
        "top90_accuracy": test_acc_best90,
        "w2": w2,
        "energy_gt": energy_gt,
    }

    return result_str, result_dict
