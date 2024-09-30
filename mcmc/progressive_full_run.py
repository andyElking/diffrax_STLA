import os
import pickle
import time
from functools import partial

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import scipy
from evaluation import (
    adjust_max_len,  # noqa: F401
    compute_w2,
    energy_distance,  # noqa: F401
    test_accuracy,  # noqa: F401
    vec_dict_to_array,  # noqa: F401
)
from get_model import get_model_and_data  # noqa: F401
from main import run_simple_lmc_numpyro  # noqa: F401
from numpyro.infer import MCMC, NUTS, Predictive  # pyright: ignore


def make_result_str(result_dict, method_name):
    best_acc = jnp.max(result_dict["test_acc"])
    best_acc90 = jnp.max(result_dict["test_acc_best90"])
    best_energy = jnp.min(result_dict["energy_err"])
    best_w2 = jnp.min(result_dict["w2"])
    str_out = (
        f"{method_name} acc: {best_acc:.4}, acc top 90%: {best_acc90:.4},"
        f" energy: {best_energy:.3e}, w2: {best_w2:.3e}"
    )
    return str_out


def run_nuts(
    model, num_particles, x_train, labels_train, x_test, labels_test, gt_logreg
):
    x0 = Predictive(model, num_samples=num_particles)(jr.key(0), x_train, labels_train)
    del x0["obs"]

    # run NUTS and record wall time
    start_nuts = time.time()
    nuts_num_warmup = 20
    chain_len_nuts = 2**6
    nuts = MCMC(
        NUTS(model),
        num_warmup=nuts_num_warmup,
        num_samples=chain_len_nuts - nuts_num_warmup,
        num_chains=num_particles,
        chain_method="vectorized",
    )
    nuts.warmup(
        jr.PRNGKey(2),
        x_train,
        labels_train,
        init_params=x0,
        extra_fields=("num_steps",),
        collect_warmup=True,
    )
    warmup_steps = jnp.reshape(
        nuts.get_extra_fields()["num_steps"], (num_particles, nuts_num_warmup)
    )
    warmup_samples = nuts.get_samples(group_by_chain=True)
    nuts.run(jr.PRNGKey(0), x_train, labels_train, extra_fields=("num_steps",))
    run_samples = nuts.get_samples(group_by_chain=True)
    time_nuts = time.time() - start_nuts
    out_nuts = jtu.tree_map(
        lambda x, y: jnp.concatenate((x, y), axis=1), warmup_samples, run_samples
    )
    run_steps = jnp.reshape(nuts.get_extra_fields()["num_steps"], (num_particles, -1))
    steps_nuts = jnp.concatenate((warmup_steps, run_steps), axis=-1)
    steps_nuts = jnp.reshape(steps_nuts, (-1,))

    result_dict_nuts = eval_progressive_logreg(
        out_nuts,
        gt_logreg,
        steps_nuts,
        x_test,
        labels_test,
        metric_eval_interval=chain_len_nuts // (2**5),
        max_samples_w2=2**11,
    )
    result_dict_nuts["time"] = time_nuts
    return result_dict_nuts


def run_progressive_logreg(
    data_name,
    log_filename,
    timestamp,
    quic_dict_filename=None,
    euler_dict_filename=None,
    nuts_dict_filename=None,
):
    name_and_date = f"{data_name}_{timestamp}"
    result_dict_filename = f"progressive_results/result_dict_{name_and_date}.pkl"

    dataset = scipy.io.loadmat("mcmc_data/benchmarks.mat")
    model, data_split = get_model_and_data(dataset, data_name)
    x_train, labels_train, x_test, labels_test = data_split

    num_particles = 2**14
    data_dim = x_train.shape[1]
    num_particles = adjust_max_len(num_particles, data_dim)

    gt_filename = f"mcmc_data/{data_name}_ground_truth.npy"

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
        f.write(f"======= {data_name} =======\n" f"{str_gt}\n")

    quic_tol = 0.01 if data_name == "splice" else 0.05

    if quic_dict_filename is None:
        # run LMC with QUICSORT and record wall time
        start_quic = time.time()
        out_quic, steps_quic = run_simple_lmc_numpyro(
            jr.key(0),
            model,
            (x_train, labels_train),
            num_particles,
            chain_len=2**5,
            chain_sep=1.0,
            tol=quic_tol,
            solver=diffrax.QUICSORT(0.1),
        )
        time_quic = time.time() - start_quic
        result_dict_quic = eval_progressive_logreg(
            out_quic, gt_logreg, steps_quic, x_test, labels_test
        )
        del out_quic
        result_dict_quic["time"] = time_quic
    else:
        with open(quic_dict_filename, "rb") as f:
            loaded_result_dict = pickle.load(f)
            result_dict_quic = loaded_result_dict["quic"]

    quic_str = make_result_str(result_dict_quic, "QUICSORT")
    print(quic_str)
    with open(log_filename, "a") as f:
        f.write(f"{quic_str}\n")

    if euler_dict_filename is None:
        # run LMC with EULER and record wall time
        start_euler = time.time()
        out_euler, steps_euler = run_simple_lmc_numpyro(
            jr.key(0),
            model,
            (x_train, labels_train),
            num_particles,
            chain_len=2**5,
            chain_sep=1.0,
            tol=quic_tol / 10,
            solver=diffrax.Euler(),
        )
        time_euler = time.time() - start_euler
        result_dict_euler = eval_progressive_logreg(
            out_euler, gt_logreg, steps_euler, x_test, labels_test
        )
        del out_euler
        result_dict_euler["time"] = time_euler
    else:
        with open(euler_dict_filename, "rb") as f:
            loaded_result_dict = pickle.load(f)
            result_dict_euler = loaded_result_dict["euler"]

    euler_str = make_result_str(result_dict_euler, "Euler")
    print(euler_str)
    with open(log_filename, "a") as f:
        f.write(f"{euler_str}\n")

    if nuts_dict_filename is None:
        result_dict_nuts = run_nuts(
            model, num_particles, x_train, labels_train, x_test, labels_test, gt_logreg
        )
    else:
        with open(nuts_dict_filename, "rb") as f:
            loaded_result_dict = pickle.load(f)
            result_dict_nuts = loaded_result_dict["nuts"]
    nuts_str = make_result_str(result_dict_nuts, "NUTS")
    print(nuts_str)
    with open(log_filename, "a") as f:
        f.write(f"{nuts_str}\n\n")

    whole_result_dict = {
        "data_name": data_name,
        "quic": result_dict_quic,
        "euler": result_dict_euler,
        "nuts": result_dict_nuts,
    }

    with open(result_dict_filename, "wb") as f:
        pickle.dump(whole_result_dict, f)


def compute_metrics(sample_slice, ground_truth, x_test, labels_test):
    energy_err = energy_distance(sample_slice, ground_truth, max_len=2**14)

    if x_test is not None and labels_test is not None:
        test_acc, test_acc_best90 = test_accuracy(x_test, labels_test, sample_slice)
    else:
        test_acc, test_acc_best90 = None, None

    return energy_err, test_acc, test_acc_best90


def eval_progressive_logreg(
    samples,
    ground_truth,
    evals_per_sample,
    x_test,
    labels_test,
    num_iters_w2=100000,
    max_samples_w2=2**11,
    metric_eval_interval=1,
):
    if isinstance(samples, dict):
        samples = vec_dict_to_array(samples)

    num_chains, chain_len, sample_dim = samples.shape

    if jnp.shape(evals_per_sample) == (num_chains * chain_len,):
        evals_per_sample = jnp.reshape(evals_per_sample, (num_chains, chain_len))
        evals_per_sample = jnp.mean(evals_per_sample, axis=0)
    elif jnp.size(evals_per_sample) == 1:
        evals_per_sample = jnp.broadcast_to(evals_per_sample, (chain_len,))
    else:
        assert False, f"evals_per_sample shape: {evals_per_sample.shape}"

    assert jnp.shape(evals_per_sample) == (
        chain_len,
    ), f"{evals_per_sample.shape} != {(chain_len,)}"

    cumulative_evals = jnp.cumsum(evals_per_sample)[::metric_eval_interval]

    samples_for_eval = samples[:, ::metric_eval_interval]

    # now we go along chain_len and compute the metrics for each step
    partial_metrics = partial(
        compute_metrics,
        ground_truth=ground_truth,
        x_test=x_test,
        labels_test=labels_test,
    )
    # vectorize over the chain_len dimension
    vec_metrics = jax.vmap(partial_metrics, in_axes=1)
    energy_err, test_acc, test_acc_best90 = vec_metrics(samples_for_eval)

    if num_iters_w2 > 0:
        # wasserstein-2 distance is done via numpy, so cannot be vectorised
        w2_list = []
        for i in range(jnp.shape(samples_for_eval)[1]):
            w2_single = compute_w2(
                samples_for_eval[:, i], ground_truth, num_iters_w2, max_samples_w2
            )
            w2_list.append(w2_single)
        w2 = jnp.array(w2_list)
    else:
        w2 = None

    result_dict = {
        "energy_err": energy_err,
        "test_acc": test_acc,
        "test_acc_best90": test_acc_best90,
        "cumulative_evals": cumulative_evals,
        "w2": w2,
    }
    return result_dict
