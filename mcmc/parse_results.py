import math
import pickle

import numpy as np


def dict_to_latex(result_dict):
    eps = result_dict["ess_per_sample"]
    grad_evals = result_dict["grad_evals_per_sample"]
    log_energy = math.log(result_dict["energy_gt"])
    log_w2 = math.log(result_dict["w2"])
    test_accuracy = result_dict["test_accuracy"]
    top90_accuracy = result_dict["top90_accuracy"]
    return (
        f"{grad_evals:.2} & {eps:.2} & {log_energy:.2} & {log_w2:.2} &"
        f" {test_accuracy:.2} & {top90_accuracy:.2}"
    )


names = [
    "banana",
    "breast_cancer",
    "diabetis",
    "flare_solar",
    "german",
    "heart",
    "image",
    "ringnorm",
    "splice",
    "thyroid",
    "titanic",
    "twonorm",
    "waveform",
]


def result_dicts_to_latex(dict_filename, output_filename):
    str = "Results\n\n\n"

    reverse_names = names[::-1]
    with open(dict_filename, "rb") as f:
        for name in reverse_names:
            result_dict = pickle.load(f)
            print(name)
            data_name = result_dict["dataset_name"]
            assert data_name == name, f"Expected {name}, got {data_name}"
            quic_result = result_dict["QUICSORT"]
            nuts_result = result_dict["NUTS"]
            euler_result = result_dict["Euler"]

            str += f"{data_name}\n"
            str += f"QUICSORT & {dict_to_latex(quic_result)} \\\\\n"
            str += f"Euler & {dict_to_latex(euler_result)} \\\\\n"
            str += f"NUTS & {dict_to_latex(nuts_result)} \\\\\n"
            str += "\n\n"

    with open(output_filename, "w") as f:
        f.write(str)


def result_dict_to_string(result_dict):
    result_str = ""
    ess_per_sample = result_dict["ess_per_sample"]
    result_str += f"\nESS per sample: {ess_per_sample:.4}"
    evals_per_sample = result_dict["grad_evals_per_sample"]
    if evals_per_sample is not None:
        avg_evals = np.mean(evals_per_sample)
        result_str += f", grad evals per sample: {avg_evals:.4}"
        # grad evals per effective sample
        gepes = avg_evals / ess_per_sample
        result_str += f", GEPS/ESS: {gepes:.4}"

    energy_gt = result_dict["energy_gt"]
    if energy_gt is not None:
        result_str += f"\nEnergy dist vs ground truth: {energy_gt:.4}"

    w2 = result_dict["w2"]
    if w2 is not None:
        result_str += f", Wasserstein-2 error: {w2:.4}"

    test_acc = result_dict["test_accuracy"]
    if test_acc is not None:
        test_acc_best90 = result_dict["top90_accuracy"]
        result_str += (
            f"\nTest_accuracy: {test_acc:.4}, top 90% accuracy: {test_acc_best90:.4}"
        )

    return result_str


if __name__ == "__main__":
    dicts_filename = "mcmc_data/results_dict_2024-09-29_16-43-33.pkl"
    output_filename = "mcmc_data/latex_string_2024-09-29_16-43-33.txt"
    result_dicts_to_latex(dicts_filename, output_filename)
