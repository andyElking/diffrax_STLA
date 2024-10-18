import glob
import os.path
import pickle

import matplotlib  # pyright: ignore
import numpy as np
from matplotlib import pyplot as plt  # pyright: ignore


def plot_progressive_results(
    result_dict, axs, label=None, plot_accuracy=True, width=float("inf")
):
    cumulative_evals = result_dict["cumulative_evals"]
    # we only use the entries with cumulative_evals <= width
    num_entries = np.sum(cumulative_evals <= width * 1.2)
    cumulative_evals = cumulative_evals[:num_entries]

    energy_err = result_dict["energy_err"][:num_entries]
    w2 = result_dict["w2"]

    if len(axs) == 1:
        axs = [axs]
    i = 0
    axs[i].plot(cumulative_evals, energy_err, label=label)
    axs[i].set_yscale("log")
    axs[i].set_ylabel("Energy distance error")

    if plot_accuracy:
        i += 1
        test_acc = result_dict["test_acc"]
        test_acc = test_acc[:num_entries]
        axs[i].plot(cumulative_evals, test_acc, label=label)
        axs[i].set_ylabel("Accuracy")

    if w2 is not None:
        i += 1
        w2 = w2[:num_entries]
        axs[i].plot(cumulative_evals, w2, label=label)
        axs[i].set_yscale("log")
        axs[i].set_ylabel("Wasserstein-2 error")
    axs[-1].set_xlabel("Number of function evaluations")


def plot_bnn_results(result_dict, axs, label=None):
    energy_err = result_dict["energy_err"]
    mean_err = result_dict["mean_err"]
    pred_energy_err = result_dict["pred_energy_err"]
    cumulative_evals = result_dict["cumulative_evals"]
    w2 = result_dict["w2"]

    num_subplots = 4 if w2 is not None else 3
    assert len(axs) == num_subplots

    axs[0].plot(cumulative_evals, energy_err, label=label)
    axs[0].set_yscale("log")

    axs[1].plot(cumulative_evals, mean_err, label=label)
    axs[1].set_ylabel("Mean error")

    axs[2].plot(cumulative_evals, pred_energy_err, label=label)
    axs[2].set_yscale("log")
    axs[2].set_ylabel("Energy error of model predictions")

    axs[-1].set_xlabel("Number of function evaluations")

    if w2 is not None:
        axs[3].plot(cumulative_evals, w2, label=label)
        axs[3].set_yscale("log")
        axs[3].set_ylabel("Wasserstein-2 error")


def make_figs(
    result_dict_filename, save_name=None, plot_accuracy=True, exclude_methods=()
):
    matplotlib.rcParams.update({"font.size": 15})
    with open(result_dict_filename, "rb") as f:
        result_dict = pickle.load(f)
    # data_name = result_dict["model_name"]
    num_rows = 1
    num_rows += 1 if plot_accuracy else 0
    num_rows += 1 if "w2" in result_dict["QUICSORT"] else 0
    fig, axs = plt.subplots(num_rows, 1, figsize=(7, 5 * num_rows))
    # fig.suptitle(data_name)
    # first find the width of the plot
    width = 0
    # "model_name" is the only key that is not a method
    exclude_methods = set(exclude_methods) | {"model_name"}
    for method, value in result_dict.items():
        if method in exclude_methods:
            continue
        method_width = value["cumulative_evals"][-1]
        method_width = method_width if method == "QUICSORT" else method_width / 3
        width = max(width, method_width)

    # now plot the results
    for method, value in result_dict.items():
        if method in exclude_methods:
            # "model_name" is the only key that is not a method
            continue
        plot_progressive_results(value, axs, method, plot_accuracy, width)

    for i in range(num_rows):
        axs[i].set_xlim(0, width)
        axs[i].legend()
        # the lower ylim is set so that even the values to the right of
        # width are in range, so the lower portion of the plot is just empty.
        # We must set the lower ylim to what is actually displayed.
        axs[i].set_ylim(auto=True)

    fig.tight_layout()

    if save_name is not None:
        fig.savefig(save_name)
    return fig


if __name__ == "__main__":
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
        # "tbp",
        # "isolet_ab",
    ]
    for name in names:
        # search for a file of the form
        # f"progressive_results/result_dict_{name}_{timestamp}.pkl"
        filenames = glob.glob(f"progressive_results/good_results/{name}_*.pkl")
        filenames.sort(key=os.path.getmtime)
        latest_dict = filenames[-1]
        print(f"Plotting {latest_dict}")
        save_name = f"progressive_results/good_plots/{name}_final.pdf"
        # save_name = (f"progressive_results/good_plots/"
        #              f"paper_versions/{name}_paper_version.pdf")
        figs = make_figs(latest_dict, save_name, True, ("UBU",))
