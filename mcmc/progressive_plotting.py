import glob
import os.path
import pickle

from matplotlib import pyplot as plt  # pyright: ignore


def plot_progressive_results(result_dict, axs, label=None, use_ylabels=True):
    energy_err = result_dict["energy_err"]
    test_acc = result_dict["test_acc"]
    cumulative_evals = result_dict["cumulative_evals"]
    w2 = result_dict["w2"]

    num_subplots = 2 if w2 is None else 3
    assert len(axs) == num_subplots

    axs[0].plot(cumulative_evals, energy_err, label=label)
    axs[0].set_yscale("log")

    axs[1].plot(cumulative_evals, test_acc, label=label)
    if use_ylabels:
        axs[0].set_ylabel("Energy distance error")
        axs[1].set_ylabel("Accuracy")

    if w2 is not None:
        axs[2].plot(cumulative_evals, w2, label=label)
        axs[2].set_yscale("log")
        axs[2].set_xlabel("Number of function evaluations")
        if use_ylabels:
            axs[2].set_ylabel("Wasserstein-2 error")
    else:
        axs[1].set_xlabel("Number of function evaluations")


def make_figs(result_dict_filename, save_name=None):
    with open(result_dict_filename, "rb") as f:
        result_dict = pickle.load(f)
    data_name = result_dict["data_name"]
    num_rows = 3 if "w2" in result_dict["quic"] else 2
    fig, axs = plt.subplots(num_rows, 1, figsize=(7, 5 * num_rows))
    fig.suptitle(data_name)
    plot_progressive_results(result_dict["nuts"], axs, "NUTS", use_ylabels=True)
    plot_progressive_results(result_dict["quic"], axs, "QUICSORT", use_ylabels=False)
    plot_progressive_results(result_dict["euler"], axs, "Euler", use_ylabels=False)

    width = result_dict["quic"]["cumulative_evals"][-1]
    axs[0].set_xlim(0, width)
    axs[1].set_xlim(0, width)

    axs[0].legend()
    axs[1].legend()
    if num_rows == 3:
        axs[2].set_xlim(0, width)
        axs[2].legend()

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
    ]
    for name in names:
        # search for a file of the form
        # f"progressive_results/result_dict_{name}_{timestamp}.pkl"
        filenames = glob.glob(f"progressive_results/result_dict_{name}_*.pkl")
        filenames.sort(key=os.path.getmtime)
        latest_dict = filenames[-1]
        print(f"Plotting {latest_dict}")
        save_name = f"progressive_results/plots/fig_{name}.pdf"
        figs = make_figs(latest_dict, save_name)
