import pickle

from matplotlib import pyplot as plt  # pyright: ignore


def plot_progressive_results(result_dict, axs, column_title=None, use_ylabels=True):
    energy_err = result_dict["energy_err"]
    test_acc = result_dict["test_acc"]
    test_acc_best90 = result_dict["test_acc_best90"]
    cumulative_evals = result_dict["cumulative_evals"]
    w2 = result_dict["w2"]

    num_subplots = 2 if w2 is None else 3
    assert len(axs) == num_subplots

    axs[0].plot(cumulative_evals, energy_err)
    axs[0].set_yscale("log")
    if column_title is not None:
        axs[0].set_title(column_title)

    axs[1].plot(cumulative_evals, test_acc, label="Test accuracy")
    axs[1].plot(cumulative_evals, test_acc_best90, label="Top 90% accuracy")
    axs[1].legend()
    if use_ylabels:
        axs[0].set_ylabel("Energy distance error")
        axs[1].set_ylabel("Accuracy")

    if w2 is not None:
        axs[2].plot(cumulative_evals, w2)
        axs[2].set_yscale("log")
        axs[2].set_xlabel("Cumulative gradient evaluations")
        if use_ylabels:
            axs[2].set_ylabel("Wasserstein-2 error")
    else:
        axs[1].set_xlabel("Cumulative gradient evaluations")


def make_figs(result_dict_filename, save_name=None):
    with open(result_dict_filename, "rb") as f:
        loaded_result_dict = pickle.load(f)
    data_name = loaded_result_dict["data_name"]

    num_rows = 3 if "w2" in loaded_result_dict["quic"] else 2
    fig, axs = plt.subplots(num_rows, 3, figsize=(18, 5 * num_rows))
    fig.suptitle(data_name)
    plot_progressive_results(
        loaded_result_dict["nuts"], axs[:, 0], "NUTS", use_ylabels=True
    )
    plot_progressive_results(
        loaded_result_dict["quic"], axs[:, 1], "QUICSORT", use_ylabels=False
    )
    plot_progressive_results(
        loaded_result_dict["euler"], axs[:, 2], "Euler", use_ylabels=False
    )

    if save_name is not None:
        fig.savefig(save_name)
    return fig
