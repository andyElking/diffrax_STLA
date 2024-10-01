from mcmc.evaluation import compute_energy, test_accuracy


def eval_gt_logreg(gt, config):
    x_test, labels_test = config["test_args"]
    size_gt_half = int(gt.shape[0] // 2)
    gt_energy_bias = compute_energy(gt[:size_gt_half], gt[size_gt_half:])
    gt_test_acc, gt_test_acc_best90 = test_accuracy(x_test, labels_test, gt)
    str_gt = (
        f"GT energy bias: {gt_energy_bias:.3e}, test acc: {gt_test_acc:.4},"
        f" test acc top 90%: {gt_test_acc_best90:.4}"
    )
    return str_gt
