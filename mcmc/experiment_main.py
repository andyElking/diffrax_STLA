import pickle
from typing import Callable, Optional

from .evaluation import AbstractEvaluator
from .logging import AbstractLogger
from .methods import AbstractMethod
from .utils import get_ground_truth


def run_experiment(
    key,
    model,
    model_args,
    model_name,
    methods: list[AbstractMethod],
    config,
    evaluator: AbstractEvaluator,
    logger: AbstractLogger,
    gt_dirname: Optional[str],
    gt_eval_fun: Optional[Callable],
    get_result_filename: Optional[Callable[[str], str]] = None,
):
    gt_str = ""
    if gt_dirname is not None:
        gt_filename = f"{gt_dirname}/{model_name}_ground_truth.npy"
        gt = get_ground_truth(model, gt_filename, *model_args)
        if gt_eval_fun is not None:
            gt_str = gt_eval_fun(gt, config)
    else:
        gt = None

    logger.start_model_section(model_name, gt_str)
    result_dict = {"model_name": model_name}

    for method in methods:
        loaded_dict = method.previous_results(model_name)
        if loaded_dict is None:
            samples, aux_output = method.run(
                key, model, model_args, result_dict, config
            )
            method_dict = evaluator.eval(samples, aux_output, gt, config, None)
            del samples, aux_output
        else:
            method_dict = loaded_dict
        result_dict[method.method_name] = method_dict
        logger.log_method(method.method_name, method_dict)

    if get_result_filename is not None:
        result_filename = get_result_filename(model_name)
        with open(result_filename, "wb") as f:
            pickle.dump(result_dict, f)
