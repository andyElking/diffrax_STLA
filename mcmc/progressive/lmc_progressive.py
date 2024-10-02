import time
from typing import Callable, Optional

import jax.random as jr

from ..lmc import run_simple_lmc_numpyro
from ..methods.abstract_method import AbstractMethod


class ProgressiveLMC(AbstractMethod):
    def __init__(
        self,
        lmc_kwargs: dict,
        get_previous_result_filename: Optional[Callable[[str], str]] = None,
    ):
        super().__init__(get_previous_result_filename)
        self.lmc_kwargs = lmc_kwargs
        self.method_name = lmc_kwargs["solver"].__class__.__name__

    def run(self, key, model, model_args, result_dict, config):
        num_particles = config["num_particles"]
        start_time = time.time()
        samples, cumulative_evals = run_simple_lmc_numpyro(
            jr.key(0), model, model_args, num_particles, **self.lmc_kwargs
        )
        wall_time = time.time() - start_time

        aux_output = {"cumulative_evals": cumulative_evals, "wall_time": wall_time}
        return samples, aux_output
