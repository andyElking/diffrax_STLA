import time
from typing import Callable, Optional

import jax.random as jr

from ..lmc import run_simple_lmc_numpyro
from ..methods.abstract_method import AbstractMethod


class ProgressiveLMC(AbstractMethod):
    def __init__(
        self,
        chain_len: int,
        chain_sep,
        tol,
        solver,
        get_previous_result_filename: Optional[Callable[[str], str]] = None,
    ):
        super().__init__(get_previous_result_filename)
        self.chain_len = chain_len
        self.chain_sep = chain_sep
        self.tol = tol
        self.solver = solver
        self.method_name = solver.__class__.__name__

    def run(self, key, model, model_args, result_dict, config):
        num_particles = config["num_particles"]
        start_time = time.time()
        samples, evals_per_sample = run_simple_lmc_numpyro(
            jr.key(0),
            model,
            model_args,
            num_particles,
            chain_len=self.chain_len,
            chain_sep=self.chain_sep,
            tol=self.tol,
            solver=self.solver,
        )
        wall_time = time.time() - start_time

        aux_output = {"evals_per_sample": evals_per_sample, "wall_time": wall_time}
        return samples, aux_output
