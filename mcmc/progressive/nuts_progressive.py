import time
from typing import Callable, Optional

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from numpyro.infer import MCMC, NUTS, Predictive

from ..methods.abstract_method import AbstractMethod


class ProgressiveNUTS(AbstractMethod):
    def __init__(
        self,
        num_warmup: int,
        chain_len: int,
        get_previous_result_filename: Optional[Callable[[str], str]] = None,
    ):
        super().__init__(get_previous_result_filename)
        self.method_name = "NUTS"
        self.num_warmup = num_warmup
        self.chain_len = chain_len

    def run(self, key, model, model_args, result_dict, config):
        num_particles = config["num_particles"]

        key_init, key_warmup, key_run = jr.split(key, 3)
        x0 = Predictive(model, num_samples=num_particles)(key_init, *model_args)
        del x0["obs"]

        # run NUTS and record wall time
        start_nuts = time.time()
        nuts = MCMC(
            NUTS(model),
            num_warmup=self.num_warmup,
            num_samples=self.chain_len - self.num_warmup,
            num_chains=num_particles,
            chain_method="vectorized",
        )
        nuts.warmup(
            key_warmup,
            *model_args,
            init_params=x0,
            extra_fields=("num_steps",),
            collect_warmup=True,
        )
        warmup_steps = jnp.reshape(
            nuts.get_extra_fields()["num_steps"], (num_particles, self.num_warmup)
        )
        warmup_samples = nuts.get_samples(group_by_chain=True)
        nuts.run(key_run, *model_args, extra_fields=("num_steps",))
        run_samples = nuts.get_samples(group_by_chain=True)
        time_nuts = time.time() - start_nuts
        samples = jtu.tree_map(
            lambda x, y: jnp.concatenate((x, y), axis=1), warmup_samples, run_samples
        )
        run_steps = jnp.reshape(
            nuts.get_extra_fields()["num_steps"], (num_particles, -1)
        )
        steps_nuts = jnp.concatenate((warmup_steps, run_steps), axis=-1)
        steps_nuts = jnp.reshape(steps_nuts, (-1,))

        aux_output = {"evals_per_sample": steps_nuts, "wall_time": time_nuts}
        return samples, aux_output
