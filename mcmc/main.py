from test.helpers import _batch_sde_solve, _batch_sde_solve_multi_y0

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from diffrax import LangevinTerm


def run_sortmc(
    key,
    log_p,
    x0,
    num_particles: int,
    chain_len: int,
    chain_sep: float = 0.1,
    tol: float = 1e-4,
    warmup_mult: float = 32.0,
    warmup_tol_mult: float = 4.0,
):
    key_warmup, key_mcmc = jr.split(key, 2)
    keys_warmup = jr.split(key_warmup, num_particles)
    keys_mcmc = jr.split(key_mcmc, num_particles)
    grad_f = jax.grad(log_p)
    v0 = jnp.zeros_like(x0)
    y0 = (x0, v0)
    w_shape: tuple[int, ...] = x0.shape

    gamma, u = 1.0, 1.0

    def get_terms(bm):
        args = (gamma, u, grad_f)
        return LangevinTerm(args, bm)

    t_warmup = warmup_mult * chain_sep
    controller_warmup = diffrax.PIDController(
        rtol=0.0, atol=warmup_tol_mult * tol, pcoeff=0.1, icoeff=0.3, dtmin=2**-4
    )
    half_solver = diffrax.HalfSolver(diffrax.SORT(0.1))

    out_warmup, steps_warmup = _batch_sde_solve(
        keys_warmup,
        get_terms,
        w_shape,
        0.0,
        t_warmup,
        y0,
        None,
        half_solver,
        "space-time-time",
        0.1,
        controller_warmup,
        2**-9,
        diffrax.SaveAt(t1=True),
    )
    y_warm = jtu.tree_map(lambda x: jnp.squeeze(x, axis=1), out_warmup)

    t0_mcmc = 4 * chain_sep
    t1_mcmc = chain_len * chain_sep + t0_mcmc
    save_ts = jnp.linspace(t0_mcmc, t1_mcmc, num=chain_len, endpoint=True)
    saveat = diffrax.SaveAt(ts=save_ts)
    controller_mcmc = diffrax.PIDController(
        rtol=0.0, atol=tol, pcoeff=0.1, icoeff=0.4, dtmin=2**-6, step_ts=save_ts
    )
    out_mcmc, steps_mcmc = _batch_sde_solve_multi_y0(
        keys_mcmc,
        get_terms,
        w_shape,
        0.0,
        t1_mcmc,
        y_warm,
        None,
        half_solver,
        "space-time-time",
        chain_sep,
        controller_mcmc,
        2**-12,
        saveat,
    )

    avg_steps_warmup = jnp.mean(steps_warmup)
    avg_steps_mcmc = jnp.mean(steps_mcmc)
    steps_per_sample = (avg_steps_mcmc + avg_steps_warmup) / chain_len
    print(
        f"Steps warmup: {avg_steps_warmup}, steps mcmc: {avg_steps_mcmc},"
        f" steps per output: {steps_per_sample}"
    )

    return out_mcmc[0], steps_per_sample
