import math
from test.helpers import _batch_sde_solve, _batch_sde_solve_multi_y0

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from diffrax import (
    AbstractSolver,
    ConstantStepSize,
    HalfSolver,
    LangevinTerm,
    PIDController,
    SaveAt,
    SORT,
    SpaceTimeTimeLevyArea,
    StepTo,
    UBU3,
)
from jaxtyping import PyTree
from numpyro.infer.util import initialize_model  # pyright: ignore


def run_lmc_numpyro(
    key,
    model,
    model_args,
    num_particles: int,
    chain_len: int,
    chain_sep: float = 0.1,
    tol: float = 2**-6,
    warmup_mult: float = 32.0,
    warmup_tol_mult: float = 4.0,
    use_adaptive: bool = True,
    solver: AbstractSolver = UBU3(0.1),
):
    model_key, lmc_key = jr.split(key, 2)
    model_info = initialize_model(model_key, model, model_args=model_args)
    log_p = jax.jit(model_info.potential_fn)
    x0 = model_info.param_info.z
    return run_lmc(
        lmc_key,
        log_p,
        x0,
        num_particles,
        chain_len,
        chain_sep,
        tol,
        warmup_mult,
        warmup_tol_mult,
        use_adaptive,
        solver,
    )


def run_lmc(
    key,
    log_p,
    x0,
    num_particles: int,
    chain_len: int,
    chain_sep: float = 0.5,
    tol: float = 2**-6,
    warmup_mult: float = 32.0,
    warmup_tol_mult: float = 32.0,
    use_adaptive: bool = True,
    solver: AbstractSolver = UBU3(0.1),
):
    key_warmup, key_mcmc = jr.split(key, 2)
    keys_warmup = jr.split(key_warmup, num_particles)
    keys_mcmc = jr.split(key_mcmc, num_particles)
    grad_f = jax.jit(jax.grad(log_p))
    v0 = jtu.tree_map(lambda x: jnp.zeros_like(x), x0)
    y0 = (x0, v0)
    w_shape: PyTree[jax.ShapeDtypeStruct] = jtu.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), x0
    )

    gamma, u = 1.0, 1.0

    def get_terms(bm):
        args = (gamma, u, grad_f)
        return LangevinTerm(args, bm, x0)

    t_warmup = warmup_mult * chain_sep
    tol_warmup = warmup_tol_mult * tol

    t0_mcmc = 4 * chain_sep
    t1_mcmc: float = (chain_len - 1) * chain_sep + t0_mcmc
    save_ts = jnp.linspace(t0_mcmc, t1_mcmc, num=chain_len, endpoint=True)
    saveat = SaveAt(ts=save_ts)

    if use_adaptive:
        dtmin = 2**-9
        dtmin_warmup = 2**-6
        bm_tol_warmup = dtmin_warmup / 2.0
        bm_tol = dtmin / 2.0
        controller_warmup = PIDController(
            rtol=0.0,
            atol=tol_warmup,
            pcoeff=0.1,
            icoeff=0.4,
            dtmin=dtmin_warmup,
            dtmax=1.0,
        )
        pid_mcmc = PIDController(
            rtol=0.0,
            atol=tol,
            pcoeff=0.1,
            icoeff=0.4,
            dtmin=dtmin,
            dtmax=0.2,
        )
        controller_mcmc = diffrax.JumpStepWrapper(pid_mcmc, step_ts=save_ts)

        if not isinstance(solver, diffrax.ShARK):
            solver = HalfSolver(solver)
    else:
        controller_warmup = ConstantStepSize()
        steps_per_sample = int(math.ceil(chain_sep / tol))
        num_steps = (chain_len + 3) * steps_per_sample + 1
        step_ts = jnp.linspace(0.0, t1_mcmc, num=num_steps, endpoint=True)
        num_steps_before_t0 = 4 * steps_per_sample
        save_ts = step_ts[num_steps_before_t0::steps_per_sample]
        assert save_ts.shape == (
            chain_len,
        ), f"{save_ts.shape}, expected {(chain_len,)}"

        controller_mcmc = StepTo(ts=step_ts)
        bm_tol = tol / 4.0
        bm_tol_warmup = tol_warmup / 4.0

    out_warmup, steps_warmup = _batch_sde_solve(
        keys_warmup,
        get_terms,
        w_shape,
        0.0,
        t_warmup,
        y0,
        None,
        solver,
        SpaceTimeTimeLevyArea,
        tol_warmup,
        controller_warmup,
        bm_tol_warmup,
        SaveAt(t1=True),
        use_progress_meter=True,
        use_vbt=True,
    )
    y_warm = jtu.tree_map(
        lambda x: jnp.nan_to_num(x[:, 0], nan=0, posinf=0, neginf=0), out_warmup
    )

    out_mcmc, steps_mcmc = _batch_sde_solve_multi_y0(
        keys_mcmc,
        get_terms,
        w_shape,
        0.0,
        t1_mcmc,
        y_warm,
        None,
        solver,
        SpaceTimeTimeLevyArea,
        None,
        controller_mcmc,
        bm_tol,
        saveat,
        use_progress_meter=True,
        use_vbt=True,
    )
    ys_mcmc = out_mcmc[0]
    ys_mcmc = jtu.tree_map(
        lambda x: jnp.nan_to_num(x, nan=0, posinf=0, neginf=0), ys_mcmc
    )

    avg_steps_warmup = jnp.mean(steps_warmup)
    avg_steps_mcmc = jnp.mean(steps_mcmc)
    grad_evals_per_sample = (avg_steps_mcmc + avg_steps_warmup) / chain_len
    # When a HalfSolver is used, the number of gradient evaluations is tripled,
    # but the output of batch_sde_solve already accounts for this.
    if isinstance(solver, (SORT, UBU3)):
        grad_evals_per_sample *= 2

    # print(
    #     f"LMC: "
    #     # f"Steps warmup: {avg_steps_warmup}, steps mcmc: {avg_steps_mcmc}, "
    #     f"gradient evaluations per output: {grad_evals_per_sample:.4}"
    # )

    return ys_mcmc, grad_evals_per_sample
