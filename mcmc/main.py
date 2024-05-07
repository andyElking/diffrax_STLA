from test.helpers import _batch_sde_solve, _batch_sde_solve_multi_y0

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


def run_lmc_numpyro(
    key,
    model,
    num_particles: int,
    chain_len: int,
    chain_sep: float = 0.1,
    tol: float = 2**-6,
    warmup_mult: float = 32.0,
    warmup_tol_mult: float = 4.0,
    use_adaptive: bool = True,
    solver: AbstractSolver = UBU3(0.1),
):
    log_p = jax.jit(model.potential_fn)
    x0 = model.param_info.z
    return run_lmc(
        key,
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
    chain_sep: float = 0.1,
    tol: float = 2**-6,
    warmup_mult: float = 32.0,
    warmup_tol_mult: float = 4.0,
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

    if use_adaptive:
        controller_warmup = PIDController(
            rtol=0.0,
            atol=warmup_tol_mult * tol,
            pcoeff=0.1,
            icoeff=0.3,
            dtmin=2**-6,
            dtmax=1.0,
        )
        solver = HalfSolver(solver)
    else:
        controller_warmup = ConstantStepSize()

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
        2**-9,
        SaveAt(t1=True),
    )
    y_warm = jtu.tree_map(
        lambda x: jnp.nan_to_num(x[:, 0], nan=0, posinf=0, neginf=0), out_warmup
    )

    t0_mcmc = 4 * chain_sep
    t1_mcmc: float = chain_len * chain_sep + t0_mcmc
    save_ts = jnp.linspace(t0_mcmc, t1_mcmc, num=chain_len, endpoint=True)
    saveat = SaveAt(ts=save_ts)
    if use_adaptive:
        dtmin = 2**-8
        bm_tol = dtmin / 2.0
        controller_mcmc = PIDController(
            rtol=0.0,
            atol=tol,
            pcoeff=0.1,
            icoeff=0.3,
            dtmin=dtmin,
            step_ts=save_ts,
            dtmax=1.0,
        )
    else:
        step_ts = jnp.linspace(0.0, t1_mcmc, num=int(t1_mcmc / tol) + 1)
        step_ts = jnp.unique(jnp.sort(jnp.concatenate((step_ts, save_ts))))
        controller_mcmc = StepTo(ts=step_ts)
        bm_tol = tol / 8.0

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

    print(
        f"LMC: "
        # f"Steps warmup: {avg_steps_warmup}, steps mcmc: {avg_steps_mcmc}, "
        f"gradient evaluations per output: {grad_evals_per_sample:.4}"
    )

    return ys_mcmc, grad_evals_per_sample
