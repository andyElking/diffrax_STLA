import jax
import jax.numpy as jnp
from jax import config
# config.update("jax_enable_x64", True)

from diffrax import diffeqsolve, ControlTerm, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree


def l2_dist(ys1: jax.Array, ys2: jax.Array):
    assert ys1.shape == ys2.shape
    n = ys1.shape[0]
    square_dist = jnp.square(ys1 - ys2)
    avg = 1 / n * jnp.sum(square_dist)
    return jnp.sqrt(avg)


def solutions(keys, sde, dt0, solver):
    drift, diffusion, args, y0, t0, t1, w_dim = sde
    saveat = SaveAt(ts=[t1])
    ode_term = ODETerm(drift)

    def end_value(key):
        path = VirtualBrownianTree(t0=t0, t1=t1, shape=(w_dim,), tol=2 ** -9, key=key, compute_stla=True)
        terms = MultiTerm(ode_term, ControlTerm(diffusion, path))
        sol = diffeqsolve(terms, solver, t0, t1, dt0=dt0, y0=y0, args=args, saveat=saveat)
        return sol.ys[0]

    return jax.vmap(end_value)(keys)


def solver_distance(keys, sde, solver1, dt1, solver2, dt2):
    sols1 = solutions(keys, sde, dt0=dt1, solver=solver1)
    sols2 = solutions(keys, sde, dt0=dt2, solver=solver2)
    return l2_dist(sols1, sols2)


def solver_order(keys, sde, solver, correct_solver, dt_precise, hs_num=5):
    correct_sols = solutions(keys, sde, dt0=dt_precise, solver=correct_solver)
    hs = 0.025 * jnp.power(jnp.float32(2.0), jnp.arange(0, hs_num))

    def get_single_err(h):
        sols = solutions(keys, sde, dt0=h, solver=solver)
        return l2_dist(sols, correct_sols)

    errs = jax.vmap(get_single_err)(hs)
    return hs, errs
