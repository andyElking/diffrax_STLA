import timeit
from warnings import simplefilter


simplefilter(action="ignore", category=FutureWarning)

from test.helpers import (
    get_mlp_sde,
    SDE,
)

import jax
import jax.numpy as jnp
import jax.random as jr
from diffrax import BrownianIncrement, diffeqsolve, Euler, SaveAt
from weak_srk.w2ito import W2Ito1, W2Ito2
from weak_srk.w2ito_foster import W2ItoFoster


jax.config.update("jax_enable_x64", True)


def time_srk(
    solver, noise_dim=5, y_dim=4, dt0=0.01, n_steps=100, n_samples=10000, nn_depth=2
):
    t0 = 0.0
    t1 = dt0 * n_steps
    saveat = SaveAt(t1=True)
    mlp_sde: SDE = get_mlp_sde(
        t0, t1, jnp.float64, jr.key(0), noise_dim, y_dim, nn_depth
    )
    terms_mlp = mlp_sde.get_terms(mlp_sde.get_bm(jr.key(0), BrownianIncrement, 0.01))
    y0, args = mlp_sde.y0, mlp_sde.args

    @jax.jit
    @jax.vmap
    def solve(key):
        sol = diffeqsolve(terms_mlp, solver, t0, t1, dt0, y0, args, saveat=saveat)
        return sol.ys

    keys = jr.split(jr.key(0), n_samples)
    fun = jax.jit(lambda: solve(keys))
    # time with compilation
    time_with_compile = timeit.timeit(fun, number=1)

    # time without compilation
    number = 4
    time_without_compile = timeit.timeit(fun, number=number) / number

    return time_with_compile, time_without_compile


if __name__ == "__main__":
    n_steps = 100
    n_samples = 1000
    noise_dim = 10
    y_dim = 4
    nn_depth = 10
    dt0 = 0.01
    solvers = [Euler(), W2Ito1(jr.key(0)), W2Ito2(jr.key(0)), W2ItoFoster(jr.key(0))]

    for solver in solvers:
        time_with_compile, time_without_compile = time_srk(
            solver, noise_dim, y_dim, dt0, n_steps, n_samples, nn_depth
        )
        print(
            f"{solver.__class__.__name__}: with compile={time_with_compile:.3}, "
            f"without compile={time_without_compile:.3}, "
            f"difference={time_with_compile-time_without_compile:.3}"
        )
