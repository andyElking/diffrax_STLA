import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from diffrax import (
    ALIGN,
    ControlTerm,
    diffeqsolve,
    Euler,
    MultiTerm,
    ODETerm,
    SaveAt,
    ShARK,
    VirtualBrownianTree,
)

from .helpers import batch_sde_solve, l2_dist, sde_solver_order


def solver_distance(keys, sde, solver1, dt1, solver2, dt2):
    sols1 = batch_sde_solve(keys, sde, dt0=dt1, solver=solver1)
    sols2 = batch_sde_solve(keys, sde, dt0=dt2, solver=solver2)
    return l2_dist(sols1, sols2)


def get_bm(sde, key, tol=2**-15, compute_stla=True):
    _, _, _, y0, _t0, _t1, w_dim = sde
    shp_dtype = jax.ShapeDtypeStruct((w_dim,), dtype=y0.dtype)
    return VirtualBrownianTree(
        t0=_t0, t1=_t1, shape=shp_dtype, tol=tol, key=key, compute_stla=compute_stla
    )


def drift(t, y, args):
    gamma, u, grad_f = args
    dim = int(y.shape[0] / 2)
    x, v = y[:dim], y[dim:]
    d_x = v
    d_v = -gamma * v - u * grad_f(x)
    d_y = jnp.array([d_x, d_v], dtype=y.dtype).flatten()
    return d_y


def diffusion(t, y, args):
    gamma, u, _ = args
    dim = int(y.shape[0] / 2)
    assert y.shape[0] == 2 * dim
    d_v = jnp.sqrt(2 * gamma * u) * jnp.ones((dim,), dtype=y.dtype)
    d_y = jnp.concatenate((jnp.zeros((dim, dim), dtype=y.dtype), jnp.diag(d_v)), axis=0)
    return d_y


def get_terms(bm):
    return MultiTerm(ODETerm(drift), ControlTerm(diffusion, bm))


t0, t1 = 0.3, 5

gamma_hosc = jnp.array([2, 0.5], dtype=jnp.float32)
u_hosc = jnp.array([0.5, 2], dtype=jnp.float32)
args_hosc = (gamma_hosc, u_hosc, lambda x: 2 * x)
y0_hosc = jnp.zeros((4,), dtype=jnp.float32)
w_dim_hosc = 2
harmonic_osc = (drift, diffusion, args_hosc, y0_hosc, t0, t1, w_dim_hosc)

grad_f_bqp = lambda x: 4 * x * (jnp.square(x) - 1)
args_bqp = (jnp.float32(0.8), jnp.float32(0.2), grad_f_bqp)
y0_bqp = jnp.zeros((2,), dtype=jnp.float32)
w_dim_bqp = 1
bqp = (drift, diffusion, args_bqp, y0_bqp, t0, t1, w_dim_bqp)


@pytest.mark.parametrize("solver", [ALIGN(0.1), ShARK()])
def test_shape(solver):
    t0, t1 = 0.3, 1.0
    for dtype in [jnp.float16, jnp.float32]:
        saveat = SaveAt(ts=jnp.linspace(t0, t1, 10, dtype=dtype))
        for dim in [1, 3]:
            u = dtype(1.0)
            gam = dtype(1.0)
            vec_u = jnp.ones((dim,), dtype=dtype)
            vec_gam = jnp.ones((dim,), dtype=dtype)
            y0 = jnp.zeros((2 * dim,), dtype=dtype)
            f = lambda x: 0.5 * x
            shp_dtype = jax.ShapeDtypeStruct((dim,), dtype)
            terms = get_terms(
                VirtualBrownianTree(
                    t0,
                    t1,
                    tol=2**-9,
                    shape=shp_dtype,
                    key=jrandom.PRNGKey(4),
                    compute_stla=True,
                )
            )
            for args in [
                (gam, u, f),
                (vec_gam, u, f),
                (gam, vec_u, f),
                (vec_gam, vec_u, f),
            ]:
                sol = diffeqsolve(
                    terms, solver, t0, t1, dt0=0.3, y0=y0, args=args, saveat=saveat
                )
                assert sol.ys.shape == (10, 2 * dim)
                assert sol.ys.dtype == dtype


@pytest.mark.parametrize("solver", [ALIGN(0.1), ShARK()])
def test_convergence(solver):
    num_samples = 1000
    keys = jrandom.split(jrandom.PRNGKey(2), num=num_samples)

    for sde in [harmonic_osc, bqp]:
        hs = 0.1 * jnp.power(jnp.float32(2.0), jnp.arange(0, 4, dtype=jnp.float32))
        _, errs, order_vs_euler = sde_solver_order(
            keys, sde, solver, Euler(), jnp.float32(0.005), hs=hs
        )
        assert errs[0] < (0.1 if isinstance(solver, ALIGN) else 0.3)
        assert order_vs_euler > 1.3

        hs = 0.025 * jnp.power(jnp.float32(2.0), jnp.arange(0, 5, dtype=jnp.float32))
        _, _, order_vs_itself = sde_solver_order(
            keys, sde, solver, solver, jnp.float32(0.005), hs=hs
        )
        assert order_vs_itself > (1.9 if isinstance(solver, ALIGN) else 1.0)
