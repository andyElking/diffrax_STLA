import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import pytest
from diffrax import (
    ALIGN,
    ControlTerm,
    diffeqsolve,
    Euler,
    MultiTerm,
    ODETerm,
    SaveAt,
    SEA,
    ShARK,
    SRA1,
    VirtualBrownianTree,
)

from .helpers import sde_solver_order


def get_bm(sde, key, tol=2**-15, spacetime_levyarea=True):
    _, _, _, y0, _t0, _t1, w_dim = sde
    dtype = jtu.tree_leaves(y0)[0].dtype
    shp_dtype = jax.ShapeDtypeStruct((w_dim,), dtype=dtype)
    return VirtualBrownianTree(
        t0=_t0,
        t1=_t1,
        shape=shp_dtype,
        tol=tol,
        key=key,
        spacetime_levyarea=spacetime_levyarea,
    )


def langevin_drift(t, y, args):
    gamma, u, grad_f = args
    x, v = y
    d_x = v
    d_v = -gamma * v - u * grad_f(x)
    d_y = (d_x, d_v)
    return d_y


def langevin_diffusion(t, y, args):
    gamma, u, _ = args
    dtype = y[0].dtype
    if y[0].ndim == 0:
        zeros = jnp.zeros((), dtype=dtype)
    else:
        assert y[0].ndim == 1
        dim = y[0].shape[0]
        zeros = jnp.zeros((dim, dim), dtype=dtype)
    d_v = jnp.sqrt(2 * gamma * u) * jnp.ones(y[0].shape, dtype=dtype)
    d_y = (zeros, jnp.diag(d_v))
    return d_y


def get_terms(bm):
    return MultiTerm(ODETerm(langevin_drift), ControlTerm(langevin_diffusion, bm))


@pytest.mark.parametrize("solver", [ALIGN(0.1), ShARK()])
def test_shape(solver):
    t0, t1 = 0.3, 1.0
    for dtype in [jnp.float16, jnp.float32, jnp.float64]:
        saveat = SaveAt(ts=jnp.linspace(t0, t1, 10, dtype=dtype))
        for dim in [1, 3]:
            u = dtype(1.0)
            gam = dtype(1.0)
            vec_u = jnp.ones((dim,), dtype=dtype)
            vec_gam = jnp.ones((dim,), dtype=dtype)
            x0 = jnp.zeros((dim,), dtype=dtype)
            v0 = jnp.zeros((dim,), dtype=dtype)
            y0 = (x0, v0)
            f = lambda x: 0.5 * x
            shp_dtype = jax.ShapeDtypeStruct((dim,), dtype)
            terms = get_terms(
                VirtualBrownianTree(
                    t0,
                    t1,
                    tol=2**-9,
                    shape=shp_dtype,
                    key=jrandom.PRNGKey(4),
                    spacetime_levyarea=True,
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


def _solvers():
    # solver, order
    yield ALIGN(0.1), 2.0
    yield ShARK(), 2.0
    yield SRA1(), 2.0
    yield SEA(), 1.0


@pytest.mark.parametrize("solver,theoretical_order", _solvers())
def test_convergence(solver, theoretical_order):
    num_samples = 100
    keys = jrandom.split(jrandom.PRNGKey(5678), num=num_samples)

    t0, t1 = 0.3, 5.1

    gamma_hosc = jnp.array([2, 0.5], dtype=jnp.float64)
    u_hosc = jnp.array([0.5, 2], dtype=jnp.float64)
    args_hosc = (gamma_hosc, u_hosc, lambda x: 2 * x)
    y0_hosc = jnp.zeros((4,), dtype=jnp.float64)
    w_dim_hosc = 2
    harmonic_osc = (
        langevin_drift,
        langevin_diffusion,
        args_hosc,
        y0_hosc,
        t0,
        t1,
        w_dim_hosc,
    )

    grad_f_bqp = lambda x: 4 * x * (jnp.square(x) - 1)
    args_bqp = (jnp.float64(0.8), jnp.float64(0.2), grad_f_bqp)
    y0_bqp = jnp.zeros((2,), dtype=jnp.float64)
    w_dim_bqp = 1
    bqp = (langevin_drift, langevin_diffusion, args_bqp, y0_bqp, t0, t1, w_dim_bqp)

    hs1 = jnp.power(2.0, jnp.arange(-2, -6, -1, dtype=jnp.float64))
    hs2 = jnp.power(2.0, jnp.arange(-4, -9, -1, dtype=jnp.float64))

    for sde in [harmonic_osc, bqp]:
        _, errs, order_v_euler = sde_solver_order(
            keys, sde, solver, Euler(), 2**-12, hs=hs1
        )
        _, _, order_v_self = sde_solver_order(
            keys, sde, solver, solver, 2**-12, hs=hs2
        )
        assert -0.2 < order_v_self - theoretical_order < 0.2
        assert -0.4 < order_v_euler - theoretical_order < 0.5
