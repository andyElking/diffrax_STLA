import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from diffrax import (
    ALIGN,
    diffeqsolve,
    Euler,
    LangevinTerm,
    SaveAt,
    SEA,
    ShARK,
    SRA1,
    VirtualBrownianTree,
)

from .helpers import SDE, sde_solver_order


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
            bm = VirtualBrownianTree(
                t0,
                t1,
                tol=2**-9,
                shape=shp_dtype,
                key=jrandom.PRNGKey(4),
                spacetime_levyarea=True,
            )
            for args in [
                (gam, u, f),
                (vec_gam, u, f),
                (gam, vec_u, f),
                (vec_gam, vec_u, f),
            ]:
                terms = LangevinTerm(args, bm)
                sol = diffeqsolve(
                    terms, solver, t0, t1, dt0=0.3, y0=y0, args=None, saveat=saveat
                )
                assert sol.ys.shape == (10, 2 * dim)
                assert sol.ys.dtype == dtype


def _solvers():
    # solver, order
    yield ALIGN(0.1), 2.0
    yield ShARK(), 2.0
    yield SRA1(), 2.0
    yield SEA(), 1.0


def get_harmonic_oscillator(t0=0.3, t1=15.0, dtype=jnp.float32):
    gamma_hosc = jnp.array([2, 0.5], dtype=dtype)
    u_hosc = jnp.array([0.5, 2], dtype=dtype)
    args_hosc = (gamma_hosc, u_hosc, lambda x: 2 * x)
    x0 = jnp.zeros((2,), dtype=dtype)
    v0 = jnp.zeros((2,), dtype=dtype)
    y0_hosc = (x0, v0)
    w_dim_hosc = 2

    def get_terms_hosc(bm):
        return LangevinTerm(args_hosc, bm)

    return SDE(get_terms_hosc, None, y0_hosc, t0, t1, w_dim_hosc)


def get_bqp(t0=0.3, t1=15.0, dtype=jnp.float32):
    grad_f_bqp = lambda x: 4 * x * (jnp.square(x) - 1)
    args_bqp = (dtype(0.8), dtype(0.2), grad_f_bqp)
    y0_bqp = jnp.zeros((2,), dtype=dtype)
    w_dim_bqp = 1

    def get_terms_bqp(bm):
        return LangevinTerm(args_bqp, bm)

    return SDE(get_terms_bqp, None, y0_bqp, t0, t1, w_dim_bqp)


@pytest.mark.parametrize("solver,theoretical_order", _solvers())
def test_convergence(solver, theoretical_order):
    num_samples = 100
    keys = jrandom.split(jrandom.PRNGKey(5678), num=num_samples)

    t0, t1 = 0.3, 5.1

    hosc = get_harmonic_oscillator(t0, t1, jnp.float64)
    bqp = get_bqp(t0, t1, jnp.float64)

    hs1 = jnp.power(2.0, jnp.arange(-2, -6, -1, dtype=jnp.float64))
    hs2 = jnp.power(2.0, jnp.arange(-4, -9, -1, dtype=jnp.float64))

    for sde in [hosc, bqp]:
        _, errs, order_v_euler = sde_solver_order(
            keys, sde, solver, Euler(), 2**-12, hs=hs1
        )
        _, _, order_v_self = sde_solver_order(
            keys, sde, solver, solver, 2**-12, hs=hs2
        )
        assert -0.2 < order_v_self - theoretical_order < 0.2
        assert -0.4 < order_v_euler - theoretical_order < 0.5
