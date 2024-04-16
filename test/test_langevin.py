import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from diffrax import diffeqsolve, LangevinTerm, SaveAt, VirtualBrownianTree

from .helpers import (
    _abstract_la_to_la,
    get_bqp,
    get_harmonic_oscillator,
    simple_sde_order,
)


def _solvers():
    # solver, order
    yield diffrax.ALIGN(0.1), 2.0
    yield diffrax.ShARK(), 2.0
    yield diffrax.SRA1(), 2.0
    yield diffrax.SORT(0.01), 3.0
    yield diffrax.ShOULD(0.01), 3.0


@pytest.mark.parametrize("solver,order", _solvers())
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.float64])
@pytest.mark.parametrize("dim", [1, 3])
def test_shape(solver, order, dtype, dim):
    if dtype == jnp.float16 and isinstance(solver, (diffrax.SORT, diffrax.ShOULD)):
        pytest.skip(
            "Due to the use of multivariate normal in the the computation"
            " of space-time-time Levy area, SORT and ShOULD are not"
            " compatible with float16"
        )
    t0, t1 = 0.3, 1.0
    saveat = SaveAt(ts=jnp.linspace(t0, t1, 10, dtype=dtype))
    u = jnp.astype(1.0, dtype)
    gam = jnp.astype(1.0, dtype)
    vec_u = jnp.ones((dim,), dtype=dtype)
    vec_gam = jnp.ones((dim,), dtype=dtype)
    x0 = jnp.zeros((dim,), dtype=dtype)
    v0 = jnp.zeros((dim,), dtype=dtype)
    y0 = (x0, v0)
    f = lambda x: 0.5 * x
    shp_dtype = jax.ShapeDtypeStruct((dim,), dtype)
    levy_area = _abstract_la_to_la(solver.minimal_levy_area)
    bm = VirtualBrownianTree(
        t0,
        t1,
        tol=2**-4,
        shape=shp_dtype,
        key=jr.key(4),
        levy_area=levy_area,
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
        assert sol.ys is not None
        for entry in sol.ys:
            assert entry.shape == (10, dim)
            assert jnp.dtype(entry) == dtype


sdes = (
    get_harmonic_oscillator,
    get_bqp,
)


@pytest.mark.parametrize("get_sde", sdes)
@pytest.mark.parametrize("solver,theoretical_order", _solvers())
def test_langevin_strong_order(get_sde, solver, theoretical_order):
    bmkey = jr.key(5678)
    num_samples = 1000
    bmkeys = jr.split(bmkey, num=num_samples)
    t0 = 0.1
    t1 = 5.3

    sde = get_sde(t0, t1, jnp.float64)

    ref_solver = solver
    level_coarse, level_fine = 3, 7

    # We specify the times to which we step in way that each level contains all the
    # steps of the previous level. This is so that we can compare the solutions at
    # all the times in saveat, and not just at the end time.
    def get_dt_and_controller(level):
        step_ts = jnp.linspace(t0, t1, 2**level + 1, endpoint=True)
        return None, diffrax.StepTo(ts=step_ts)

    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 2**level_coarse + 1, endpoint=True))

    hs, errors, order = simple_sde_order(
        bmkeys,
        sde,
        solver,
        ref_solver,
        (level_coarse, level_fine),
        get_dt_and_controller,
        saveat,
        bm_tol=2**-13,
    )
    # The upper bound needs to be 0.25, otherwise we fail.
    # This still preserves a 0.05 buffer between the intervals
    # corresponding to the different orders.
    assert (
        -0.2 < order - theoretical_order < 0.25
    ), f"order={order}, theoretical_order={theoretical_order}"
