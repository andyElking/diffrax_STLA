import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr


@jax.jit
@jax.vmap
def get_toy_data(key):
    bm_key, y0_key, drop_key = jr.split(key, 3)

    mu = 0.02
    theta = 0.1
    sigma = 0.4

    t0 = 0
    t1 = 63
    t_size = 64

    def drift(t, y, args):
        return mu * t - theta * y

    def diffusion(t, y, args):
        return 2 * sigma * t / t1

    bm = diffrax.UnsafeBrownianPath(
        shape=(), key=bm_key, levy_area=diffrax.SpaceTimeLevyArea
    )
    drift_term = diffrax.ODETerm(drift)
    diffusion_term = diffrax.ControlTerm(diffusion, bm)
    terms = diffrax.MultiTerm(drift_term, diffusion_term)
    solver = diffrax.ShARK()
    dt0 = 0.01
    y0 = jr.uniform(y0_key, (1,), minval=-1, maxval=1)
    ts = jnp.linspace(t0, t1, t_size)
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat=saveat,
        adjoint=diffrax.DirectAdjoint(),
        max_steps=6400,
    )

    # Make the data irregularly sampled
    to_drop = jr.bernoulli(drop_key, 0.3, (t_size, 1))
    ys = jnp.where(to_drop, jnp.nan, sol.ys)

    return ts, ys


def dataloader(arrays, batch_size, loop, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        key = jr.split(key, 1)[0]
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
        if not loop:
            break
