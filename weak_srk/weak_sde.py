from test.helpers import SDE

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from diffrax import ControlTerm, MultiTerm, ODETerm


def drift(t, y, args):
    mx = np.array([[-273 / 512, 0], [-1 / 160, -785 / 512 + np.sqrt(2) / 8]])
    return jnp.matmul(mx, y)


def diffusion(t, y, args):
    assert jnp.shape(y)[-1] == 2
    y1 = y[..., 0]
    y2 = y[..., 1]
    return jnp.array(
        [
            [1 / 4 * y1, 1 / 16 * y1],
            [(1 - jnp.sqrt(8)) / 4 * y2, 1 / 10 * y1 + 1 / 16 * y2],
        ]
    )


def get_weak_sde(t0, t1, dtype):
    y0 = jnp.ones((2,), dtype=dtype)

    def get_terms(bm):
        return MultiTerm(ODETerm(drift), ControlTerm(diffusion, bm))

    return SDE(get_terms, None, y0, t0, t1, (2,))


@eqx.filter_jit
def weak_error(ys, t0, t1):
    empirical = jnp.mean(ys[..., 0] ** 2)
    true = jnp.exp(t0 - t1)
    return jnp.abs(empirical - true)
