import math
from typing import Union

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from equinox import AbstractVar


def lipswish(x):
    return 0.909 * jnn.silu(x)


class VectorField(eqx.Module):
    scale: Union[int, jnp.ndarray]
    mlp: eqx.nn.MLP

    def __init__(self, hidden_size, width_size, depth, scale, *, key, **kwargs):
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if scale:
            self.scale = jr.uniform(scale_key, (hidden_size,), minval=0.9, maxval=1.1)
        else:
            self.scale = 1
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )

    def __call__(self, t, y, args):
        t = jnp.asarray(t)
        return self.scale * self.mlp(jnp.concatenate([t[None], y]))


class AbstractControlVF(eqx.Module):
    scale: AbstractVar[jnp.ndarray]
    mlp: AbstractVar[Union[jnp.ndarray, eqx.nn.MLP]]
    control_size: AbstractVar[int]
    hidden_size: AbstractVar[int]

    def __call__(self, t, y, args):
        raise NotImplementedError


class MLPControlledVF(AbstractControlVF):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP
    control_size: int
    hidden_size: int

    def __init__(
        self, control_size, hidden_size, width_size, depth, scale, *, key, **kwargs
    ):
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if scale:
            self.scale = jr.uniform(
                scale_key, (hidden_size, control_size), minval=0.9, maxval=1.1
            )
        else:
            dtype = jnp.result_type(0.1)
            self.scale = jnp.ones((), dtype=dtype)

        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size * control_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )
        self.control_size = control_size
        self.hidden_size = hidden_size

    def __call__(self, t, y, args):
        t = jnp.asarray(t)
        return self.scale * self.mlp(jnp.concatenate([t[None], y])).reshape(
            self.hidden_size, self.control_size
        )


class MatrixControlledVF(AbstractControlVF):
    scale: jnp.ndarray
    mlp: jnp.ndarray
    control_size: int
    hidden_size: int

    def __init__(self, control_size, hidden_size, *, key, **kwargs):
        super().__init__(**kwargs)
        dtype = jnp.result_type(0.1)
        self.scale = jnp.ones((), dtype=dtype)
        # just use a learnable matrix instead of MLP
        sz = 1 / math.sqrt(control_size)
        self.mlp = jr.uniform(key, (hidden_size, control_size), minval=-sz, maxval=sz)
        self.control_size = control_size
        self.hidden_size = hidden_size

    def __call__(self, t, y, args):
        return self.mlp
