import dataclasses
from typing import Callable

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optimistix as optx
from diffrax import (
    AbstractBrownianPath,
    AbstractTerm,
    ControlTerm,
    diffeqsolve,
    MultiTerm,
    ODETerm,
    SaveAt,
    UnsafeBrownianPath,
    VirtualBrownianTree,
)
from jaxtyping import PyTree


all_ode_solvers = (
    diffrax.Bosh3(),
    diffrax.Dopri5(),
    diffrax.Dopri8(),
    diffrax.Euler(),
    diffrax.Ralston(),
    diffrax.Midpoint(),
    diffrax.Heun(),
    diffrax.LeapfrogMidpoint(),
    diffrax.ReversibleHeun(),
    diffrax.Tsit5(),
    diffrax.ImplicitEuler(),
    diffrax.Kvaerno3(),
    diffrax.Kvaerno4(),
    diffrax.Kvaerno5(),
)

all_split_solvers = (
    diffrax.Sil3(),
    diffrax.KenCarp3(),
    diffrax.KenCarp4(),
    diffrax.KenCarp5(),
)


def implicit_tol(solver):
    if isinstance(solver, diffrax.AbstractImplicitSolver):
        return eqx.tree_at(
            lambda s: (s.root_finder.rtol, s.root_finder.atol, s.root_finder.norm),
            solver,
            (1e-3, 1e-6, optx.rms_norm),
        )
    return solver


def random_pytree(key, treedef, dtype):
    keys = jr.split(key, treedef.num_leaves)
    leaves = []
    for key in keys:
        dimkey, sizekey, valuekey = jr.split(key, 3)
        num_dims = jr.randint(dimkey, (), 0, 5).item()
        dim_sizes = jr.randint(sizekey, (num_dims,), 0, 5)
        value = jr.normal(valuekey, tuple(dim_sizes.tolist()), dtype=dtype)
        leaves.append(value)
    return jtu.tree_unflatten(treedef, leaves)


treedefs = [
    jtu.tree_structure(x)
    for x in (
        0,
        None,
        {"a": [0, 0], "b": 0},
    )
]


def _no_nan(x):
    if eqx.is_array(x):
        return x.at[jnp.isnan(x)].set(8.9568)  # arbitrary magic value
    else:
        return x


def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8, equal_nan=False):
    if equal_nan:
        x = jtu.tree_map(_no_nan, x)
        y = jtu.tree_map(_no_nan, y)
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


def path_l2_dist(ys1: PyTree[jax.Array], ys2: PyTree[jax.Array]):
    # first compute the square of the difference and sum over
    # all but the first two axes (which represent the number of samples
    # and the length of saveat). Also sum all the PyTree leaves.
    def sum_square_diff(y1, y2):
        square_diff = jnp.square(y1 - y2)
        # sum all but the first two axes
        axes = range(2, square_diff.ndim)
        out = jnp.sum(square_diff, axis=axes)
        return out

    dist = jtu.tree_map(sum_square_diff, ys1, ys2)
    dist = sum(jtu.tree_leaves(dist))  # shape=(num_samples, len(saveat))
    dist = jnp.max(dist, axis=1)  # take sup along the length of integration
    dist = jnp.sqrt(jnp.mean(dist))
    return dist


# TODO: remove this once we have a better way to handle this
# I understand you'd prefer not to have this in the library and
# I agree this is somewhat hacky, but I think passing each of
# these args around separately is a bit of a pain. If this only appeared
# in the tests, I'd be fine with it, but it's also in the examples
# (e.g. srk_example.py) and I'd prefer if it looked a bit cleaner there.
# So how do you recommend we streamline this?
@dataclasses.dataclass
class SDE:
    get_terms: Callable[[AbstractBrownianPath], AbstractTerm]
    args: PyTree
    y0: PyTree
    t0: float
    t1: float
    w_shape: tuple

    def get_dtype(self):
        return jnp.result_type(*jtu.tree_leaves(self.y0))

    def get_bm(
        self,
        key,
        levy_area: type[diffrax.BrownianIncrement],
        use_tree=True,
        tol=2**-14,
    ):
        shp_dtype = jax.ShapeDtypeStruct(self.w_shape, dtype=self.get_dtype())
        if use_tree:
            return VirtualBrownianTree(
                t0=self.t0,
                t1=self.t1,
                shape=shp_dtype,
                tol=tol,
                key=key,
                levy_area=levy_area,
            )
        else:
            return UnsafeBrownianPath(shape=shp_dtype, key=key, levy_area=levy_area)


def _batch_sde_solve(
    keys,
    sde: SDE,
    dt0,
    solver,
    levy_area: type[diffrax.BrownianIncrement],
):
    _saveat = SaveAt(ts=[sde.t1])

    @jax.jit
    @jax.vmap
    def end_value(key):
        path = sde.get_bm(key, levy_area=levy_area, use_tree=True)
        terms = sde.get_terms(path)

        sol = diffeqsolve(
            terms,
            solver,
            sde.t0,
            sde.t1,
            dt0=dt0,
            y0=sde.y0,
            args=sde.args,
            max_steps=None,
        )
        return sol.ys

    return end_value(keys)


def sde_solver_strong_order(keys, sde: SDE, solver, ref_solver, dt_precise, dts):
    if hasattr(solver, "minimal_levy_area"):
        levy_area = solver.minimal_levy_area
    else:
        levy_area = diffrax.BrownianIncrement

    # Pick the common descendant of the minimal levy area of both solvers
    if hasattr(ref_solver, "minimal_levy_area") and issubclass(
        ref_solver.minimal_levy_area, levy_area
    ):
        levy_area = ref_solver.minimal_levy_area

    correct_sols = _batch_sde_solve(
        keys, sde, dt_precise, ref_solver, levy_area=levy_area
    )

    @jax.jit
    @jax.vmap
    def get_single_err(h):
        sols = _batch_sde_solve(keys, sde, h, solver, levy_area=levy_area)
        return path_l2_dist(sols, correct_sols)

    errs = get_single_err(dts)
    order, _ = jnp.polyfit(jnp.log(dts), jnp.log(errs), 1)
    return dts, errs, order


def _squareplus(x):
    return 0.5 * (x + jnp.sqrt(x**2 + 4))


def drift(t, y, args):
    mlp, _, _ = args
    return 0.25 * mlp(y)


def diffusion(t, y, args):
    _, mlp, noise_dim = args
    return 1.0 * mlp(y).reshape(3, noise_dim)


def get_mlp_sde(t0, t1, dtype, key, noise_dim):
    driftkey, diffusionkey, ykey = jr.split(key, 3)
    # To Patrick: I had to increase the depth of these MLPs, otherwise many SDE
    # solvers had order ~0.72 which is more than 0.5 + 0.2, which is the maximal
    # tolerated order. I think the issue was that it was too linear and too easy.
    drift_mlp = eqx.nn.MLP(
        in_size=3,
        out_size=3,
        width_size=8,
        depth=2,
        activation=_squareplus,
        final_activation=jnp.tanh,
        key=driftkey,
    )
    diffusion_mlp = eqx.nn.MLP(
        in_size=3,
        out_size=3 * noise_dim,
        width_size=8,
        depth=2,
        activation=_squareplus,
        final_activation=jnp.tanh,
        key=diffusionkey,
    )
    args = (drift_mlp, diffusion_mlp, noise_dim)
    y0 = jr.normal(ykey, (3,), dtype=dtype)

    def get_terms(bm):
        return MultiTerm(ODETerm(drift), ControlTerm(diffusion, bm))

    return SDE(get_terms, args, y0, t0, t1, (noise_dim,))


# This is needed for time_sde (i.e. the additive noise SDE) because initializing
# the weights in the drift MLP with a Gaussian makes the SDE too linear and nice,
# so we need to use a Laplace distribution, which is heavier-tailed.
def lap_init(weight: jax.Array, key) -> jax.Array:
    out, in_ = weight.shape
    stddev = jnp.sqrt(1 / in_)
    return stddev * jax.random.laplace(key, shape=(out, in_), dtype=weight.dtype)


def init_linear_weight(model, init_fn, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [
        x.weight
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]
    weights = get_weights(model)
    new_weights = [
        init_fn(weight, subkey)
        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model


def get_time_sde(t0, t1, dtype, key, noise_dim):
    y_dim = 7
    driftkey, diffusionkey, ykey = jr.split(key, 3)

    def ft(t):
        return jnp.array(
            [jnp.sin(t), jnp.cos(4 * t), 1.0, 1.0 / (t + 0.5)], dtype=dtype
        )

    drift_mlp = eqx.nn.MLP(
        in_size=y_dim + 4,
        out_size=y_dim,
        width_size=16,
        depth=4,
        activation=_squareplus,
        key=driftkey,
    )

    # The drift weights must be Laplace-distributed,
    # otherwise the SDE is too linear and nice.
    drift_mlp = init_linear_weight(drift_mlp, lap_init, driftkey)

    def _drift(t, y, _):
        return 0.25 * drift_mlp(jnp.concatenate([y, ft(t)]))

    diffusion_mx = jr.normal(diffusionkey, (4, y_dim, noise_dim), dtype=dtype)

    def _diffusion(t, _, __):
        # This needs a large coefficient to make the SDE not too easy.
        return 10000.0 * jnp.tensordot(ft(t), diffusion_mx, axes=1)

    args = (drift_mlp, None, None)
    y0 = jr.normal(ykey, (y_dim,), dtype=dtype)

    def get_terms(bm):
        return MultiTerm(ODETerm(_drift), ControlTerm(_diffusion, bm))

    return SDE(get_terms, args, y0, t0, t1, (noise_dim,))
