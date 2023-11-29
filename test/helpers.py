import dataclasses
import functools as ft
import operator
from typing import Callable, Tuple

import diffrax
import equinox as eqx
import jax
import jax.random as jrandom
import jax.tree_util as jtu
from diffrax import (
    AbstractAdditiveSRK,
    AbstractBrownianPath,
    AbstractTerm,
    ALIGN,
    ConstantStepSize,
    diffeqsolve,
    LangevinTerm,
    SaveAt,
    UnsafeBrownianPath,
    VirtualBrownianTree,
)
from diffrax.custom_types import PyTree, Scalar
from jax import numpy as jnp


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
            lambda s: (s.nonlinear_solver.rtol, s.nonlinear_solver.atol),
            solver,
            (1e-3, 1e-6),
            is_leaf=lambda x: x is None,
        )
    return solver


def random_pytree(key, treedef, dtype=None):
    keys = jrandom.split(key, treedef.num_leaves)
    leaves = []
    for key in keys:
        dimkey, sizekey, valuekey = jrandom.split(key, 3)
        num_dims = jrandom.randint(dimkey, (), 0, 5)
        dim_sizes = jrandom.randint(sizekey, (num_dims,), 0, 5)
        value = jrandom.normal(valuekey, dim_sizes, dtype=dtype)
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


def _shaped_allclose(x, y, **kwargs):
    return jnp.shape(x) == jnp.shape(y) and jnp.allclose(x, y, **kwargs)


def shaped_allclose(x, y, **kwargs):
    """As `jnp.allclose`, except:
    - It also supports PyTree arguments.
    - It mandates that shapes match as well (no broadcasting)
    """
    same_structure = jtu.tree_structure(x) == jtu.tree_structure(y)
    allclose = ft.partial(_shaped_allclose, **kwargs)
    return same_structure and jtu.tree_reduce(
        operator.and_, jtu.tree_map(allclose, x, y), True
    )


def path_l2_dist(ys1: PyTree[jax.Array], ys2: PyTree[jax.Array]):
    # first compute the square of the difference and sum over
    # all but the first two axes (which represent the number of samples
    # and the length of saveat). Also sum all the PyTree leaves
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


@dataclasses.dataclass
class SDE:
    get_terms: Callable[[AbstractBrownianPath], AbstractTerm]
    args: PyTree
    y0: PyTree
    t0: Scalar
    t1: Scalar
    w_shape: Tuple[int]

    def get_dtype(self):
        return jnp.dtype(jtu.tree_leaves(self.y0)[0])

    def get_bm(self, key, need_stla=True, use_tree=True, tol=2**-14):
        shp_dtype = jax.ShapeDtypeStruct(self.w_shape, dtype=self.get_dtype())
        if use_tree:
            return VirtualBrownianTree(
                t0=self.t0,
                t1=self.t1,
                shape=shp_dtype,
                tol=tol,
                key=key,
                spacetime_levyarea=need_stla,
            )
        else:
            return UnsafeBrownianPath(
                shape=shp_dtype, key=key, spacetime_levyarea=need_stla
            )


def batch_sde_solve(
    keys, sde: SDE, dt0, solver, stepsize_controller=ConstantStepSize(), need_stla=False
):
    _saveat = SaveAt(ts=[sde.t1])

    need_stla = need_stla or isinstance(solver, (ALIGN, AbstractAdditiveSRK))

    def end_value(key):
        path = sde.get_bm(key, need_stla=need_stla, use_tree=True)
        terms = sde.get_terms(path)

        sol = diffeqsolve(
            terms,
            solver,
            sde.t0,
            sde.t1,
            dt0=dt0,
            y0=sde.y0,
            args=sde.args,
            saveat=_saveat,
            stepsize_controller=stepsize_controller,
            max_steps=None,
        )
        return sol.ys

    return jax.vmap(end_value)(keys)


def sde_solver_order(keys, sde: SDE, solver, ref_solver, dt_precise, hs_num=5, hs=None):
    dtype = sde.get_dtype()
    need_stla = isinstance(solver, (ALIGN, AbstractAdditiveSRK)) or isinstance(
        ref_solver, (ALIGN, AbstractAdditiveSRK)
    )

    correct_sols = batch_sde_solve(
        keys, sde, dt_precise, ref_solver, need_stla=need_stla
    )
    if hs is None:
        hs = jnp.power(2.0, jnp.arange(-3, -3 - hs_num, -1, dtype=dtype))

    def get_single_err(h):
        sols = batch_sde_solve(keys, sde, h, solver, need_stla=need_stla)
        return path_l2_dist(sols, correct_sols)

    errs = jax.vmap(get_single_err)(hs)
    order, _ = jnp.polyfit(jnp.log(hs), jnp.log(errs), 1)
    return hs, errs, order


def get_bqp(t0=0.3, t1=15.0, dtype=jnp.float32):
    grad_f_bqp = lambda x: 4 * x * (jnp.square(x) - 1)
    args_bqp = (dtype(0.8), dtype(0.2), grad_f_bqp)
    y0_bqp = (dtype(0), dtype(0))
    w_shape_bqp = ()

    def get_terms_bqp(bm):
        return LangevinTerm(args_bqp, bm)

    return SDE(get_terms_bqp, None, y0_bqp, t0, t1, w_shape_bqp)


def get_harmonic_oscillator(t0=0.3, t1=15.0, dtype=jnp.float32):
    gamma_hosc = jnp.array([2, 0.5], dtype=dtype)
    u_hosc = jnp.array([0.5, 2], dtype=dtype)
    args_hosc = (gamma_hosc, u_hosc, lambda x: 2 * x)
    x0 = jnp.zeros((2,), dtype=dtype)
    v0 = jnp.zeros((2,), dtype=dtype)
    y0_hosc = (x0, v0)
    w_shape_hosc = (2,)

    def get_terms_hosc(bm):
        return LangevinTerm(args_hosc, bm)

    return SDE(get_terms_hosc, None, y0_hosc, t0, t1, w_shape_hosc)
