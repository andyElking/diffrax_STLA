from typing import Optional

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import jax.random as jr

from .vector_fields import (
    AbstractControlVF,
    MatrixControlledVF,
    MLPControlledVF,
    VectorField,
)


class SDESolveConfig:
    solver: diffrax.AbstractSolver
    step_controller: diffrax.AbstractStepSizeController
    dt0: float

    def __init__(
        self,
        solver: Optional[diffrax.AbstractSolver] = None,
        step_controller: Optional[diffrax.AbstractStepSizeController] = None,
        dt0: Optional[float] = None,
    ):
        if solver is None:
            solver = diffrax.ReversibleHeun()
        if step_controller is None:
            step_controller = diffrax.ConstantStepSize()
        if dt0 is None:
            dt0 = 1.0
        self.solver = solver
        self.step_controller = step_controller
        self.dt0 = dt0

    def to_tuple(self):
        return self.solver, self.step_controller, self.dt0


class NeuralSDE(eqx.Module):
    initial: eqx.nn.MLP
    vf: VectorField  # drift
    cvf: AbstractControlVF  # diffusion
    readout: eqx.nn.Linear
    initial_noise_size: int
    noise_size: int

    def __init__(
        self,
        data_size,
        initial_noise_size,
        noise_size,
        hidden_size,
        width_size,
        depth,
        use_matrix_control: bool = False,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        initial_key, vf_key, cvf_key, readout_key = jr.split(key, 4)

        self.initial = eqx.nn.MLP(
            initial_noise_size, hidden_size, width_size, depth, key=initial_key
        )
        self.vf = VectorField(hidden_size, width_size, depth, scale=True, key=vf_key)
        if use_matrix_control:
            self.cvf = MatrixControlledVF(noise_size, hidden_size, key=cvf_key)
        else:
            self.cvf = MLPControlledVF(
                noise_size, hidden_size, width_size, depth, scale=True, key=cvf_key
            )
        self.readout = eqx.nn.Linear(hidden_size, data_size, key=readout_key)

        self.initial_noise_size = initial_noise_size
        self.noise_size = noise_size

    def __call__(self, ts, key, sde_solve_config: SDESolveConfig):
        t0 = ts[0]
        t1 = ts[-1]
        init_key, bm_key = jr.split(key, 2)
        init = jr.normal(init_key, (self.initial_noise_size,))
        solver, step_controller, dt0 = sde_solve_config.to_tuple()
        tol = (
            dt0 / 2
            if isinstance(step_controller, diffrax.ConstantStepSize)
            else dt0 / 10
        )
        control = diffrax.VirtualBrownianTree(
            t0=t0,
            t1=t1,
            tol=tol,
            shape=(self.noise_size,),
            key=bm_key,
            levy_area=diffrax.SpaceTimeLevyArea,
        )
        vf = diffrax.ODETerm(self.vf)  # Drift term
        cvf = diffrax.ControlTerm(self.cvf, control)  # Diffusion term
        terms = diffrax.MultiTerm(vf, cvf)
        y0 = self.initial(init)
        saveat = diffrax.SaveAt(ts=ts)
        sol = diffrax.diffeqsolve(
            terms, solver, t0, t1, dt0, y0, saveat=saveat, max_steps=10000
        )
        assert sol.ys is not None
        return jax.vmap(self.readout)(sol.ys), sol.stats["num_steps"]


class NeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    vf: VectorField
    cvf: MLPControlledVF
    readout: eqx.nn.Linear

    def __init__(self, data_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        initial_key, vf_key, cvf_key, readout_key = jr.split(key, 4)

        self.initial = eqx.nn.MLP(
            data_size + 1, hidden_size, width_size, depth, key=initial_key
        )
        self.vf = VectorField(hidden_size, width_size, depth, scale=False, key=vf_key)
        self.cvf = MLPControlledVF(
            data_size, hidden_size, width_size, depth, scale=False, key=cvf_key
        )
        self.readout = eqx.nn.Linear(hidden_size, 1, key=readout_key)

    def __call__(self, ts, ys):
        # Interpolate data into a continuous path.
        ys = diffrax.linear_interpolation(
            ts, ys, replace_nans_at_start=0.0, fill_forward_nans_at_end=True
        )
        init = jnp.concatenate([ts[0, None], ys[0]])
        control = diffrax.LinearInterpolation(ts, ys)
        vf = diffrax.ODETerm(self.vf)
        cvf = diffrax.ControlTerm(self.cvf, control)
        terms = diffrax.MultiTerm(vf, cvf)
        solver = diffrax.ReversibleHeun()
        t0 = ts[0]
        t1 = ts[-1]
        y0 = self.initial(init)
        # Have the discriminator produce an output at both `t0` *and* `t1`.
        # The output at `t0` has only seen the initial point of a sample. This gives
        # additional supervision to the distribution learnt for the initial condition.
        # The output at `t1` has seen the entire path of a sample. This is needed to
        # actually learn the evolving trajectory.
        saveat = diffrax.SaveAt(t0=True, t1=True)
        step_controller = diffrax.StepTo(ts=ts)
        sol = diffrax.diffeqsolve(
            terms,
            solver,
            t0,
            t1,
            None,
            y0,
            saveat=saveat,
            stepsize_controller=step_controller,
        )
        assert sol.ys is not None
        return jax.vmap(self.readout)(sol.ys)

    @eqx.filter_jit
    def clip_weights(self):
        leaves, treedef = jax.tree_util.tree_flatten(
            self, is_leaf=lambda x: isinstance(x, eqx.nn.Linear)
        )
        new_leaves = []
        for leaf in leaves:
            if isinstance(leaf, eqx.nn.Linear):
                lim = 1 / leaf.out_features  # pyright: ignore
                leaf = eqx.tree_at(
                    lambda x: x.weight, leaf, leaf.weight.clip(-lim, lim)
                )
            new_leaves.append(leaf)
        return jax.tree_util.tree_unflatten(treedef, new_leaves)
