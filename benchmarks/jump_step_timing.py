from warnings import simplefilter


simplefilter(action="ignore", category=FutureWarning)

import timeit
from functools import partial

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from old_pid_controller import OldPIDController


t0 = 0
t1 = 5
dt0 = 0.5
y0 = 1.0
drift = diffrax.ODETerm(lambda t, y, args: -0.2 * y)


def diffusion_vf(t, y, args):
    return jnp.ones((), dtype=y.dtype)


def get_terms(key):
    bm = diffrax.VirtualBrownianTree(t0, t1, 2**-5, (), key)
    diffusion = diffrax.ControlTerm(diffusion_vf, bm)
    return diffrax.MultiTerm(drift, diffusion)


solver = diffrax.Heun()
step_ts = jnp.linspace(t0, t1, 129, endpoint=True)
pid_controller = diffrax.PIDController(
    rtol=0, atol=1e-3, dtmin=2**-9, dtmax=1.0, pcoeff=0.3, icoeff=0.7
)
new_controller = diffrax.JumpStepWrapper(
    pid_controller,
    step_ts=step_ts,
    rejected_step_buffer_len=0,
)
old_controller = OldPIDController(
    rtol=0, atol=1e-3, dtmin=2**-9, dtmax=1.0, pcoeff=0.3, icoeff=0.7, step_ts=step_ts
)


@eqx.filter_jit
@partial(jax.vmap, in_axes=(0, None))
def solve(key, controller):
    term = get_terms(key)
    return diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        stepsize_controller=controller,
        saveat=diffrax.SaveAt(ts=step_ts),
    )


num_samples = 100
keys = jr.split(jr.PRNGKey(0), num_samples)


# NEW CONTROLLER
@jax.jit
@eqx.debug.assert_max_traces(max_traces=1)
def time_new_controller_fun():
    sols = solve(keys, new_controller)
    assert sols.ys is not None
    assert sols.ys.shape == (num_samples, len(step_ts))
    return sols.ys


def time_new_controller():
    jax.block_until_ready(time_new_controller_fun())


# OLD CONTROLLER
@jax.jit
@eqx.debug.assert_max_traces(max_traces=1)
def time_old_controller_fun():
    sols = solve(keys, old_controller)
    assert sols.ys is not None
    assert sols.ys.shape == (num_samples, len(step_ts))
    return sols.ys


def time_old_controller():
    jax.block_until_ready(time_old_controller_fun())


time_new = min(timeit.repeat(time_new_controller, number=3, repeat=20))

time_old = min(timeit.repeat(time_old_controller, number=3, repeat=20))

print(f"New controller: {time_new:.5} s, Old controller: {time_old:.5} s")

# How expensive is revisiting rejected steps?
new_revisiting_controller = diffrax.JumpStepWrapper(
    pid_controller,
    step_ts=step_ts,
    rejected_step_buffer_len=10,
)


def time_revisiting_controller_fun():
    sols = solve(keys, new_revisiting_controller)
    assert sols.ys is not None
    assert sols.ys.shape == (num_samples, len(step_ts))
    return sols.ys


def time_revisiting_controller():
    jax.block_until_ready(time_revisiting_controller_fun())


time_revisiting = min(timeit.repeat(time_revisiting_controller, number=3, repeat=20))

print(f"Revisiting controller: {time_revisiting:.5} s")

# ======= RESULTS =======
# New controller: 0.29384 s, Old controller: 0.30669 s
# Revisiting controller: 0.38819 s
