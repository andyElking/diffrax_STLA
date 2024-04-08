from test.helpers import simple_sde_order

import diffrax
import jax.numpy as jnp
import numpy as np
from diffrax import PIDController
from matplotlib import animation, pyplot as plt  # type: ignore


def draw_order_multiple(results_list, names_list, title=None):
    plt.figure(dpi=200)
    if title is not None:
        plt.title(title)

    orders = "Orders:\n"
    for results, name in zip(results_list, names_list):
        steps, errs, order = results
        plt.scatter(steps, errs, label=f"{name}: {order:.2f}")
        trend = np.polyfit(-np.log(steps), np.log(errs), 1)
        trend_f = np.poly1d(trend)
        plt.plot(steps, np.exp(trend_f(-np.log(steps))))
        orders += f"{name}: {order:.2f}\n"
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("RMS error")
    plt.xlabel("average number of steps")
    plt.legend()
    plt.show()


def plot_sol_general(sol):
    plt.plot(sol.ts, sol.ys)
    plt.show()


def plot_sol_langevin(sol):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(sol.ts, sol.ys[0], label="x")
    ax.plot(sol.ts, sol.ys[1], label="v")
    # ax.set_ylim([-3.0, 3.0])
    ax.legend()
    plt.show()


def plot_sol3D(sol):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    x1 = sol.ys[0][:, :3]  # this is an Nx3 array which we want to plot
    x2 = sol.ys[0][:, 3:6]
    x3 = sol.ys[0][:, 6:]
    ax.plot(*x1.T, label="particle 1")
    ax.plot(*x2.T, label="particle 2")
    ax.plot(*x3.T, label="particle 3")
    ax.legend()
    plt.show()


FUNNEL_LIMS = [-8, 8]


def animated_funnel_plot(sol, skip=8):
    xs = sol[0][0]
    num_times = xs.shape[1]
    fig, (ax1, ax2) = plt.subplots(2)
    scat = ax1.scatter(xs[:, 0, 0], xs[:, 0, 1])
    ax1.set(xlim=FUNNEL_LIMS, ylim=FUNNEL_LIMS)

    def update(frame):
        idx = min(frame * skip, num_times - 1)
        x = xs[:, idx, 0]
        y = xs[:, idx, 1]
        scat.set_offsets(jnp.stack([x, y]).T)
        return (scat,)

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=int(num_times / skip) + 2, interval=4
    )

    ax2.scatter(xs[:, -1, 0], xs[:, -1, 1])
    ax2.set(xlim=FUNNEL_LIMS, ylim=FUNNEL_LIMS)
    plt.show()
    return ani, fig, ax1


def draw_funnel(samples):
    samples_rshp = jnp.reshape(samples, (-1, 10))
    fig, ax = plt.subplots(1, 1)
    ax.scatter(samples_rshp[:, 0], samples_rshp[:, 1], alpha=0.2, s=8.0)
    ax.set(xlim=FUNNEL_LIMS, ylim=FUNNEL_LIMS)
    plt.show()


def constant_step_strong_order(keys, sde, solver, levels, bm_tol=None):
    def _step_ts(level):
        return jnp.linspace(sde.t0, sde.t1, 2**level + 1, endpoint=True)

    def get_dt_and_controller(level):
        return None, diffrax.StepTo(ts=_step_ts(level))

    _saveat = diffrax.SaveAt(ts=_step_ts(levels[0]))
    if bm_tol is None:
        bm_tol = (sde.t1 - sde.t0) * (2 ** -(levels[1] + 3))
    return simple_sde_order(
        keys, sde, solver, solver, levels, get_dt_and_controller, _saveat, bm_tol
    )  # returns steps, errs, order


def pid_strong_order(keys, sde, solver, levels, bm_tol=2**-14):
    save_ts_pid = jnp.linspace(sde.t0, sde.t1, 65, endpoint=True)

    def get_pid(level):
        return None, PIDController(
            pcoeff=0.1,
            icoeff=0.3,
            rtol=0,
            atol=2**-level,
            step_ts=save_ts_pid,
            dtmin=2**-14,
        )

    saveat_pid = diffrax.SaveAt(ts=save_ts_pid)
    return simple_sde_order(
        keys, sde, solver, solver, levels, get_pid, saveat_pid, bm_tol
    )
