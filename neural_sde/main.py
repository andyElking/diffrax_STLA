import argparse
import glob
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from warnings import simplefilter

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt  # pyright: ignore
import optax  # https://github.com/deepmind/optax
from mcmc.metrics import compute_energy

from neural_sde.sde_and_cde import NeuralCDE, NeuralSDE, SDESolveConfig
from neural_sde.training import loss, make_step
from neural_sde.utils import dataloader, get_toy_data


solvers = {
    "SPaRK": diffrax.SPaRK(),
    "Euler": diffrax.Euler(),
    "Heun": diffrax.Heun(),
    "ShARK": diffrax.ShARK(),
    "GeneralShARK": diffrax.GeneralShARK(),
}


@dataclass
class NeuralSDEConfig:
    initial_noise_size: int
    noise_size: int
    hidden_size: int
    width_size: int
    depth: int
    generator_lr: float
    discriminator_lr: float
    batch_size: int
    steps: int
    steps_per_print: int
    dataset_size: int
    seed: int
    use_pid: bool
    dt0: float
    pid_atol: float
    pid_pcoeff: float
    pid_icoeff: float
    train_solver: str

    def __init__(
        self,
        data_size=1,
        initial_noise_size=5,
        noise_size=4,
        hidden_size=8,
        width_size=8,
        depth=1,
        generator_lr=2e-5,
        discriminator_lr=1e-4,
        batch_size=1024,
        steps=4000,
        steps_per_print=100,
        dataset_size=8192,
        seed=5678,
        use_pid=True,
        dt0=0.1,
        pid_atol=1e-2,
        pid_pcoeff=0.2,
        pid_icoeff=0.6,
        train_solver="ShARK",
    ):
        self.data_size = data_size
        self.initial_noise_size = initial_noise_size
        self.noise_size = noise_size
        self.hidden_size = hidden_size
        self.width_size = width_size
        self.depth = depth
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.batch_size = batch_size
        self.steps = steps
        self.steps_per_print = steps_per_print
        self.dataset_size = dataset_size
        self.seed = seed
        self.use_pid = use_pid
        self.dt0 = dt0
        self.pid_atol = pid_atol
        self.pid_pcoeff = pid_pcoeff
        self.pid_icoeff = pid_icoeff
        self.train_solver = train_solver

    def to_json(self):
        return {
            "data_size": self.data_size,
            "initial_noise_size": self.initial_noise_size,
            "noise_size": self.noise_size,
            "hidden_size": self.hidden_size,
            "width_size": self.width_size,
            "depth": self.depth,
            "generator_lr": self.generator_lr,
            "discriminator_lr": self.discriminator_lr,
            "batch_size": self.batch_size,
            "steps": self.steps,
            "steps_per_print": self.steps_per_print,
            "dataset_size": self.dataset_size,
            "seed": self.seed,
            "use_pid": self.use_pid,
            "dt0": self.dt0,
            "pid_atol": self.pid_atol,
            "pid_pcoeff": self.pid_pcoeff,
            "pid_icoeff": self.pid_icoeff,
            "train_solver": self.train_solver,
        }


def make_model(cfg: NeuralSDEConfig):
    key = jr.PRNGKey(cfg.seed)
    generator_key, discriminator_key = jr.split(key)
    generator = NeuralSDE(
        cfg.data_size,
        cfg.initial_noise_size,
        cfg.noise_size,
        cfg.hidden_size,
        cfg.width_size,
        cfg.depth,
        use_matrix_control=True,
        key=generator_key,
    )
    discriminator = NeuralCDE(
        cfg.data_size, cfg.hidden_size, cfg.width_size, cfg.depth, key=discriminator_key
    )
    return generator, discriminator


def save_model(generator, discriminator, cfg, path):
    json_filename = f"{path}.json"
    model_filename = f"{path}.eqx"
    hyperparam_str = json.dumps(cfg.to_json())
    with open(json_filename, "w") as f:
        f.write(hyperparam_str)
    with open(model_filename, "wb") as f:
        eqx.tree_serialise_leaves(f, (generator, discriminator))


def load_model(path):
    json_filename = f"{path}.json"
    model_filename = f"{path}.eqx"
    with open(json_filename, "r") as f:
        cfg = NeuralSDEConfig(**json.load(f))
    with open(model_filename, "rb") as f:
        g_d_skeleton = make_model(cfg)
        generator, discriminator = eqx.tree_deserialise_leaves(f, g_d_skeleton)
    return generator, discriminator, cfg


def main(cfg: NeuralSDEConfig, timestamp=None, logging=True):
    key = jr.PRNGKey(cfg.seed)
    (
        data_key,
        generator_key,
        discriminator_key,
        dataloader_key,
        train_key,
        evaluate_key,
        sample_key,
    ) = jr.split(key, 7)
    data_key = jr.split(data_key, cfg.dataset_size)

    ts, ys = get_toy_data(data_key, None, True)
    assert isinstance(ys, jnp.ndarray)
    assert ys.shape[-1] == cfg.data_size

    generator = NeuralSDE(
        cfg.data_size,
        cfg.initial_noise_size,
        cfg.noise_size,
        cfg.hidden_size,
        cfg.width_size,
        cfg.depth,
        use_matrix_control=True,
        key=generator_key,
    )
    if cfg.use_pid:
        controller = diffrax.PIDController(
            atol=cfg.pid_atol,
            rtol=0,
            pcoeff=cfg.pid_pcoeff,
            icoeff=cfg.pid_icoeff,
            dtmin=cfg.dt0 / 10,
        )
    else:
        controller = diffrax.ConstantStepSize()
    train_solver = solvers[cfg.train_solver]
    training_sde_solve_config = SDESolveConfig(
        solver=train_solver,
        step_controller=controller,
        dt0=cfg.dt0,
    )
    eval_sde_solve_config = SDESolveConfig(
        solver=diffrax.GeneralShARK(),
        step_controller=diffrax.ConstantStepSize(),
        dt0=0.005,
    )

    discriminator = NeuralCDE(
        cfg.data_size, cfg.hidden_size, cfg.width_size, cfg.depth, key=discriminator_key
    )

    g_optim = optax.rmsprop(cfg.generator_lr)
    d_optim = optax.rmsprop(-cfg.discriminator_lr)
    g_opt_state = g_optim.init(eqx.filter(generator, eqx.is_inexact_array))
    d_opt_state = d_optim.init(eqx.filter(discriminator, eqx.is_inexact_array))

    infinite_dataloader = dataloader(
        (ts, ys), cfg.batch_size, loop=True, key=dataloader_key
    )
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if logging:
        # Start log
        with open(f"neural_sde/model_saves/{timestamp}.txt", "w+") as f:
            f.write(f"Training run at {timestamp}\n")
            f.write(f"Config: {cfg.to_json()}\n\n")

    def write_log(s):
        print(s)
        if logging:
            with open(f"neural_sde/model_saves/{timestamp}.txt", "a") as f:
                f.write(s + "\n")

    def evaluate(step, avg_sde_solver_steps, elapsed_time=None):
        total_score = 0
        num_batches = 0
        for ts_i, ys_i in dataloader(
            (ts, ys), cfg.batch_size, loop=False, key=evaluate_key
        ):
            score, _ = loss(
                generator,
                discriminator,
                ts_i,
                ys_i,
                sample_key,
                0,
                eval_sde_solve_config,
            )
            total_score += score.item()
            num_batches += 1

        energy_err = evaluate_energy(generator, sample_key)
        str = (
            f"Step: {step}, Loss: {total_score / num_batches :.3f},"
            f" energy error: {energy_err:.3f}, "
            f"SDE solver steps: {avg_sde_solver_steps:.1f}"
        )
        if elapsed_time is not None:
            str += f", Elapsed time: {elapsed_time:.2f} seconds"
        write_log(str)

    total_sde_steps = 0
    sde_steps_per_print = 0
    start_time = time.time()
    for step, (ts_i, ys_i) in zip(range(cfg.steps), infinite_dataloader):
        step = jnp.asarray(step)
        generator, discriminator, g_opt_state, d_opt_state, sde_steps = make_step(
            generator,
            discriminator,
            g_opt_state,
            d_opt_state,
            g_optim,
            d_optim,
            ts_i,
            ys_i,
            key,
            step,
            training_sde_solve_config,
        )
        sde_steps_per_print += sde_steps
        if ((step + 1) % cfg.steps_per_print) == 0 or step == cfg.steps - 1:
            evaluate(
                step,
                sde_steps_per_print / cfg.steps_per_print,
                time.time() - start_time,
            )
            total_sde_steps += sde_steps_per_print
            sde_steps_per_print = 0

    final_str = (
        f"Training finished. Total SDE solver steps: {total_sde_steps},"
        f" total time: {time.time() - start_time:.2f} seconds"
    )
    write_log(final_str)

    # Save the model
    save_model(generator, discriminator, cfg, f"neural_sde/model_saves/{timestamp}")


def plot_samples(generator, dataset_size, sample_key):
    keys = jr.split(sample_key, dataset_size)
    ts, ys = get_toy_data(keys, None, False)
    assert isinstance(ys, jnp.ndarray)
    # Plot samples
    fig, ax = plt.subplots()
    num_samples = min(50, dataset_size)
    ts_to_plot = ts[:num_samples]
    ys_to_plot = ys[:num_samples]

    def _interp(ti, yi):
        return diffrax.linear_interpolation(
            ti, yi, replace_nans_at_start=0.0, fill_forward_nans_at_end=True
        )

    ys_to_plot = jax.vmap(_interp)(ts_to_plot, ys_to_plot)[..., 0]
    sde_solver_config = SDESolveConfig(
        solver=diffrax.GeneralShARK(),
        step_controller=diffrax.ConstantStepSize(),
        dt0=0.005,
    )
    ys_sampled, _ = jax.vmap(generator, in_axes=(0, 0, None))(
        ts_to_plot, jr.split(sample_key, num_samples), sde_solver_config
    )
    ys_sampled = ys_sampled[..., 0]
    kwargs = dict(label="Real")
    for ti, yi in zip(ts_to_plot, ys_to_plot):
        ax.plot(ti, yi, c="dodgerblue", linewidth=0.5, alpha=0.7, **kwargs)
        kwargs = {}
    kwargs = dict(label="Generated")
    for ti, yi in zip(ts_to_plot, ys_sampled):
        ax.plot(ti, yi, c="crimson", linewidth=0.5, alpha=0.7, **kwargs)
        kwargs = {}
    ax.set_title(f"{num_samples} samples from both real and generated distributions.")
    fig.legend()
    fig.tight_layout()
    fig.savefig("neural_sde.png")
    plt.show()


def evaluate_energy(generator, key):
    num_samples = 2**12
    num_ts = 4
    ts = jnp.linspace(0.0, 37.0, num_ts)
    keys = jr.split(key, num_samples)
    ts, ys = get_toy_data(keys, ts, False)
    assert ts.shape == (num_samples, num_ts)
    assert isinstance(ys, jnp.ndarray)
    assert ys.shape == (num_samples, num_ts, 1)

    sde_solver_config = SDESolveConfig(
        solver=diffrax.GeneralShARK(),
        step_controller=diffrax.ConstantStepSize(),
        dt0=0.005,
    )
    ys_sampled, _ = jax.vmap(generator, in_axes=(0, 0, None))(
        ts, jr.split(key, num_samples), sde_solver_config
    )
    ys_sampled = ys_sampled
    assert ys_sampled.shape == ys.shape, f"{ys_sampled.shape} != {ys.shape}"

    energy_err = compute_energy(ys_sampled, ys)
    return energy_err


def parse_args():
    parser = argparse.ArgumentParser(description="Neural SDE Training")
    parser.add_argument(
        "--generator_lr",
        type=float,
        default=2e-5,
        help="Learning rate for the generator",
    )
    parser.add_argument(
        "--discriminator_lr",
        type=float,
        default=1e-4,
        help="Learning rate for the discriminator",
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument(
        "--steps", type=int, default=10000, help="Number of training steps"
    )
    parser.add_argument(
        "--steps_per_print", type=int, default=200, help="Steps per print"
    )
    parser.add_argument(
        "--disable_pid", action="store_true", help="Disable PID controller"
    )
    parser.add_argument("--dt0", type=float, default=0.1, help="Initial time step")
    parser.add_argument(
        "--pid_atol", type=float, default=1e-3, help="PID absolute tolerance"
    )
    return parser.parse_args()


if __name__ == "__main__":
    simplefilter("ignore", category=FutureWarning)
    args = parse_args()

    cfg = NeuralSDEConfig(
        generator_lr=args.generator_lr,
        discriminator_lr=args.discriminator_lr,
        batch_size=args.batch_size,
        steps=args.steps,
        steps_per_print=args.steps_per_print,
        use_pid=not args.disable_pid,
        dt0=args.dt0,
        pid_atol=args.pid_atol,
    )
    main(cfg)

    filenames = glob.glob("neural_sde/model_saves/*.eqx")
    filenames.sort(key=os.path.getmtime)
    latest_path = filenames[-1][:-4]
    generator, discriminator, cfg = load_model(latest_path)
    plot_samples(generator, cfg.dataset_size, jr.PRNGKey(cfg.seed))
    energy_err = evaluate_energy(generator, jr.PRNGKey(cfg.seed))
    print(f"Energy error: {energy_err}")
