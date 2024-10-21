import json
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

from neural_sde.sde_and_cde import NeuralCDE, NeuralSDE, SDESolveConfig
from neural_sde.training import loss, make_step
from neural_sde.utils import dataloader, get_toy_data


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
    train_solver: diffrax.AbstractSolver

    def __init__(
        self,
        data_size=1,
        initial_noise_size=5,
        noise_size=8,
        hidden_size=16,
        width_size=16,
        depth=1,
        generator_lr=1e-4,
        discriminator_lr=3e-4,
        batch_size=1024,
        steps=200,
        steps_per_print=50,
        dataset_size=8192,
        seed=5678,
        use_pid=True,
        dt0=0.5,
        pid_atol=1e-2,
        pid_pcoeff=0.2,
        pid_icoeff=0.6,
        train_solver=diffrax.SPaRK(),
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
    hyperparam_str = json.dumps(cfg.to_json())
    with open(path, "wb") as f:
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, (generator, discriminator))


def load_model(path):
    with open(path, "rb") as f:
        hyperparam_str = f.readline().decode().strip()
        cfg = NeuralSDEConfig(**json.loads(hyperparam_str))
        g_d = make_model(cfg)
        generator, discriminator = eqx.tree_deserialise_leaves(f, g_d)
    return generator, discriminator, cfg


def main(cfg: NeuralSDEConfig):
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

    ts, ys = get_toy_data(data_key)
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
            step_ts=ts,
            dtmin=cfg.dt0 / 10,
        )
    else:
        controller = diffrax.ConstantStepSize()
    training_sde_solve_config = SDESolveConfig(
        solver=cfg.train_solver,
        step_controller=controller,
        dt0=cfg.dt0,
    )
    eval_sde_solve_config = SDESolveConfig(
        solver=diffrax.GeneralShARK(),
        step_controller=diffrax.ConstantStepSize(),
        dt0=0.01,
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

    avg_sde_steps_total = 0
    avg_sde_steps_per_print = 0
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
        avg_sde_steps_per_print += sde_steps
        if (step % cfg.steps_per_print) == 0 or step == cfg.steps - 1:
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
            print(
                f"Step: {step}, Loss: {total_score / num_batches}, "
                f"SDE solver steps: {avg_sde_steps_per_print / cfg.steps_per_print}"
            )
            avg_sde_steps_total += avg_sde_steps_per_print
            avg_sde_steps_per_print = 0

    avg_sde_steps_total /= cfg.steps
    # Save the model
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_model(generator, discriminator, cfg, f"model_saves/{timestamp}.eqx")


def plot_samples(generator, dataset_size, sample_key):
    ts, ys = get_toy_data(sample_key)
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
        dt0=0.01,
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


if __name__ == "__main__":
    simplefilter("ignore", category=FutureWarning)
    cfg = NeuralSDEConfig()
    main(cfg)
    generator, discriminator, cfg = load_model("model_saves/2021-10-15_12-11-18.eqx")
    plot_samples(generator, cfg.dataset_size, jr.PRNGKey(cfg.seed))
