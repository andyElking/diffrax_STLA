import datetime
import sys
import warnings

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import scipy
from numpyro.infer import MCMC, NUTS, Predictive  # noqa: F401

from mcmc.experiment_main import run_experiment
from mcmc.lmc import run_simple_lmc_numpyro  # noqa: F401
from mcmc.logreg_utils import eval_gt_logreg, get_gt_logreg, get_model_and_data
from mcmc.metrics import adjust_max_len
from mcmc.progressive import (
    ProgressiveEvaluator,
    ProgressiveLMC,
    ProgressiveLogger,
    ProgressiveNUTS,
)
from mcmc.progressive.progressive_plotting import make_figs


warnings.simplefilter("ignore", FutureWarning)

jnp.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)
jax.config.update("jax_enable_x64", True)
print(jax.devices("cuda"))

dataset = scipy.io.loadmat("mcmc_data/benchmarks.mat")
names = [
    # "tbp",
    # "isolet_ab",
    # "banana",
    # "breast_cancer",
    # "diabetis",
    # "flare_solar",
    # "german",
    # "heart",
    # "image",
    # "ringnorm",
    "splice",
    # "thyroid",
    # "titanic",
    # "twonorm",
    # "waveform",
]


timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
prev_result_quic = lambda name: f"progressive_results/{name}_2024-10-12_21-03-56.pkl"
prev_result_nuts = lambda name: f"progressive_results/{name}_*.pkl"

evaluator = ProgressiveEvaluator()
logger = ProgressiveLogger(log_filename=f"progressive_results/log_{timestamp}.txt")
logger.start_log(timestamp)

nuts_warmup = 30
nuts_len = 2**7
nuts = ProgressiveNUTS(nuts_warmup, nuts_len)

USE_PID = True
pid_str = "pid_" if USE_PID else ""
get_result_filename = (
    lambda name: f"progressive_results/{name}_{pid_str}{timestamp}.pkl"
)


def make_pid(atol, dt0):
    if not USE_PID:
        return None
    return diffrax.PIDController(
        atol=atol,
        rtol=0.0,
        dtmax=0.5,
        dtmin=dt0 / 10,
        pcoeff=0.1,
        icoeff=0.4,
    )


quic_kwargs = {
    "chain_len": 2**5,
    "chain_sep": 1.0,
    "dt0": 0.07,
    "solver": diffrax.QUICSORT(0.1),
    "pid": make_pid(0.1, 0.07),
}
quic = ProgressiveLMC(quic_kwargs, prev_result_quic)
euler_kwargs = {
    "chain_len": 2**5,
    "chain_sep": 0.5,
    "dt0": 0.03,
    "solver": diffrax.Euler(),
    "pid": make_pid(0.1, 0.03),
}
euler = ProgressiveLMC(euler_kwargs)
methods = [nuts, quic]

dt0s = {
    "banana": 0.04,
    "splice": 0.01,
    "flare_solar": 0.1,
    "isolet_ab": 0.001,
}
seps = {
    "banana": 0.3,
    "splice": 1.0,
    "flare_solar": 2.0,
    "image": 1.0,
    "waveform": 1.0,
    "isolet_ab": 0.5,
}
atols = {}


for name in names:
    model, model_args, test_args = get_model_and_data(dataset, name)
    data_dim = model_args[0].shape[1] + 1
    num_particles = adjust_max_len(2**14, data_dim)
    config = {
        "num_particles": num_particles,
        "test_args": test_args,
    }
    quic_dt0 = dt0s.get(name, 0.07)
    chain_sep = seps.get(name, 0.5)
    atol = atols.get(name, 1.0)
    quic.lmc_kwargs["dt0"], quic.lmc_kwargs["chain_sep"] = quic_dt0, chain_sep
    quic.lmc_kwargs["pid"] = make_pid(atol, quic_dt0)
    euler.lmc_kwargs["dt0"], euler.lmc_kwargs["chain_sep"] = (
        quic_dt0 / 100,
        chain_sep / 5,
    )
    euler.lmc_kwargs["pid"] = make_pid(atol, quic_dt0 / 20)

    logger.start_model_section(name)
    quic_atol_str = f"atol={atol}, " if USE_PID else ""
    logger.print_log(
        f"NUTS(warmup={nuts.num_warmup}, total={nuts.chain_len}),"
        f" QUICSORT({quic_atol_str}dt0={quic_dt0}, sep={chain_sep})\n"
    )

    run_experiment(
        jr.key(0),
        model,
        model_args,
        name,
        methods,
        config,
        evaluator,
        logger,
        get_gt_logreg,
        eval_gt_logreg,
        get_result_filename,
    )

    result_filename = get_result_filename(name)
    figs = make_figs(
        result_filename, save_name=f"progressive_results/plots/{name}_{timestamp}.pdf"
    )
