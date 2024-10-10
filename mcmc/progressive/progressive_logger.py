import jax.numpy as jnp

from ..logging import AbstractLogger


class ProgressiveLogger(AbstractLogger):
    def make_log_string(self, method_name: str, method_dict: dict) -> str:
        best_acc = jnp.max(method_dict["test_acc"])
        best_acc90 = jnp.max(method_dict["test_acc_best80"])
        best_energy = jnp.min(method_dict["energy_err"])
        best_w2 = jnp.min(method_dict["w2"])
        str_out = (
            f"{method_name}: acc: {best_acc:.4}, acc top 80%: {best_acc90:.4},"
            f" energy: {best_energy:.3e}, w2: {best_w2:.3e}"
        )
        return str_out
