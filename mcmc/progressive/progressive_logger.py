import jax.numpy as jnp

from ..logging import AbstractLogger


class ProgressiveLogger(AbstractLogger):
    def make_log_string(self, method_name: str, method_dict: dict) -> str:
        best_acc = jnp.max(method_dict["test_acc"])
        if "test_acc_best80" in method_dict:
            best_acc80 = jnp.max(method_dict["test_acc_best80"])
            acc_top_str = f" acc top 80%: {best_acc80:.4},"
        elif "test_acc_best90" in method_dict:
            best_acc90 = jnp.max(method_dict["test_acc_best90"])
            acc_top_str = f" acc top 90%: {best_acc90:.4},"
        else:
            acc_top_str = ""

        best_energy = jnp.min(method_dict["energy_err"])
        best_w2 = jnp.min(method_dict["w2"])
        str_out = (
            f"{method_name}: acc: {best_acc:.4},{acc_top_str}"
            f" energy: {best_energy:.3e}, w2: {best_w2:.3e}"
        )
        return str_out
