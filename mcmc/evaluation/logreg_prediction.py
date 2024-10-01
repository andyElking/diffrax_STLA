import jax
from jax import numpy as jnp

from ..utils import vec_dict_to_array


def predict(x, samples):
    b = samples[:, 0]
    w = samples[:, 1:]
    logits = jnp.sum(w * x, axis=-1) + b
    # apply sigmoid
    return 1.0 / (1.0 + jnp.exp(-logits))


def test_accuracy(x_test, labels_test, samples):
    if isinstance(samples, dict):
        samples = vec_dict_to_array(samples)
    assert x_test.shape[1] + 1 == samples.shape[-1], (
        f"The last dim of {x_test.shape} should be the"
        f" last dim of {samples.shape} minus 1"
    )
    sample_dim = samples.shape[-1]
    samples = jnp.reshape(samples, (-1, sample_dim))
    if samples.shape[0] > 2**10:
        samples = samples[: 2**10]

    func = jax.jit(jax.vmap(lambda x: predict(x, samples), in_axes=0, out_axes=0))
    predictions = func(x_test)
    assert predictions.shape == (
        labels_test.shape[0],
        samples.shape[0],
    ), f"{predictions.shape} != {(labels_test.shape[0], samples.shape[0])}"

    labels_test = jnp.reshape(labels_test, (labels_test.shape[0], 1))
    is_correct = jnp.abs(predictions - labels_test) < 0.5
    accuracy_per_sample = jnp.mean(is_correct, axis=0)

    avg_accuracy = jnp.mean(accuracy_per_sample)

    len10 = int(0.1 * accuracy_per_sample.shape[0])
    best_sorted = jnp.sort(accuracy_per_sample)[len10:]
    accuracy_best90 = jnp.mean(best_sorted)
    return avg_accuracy, accuracy_best90
