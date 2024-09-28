import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro  # pyright: ignore
from numpyro import distributions as dist  # pyright: ignore


def get_model_and_data(data, name):
    dset = data[name][0, 0]
    x = dset["x"]
    labels = jnp.squeeze(dset["t"])
    # labels are -1 and 1, convert to 0 and 1
    labels = (labels + 1) / 2
    n, data_dim = x.shape
    print(f"Data shape: {x.shape}")

    # randomly shuffle the data
    perm = jax.random.permutation(jr.PRNGKey(0), n)
    x = x[perm]
    labels = labels[perm]

    n_train = min(int(n * 0.8), 1000)
    x_train = x[:n_train]
    labels_train = labels[:n_train]
    x_test = x[n_train:]
    labels_test = labels[n_train:]

    def model(x, labels):
        x_var = jnp.var(x, axis=0)
        W = numpyro.sample(
            "W",
            dist.Normal(jnp.zeros(data_dim), 0.5 / x_var),
        )
        b = numpyro.sample("b", dist.Normal(jnp.zeros((1,)), 1))
        logits = jnp.sum(W * x, axis=-1) + b
        return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    return model, (x_train, labels_train, x_test, labels_test)
