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
        # alpha = numpyro.sample("alpha", dist.Exponential(0.01))
        x_cov = jnp.cov(x, rowvar=False)
        W = numpyro.sample(
            "W",
            dist.MultivariateNormal(jnp.zeros(data_dim), precision_matrix=0.1 * x_cov),
        )
        b = numpyro.sample("b", dist.Normal(jnp.zeros((1,)), 100))
        logits = jnp.sum(W * x + b, axis=-1)
        return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    return model, (x_train, labels_train, x_test, labels_test)
