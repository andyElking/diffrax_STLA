import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import jax.random as jr


@eqx.filter_jit
def loss(generator, discriminator, ts_i, ys_i, key, step, sde_solve_config):
    batch_size, _ = ts_i.shape
    key = jr.fold_in(key, step)
    key = jr.split(key, batch_size)
    fake_ys_i = jax.vmap(generator)(ts_i, key=key, sde_solve_config=sde_solve_config)
    real_score = jax.vmap(discriminator)(ts_i, ys_i)
    fake_score = jax.vmap(discriminator)(ts_i, fake_ys_i)
    return jnp.mean(real_score - fake_score)


@eqx.filter_grad
def grad_loss(g_d, ts_i, ys_i, key, step, sde_solve_config):
    generator, discriminator = g_d
    return loss(generator, discriminator, ts_i, ys_i, key, step, sde_solve_config)


def increase_update_initial(updates):
    get_initial_leaves = lambda u: jax.tree_util.tree_leaves(u.initial)
    return eqx.tree_at(get_initial_leaves, updates, replace_fn=lambda x: x * 10)


@eqx.filter_jit
def make_step(
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
    sde_solve_config,
):
    g_grad, d_grad = grad_loss(
        (generator, discriminator), ts_i, ys_i, key, step, sde_solve_config
    )
    g_updates, g_opt_state = g_optim.update(g_grad, g_opt_state)
    d_updates, d_opt_state = d_optim.update(d_grad, d_opt_state)
    g_updates = increase_update_initial(g_updates)
    d_updates = increase_update_initial(d_updates)
    generator = eqx.apply_updates(generator, g_updates)
    discriminator = eqx.apply_updates(discriminator, d_updates)
    discriminator = discriminator.clip_weights()
    return generator, discriminator, g_opt_state, d_opt_state
