import jax.random as jr
from main import load_model, plot_samples


save_path = "model_saves/2024-10-22_16-28-45"
generator, discriminator, cfg = load_model(save_path)
plot_samples(generator, cfg.dataset_size, jr.key(0), save_path)
# energy_err = evaluate_energy(generator, jr.key(0))
# print(f"Energy error: {energy_err}")
