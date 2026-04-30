"""DDIM sampler for trained DDPM models.

Implements the denoising diffusion implicit models reverse process from
Song et al. 2021. Works with any model trained using the standard DDPM
forward process — no retraining needed.

DDIM uses a subsequence of the full timestep range, enabling faster
sampling (e.g., 50 or 200 steps instead of 1000).

Key parameters:
  - eta: Controls stochasticity. eta=0 is fully deterministic DDIM,
         eta=1 recovers DDPM behavior.
  - subsequence_size: Number of timesteps in the subsequence.
"""

import numpy as np
import keras.ops as ops
import jax


class DDIMSampler:
    """DDIM reverse sampler compatible with BaseSampler-style models.

    Works with both unconditional models (2 inputs: image, timestep)
    and conditional models (3 inputs: image, timestep, class_id)
    when used with a model_predict override.
    """

    def __init__(self, model, schedule, num_timesteps,
                 eta=0.0, subsequence_size=50):
        self.model = model
        self.schedule = schedule
        self.num_timesteps = num_timesteps
        self.eta = eta
        self.subsequence_size = subsequence_size
        # Build timestep subsequence: evenly spaced from 0 to T-1
        self.timesteps = np.linspace(
            0, num_timesteps - 1, subsequence_size
        ).astype(np.int64)

    def model_predict(self, x_tensor, t_tensor, **kwargs):
        """Run model prediction. Override for guided sampling."""
        return self.model([x_tensor, t_tensor], training=False)

    def sample(self, shape, seed=0, initial_noise=None):
        """Generate images using DDIM reverse process.

        Args:
            shape: (batch, h, w, c) output shape.
            seed: Random seed for initial noise.
            initial_noise: Optional pre-generated initial noise.

        Returns:
            Numpy array of generated images, shape (batch, h, w, c),
            range [-1, 1].
        """
        rng = jax.random.PRNGKey(seed)

        if initial_noise is not None:
            x = np.array(initial_noise, dtype="float32")
        else:
            rng, subkey = jax.random.split(rng)
            x = np.array(jax.random.normal(subkey, shape).astype("float32"))

        alphas_cumprod = self.schedule["alphas_cumprod"]

        for i in reversed(range(len(self.timesteps))):
            t_val = int(self.timesteps[i])
            t_batch = np.full((shape[0],), t_val, dtype="int32")

            x_tensor = ops.convert_to_tensor(x)
            t_tensor = ops.convert_to_tensor(t_batch)
            eps_pred = np.array(self.model_predict(x_tensor, t_tensor))

            alpha_t = alphas_cumprod[t_val]

            # x_0 prediction
            x_0_hat = (
                x - np.sqrt(1.0 - alpha_t).reshape(-1, 1, 1, 1) * eps_pred
            ) / np.sqrt(alpha_t).reshape(-1, 1, 1, 1)
            x_0_hat = np.clip(x_0_hat, -1.0, 1.0)

            # Previous timestep in subsequence (alpha_prev = 1.0 at start)
            if i > 0:
                t_prev = int(self.timesteps[i - 1])
                alpha_prev = alphas_cumprod[t_prev]
            else:
                alpha_prev = 1.0

            # Compute sigma for eta > 0 (stochastic DDIM)
            sigma_sq = (
                self.eta ** 2
                * (1.0 - alpha_prev) / (1.0 - alpha_t)
                * (1.0 - alpha_t / alpha_prev)
            )
            sigma = np.sqrt(max(sigma_sq, 0.0))

            # Direction pointing to x_t
            dir_xt = (
                np.sqrt(max(1.0 - alpha_prev - sigma_sq, 0.0))
                .reshape(-1, 1, 1, 1)
                * eps_pred
            )

            # Noise (only if stochastic)
            if self.eta > 0 and i > 0:
                rng, subkey = jax.random.split(rng)
                noise = np.array(
                    jax.random.normal(subkey, shape).astype("float32")
                )
            else:
                noise = np.zeros_like(x)

            x = (
                np.sqrt(alpha_prev).reshape(-1, 1, 1, 1) * x_0_hat
                + dir_xt
                + sigma * noise
            )

        return np.array(x)


def ddim_sample(model, schedule, num_timesteps, shape,
                eta=0.0, subsequence_size=50, seed=0,
                initial_noise=None):
    """Convenience function for unconditional DDIM sampling.

    Args:
        model: Trained denoiser model.
        schedule: Dict from compute_schedule().
        num_timesteps: Total diffusion timesteps T.
        shape: (batch, h, w, c) output shape.
        eta: Stochasticity (0 = deterministic DDIM, 1 = DDPM-like).
        subsequence_size: Number of reverse steps.
        seed: Random seed.
        initial_noise: Optional pre-generated initial noise.

    Returns:
        Numpy array of generated images, shape (batch, h, w, c), [-1, 1].
    """
    sampler = DDIMSampler(model, schedule, num_timesteps,
                          eta=eta, subsequence_size=subsequence_size)
    return sampler.sample(shape, seed=seed, initial_noise=initial_noise)
