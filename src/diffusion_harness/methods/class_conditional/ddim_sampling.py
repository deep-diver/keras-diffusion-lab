"""Classifier-Free Guidance DDIM sampler.

Combines DDIM fast sampling with CFG guided prediction.
"""

import numpy as np
import keras.ops as ops
import jax

from diffusion_harness.base.ddim_sampling import DDIMSampler


class CFGDDIMSampler(DDIMSampler):
    """DDIM sampler with classifier-free guidance.

    At each DDIM reverse step, computes both conditional and unconditional
    predictions and interpolates: eps = (1+w)*cond - w*uncond.
    """

    def __init__(self, model, schedule, num_timesteps,
                 guidance_scale=3.0, num_classes=10,
                 eta=0.0, subsequence_size=50):
        super().__init__(model, schedule, num_timesteps,
                         eta=eta, subsequence_size=subsequence_size)
        self.guidance_scale = guidance_scale
        self.null_class_id = num_classes

    def model_predict(self, x_tensor, t_tensor, class_ids=None, **kwargs):
        """Guided prediction: interpolate conditional and unconditional."""
        if class_ids is None:
            raise ValueError("CFGDDIMSampler requires class_ids for sampling")

        # Conditional prediction
        eps_cond = self.model([x_tensor, t_tensor, class_ids], training=False)

        # Unconditional prediction (null class)
        null_ids = ops.convert_to_tensor(
            np.full((ops.shape(x_tensor)[0],), self.null_class_id, dtype="int32")
        )
        eps_uncond = self.model([x_tensor, t_tensor, null_ids], training=False)

        # Guided interpolation
        w = self.guidance_scale
        return (1.0 + w) * eps_cond - w * eps_uncond

    def sample(self, shape, class_ids, seed=0, initial_noise=None):
        """Generate class-conditional images via DDIM.

        Args:
            shape: (batch, h, w, c) output shape.
            class_ids: Class label(s). Scalar or array of length batch.
            seed: Random seed.
            initial_noise: Optional pre-generated initial noise.

        Returns:
            Numpy array of generated images, shape (batch, h, w, c),
            range [-1, 1].
        """
        # Broadcast scalar class_id to batch
        if np.ndim(class_ids) == 0:
            class_ids = np.full((shape[0],), int(class_ids), dtype="int32")
        else:
            class_ids = np.asarray(class_ids, dtype="int32")

        rng = jax.random.PRNGKey(seed)

        if initial_noise is not None:
            x = np.array(initial_noise, dtype="float32")
        else:
            rng, subkey = jax.random.split(rng)
            x = np.array(jax.random.normal(subkey, shape).astype("float32"))

        alphas_cumprod = self.schedule["alphas_cumprod"]
        c_tensor = ops.convert_to_tensor(class_ids)

        for i in reversed(range(len(self.timesteps))):
            t_val = int(self.timesteps[i])
            t_batch = np.full((shape[0],), t_val, dtype="int32")

            x_tensor = ops.convert_to_tensor(x)
            t_tensor = ops.convert_to_tensor(t_batch)

            eps_pred = np.array(
                self.model_predict(x_tensor, t_tensor, class_ids=c_tensor)
            )

            alpha_t = alphas_cumprod[t_val]

            # x_0 prediction
            x_0_hat = (
                x - np.sqrt(1.0 - alpha_t).reshape(-1, 1, 1, 1) * eps_pred
            ) / np.sqrt(alpha_t).reshape(-1, 1, 1, 1)
            x_0_hat = np.clip(x_0_hat, -1.0, 1.0)

            # Previous timestep in subsequence
            if i > 0:
                t_prev = int(self.timesteps[i - 1])
                alpha_prev = alphas_cumprod[t_prev]
            else:
                alpha_prev = 1.0

            # Compute sigma for stochastic DDIM
            sigma_sq = (
                self.eta ** 2
                * (1.0 - alpha_prev) / (1.0 - alpha_t)
                * (1.0 - alpha_t / alpha_prev)
            )
            sigma = np.sqrt(max(sigma_sq, 0.0))

            dir_xt = (
                np.sqrt(max(1.0 - alpha_prev - sigma_sq, 0.0))
                .reshape(-1, 1, 1, 1)
                * eps_pred
            )

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


def cfg_ddim_sample(model, schedule, num_timesteps, shape, class_ids,
                    guidance_scale=3.0, num_classes=10,
                    eta=0.0, subsequence_size=50, seed=0,
                    initial_noise=None):
    """Convenience function for class-conditional DDIM sampling.

    Args:
        model: Trained conditional denoiser.
        schedule: Dict from compute_schedule().
        num_timesteps: Total diffusion timesteps T.
        shape: (batch, h, w, c) output shape.
        class_ids: Class label(s).
        guidance_scale: Guidance strength (w).
        num_classes: Total number of classes.
        eta: Stochasticity (0 = deterministic).
        subsequence_size: Number of reverse steps.
        seed: Random seed.
        initial_noise: Optional pre-generated initial noise.

    Returns:
        Numpy array of generated images, shape (batch, h, w, c), [-1, 1].
    """
    sampler = CFGDDIMSampler(
        model, schedule, num_timesteps,
        guidance_scale=guidance_scale, num_classes=num_classes,
        eta=eta, subsequence_size=subsequence_size
    )
    return sampler.sample(shape, class_ids, seed=seed,
                          initial_noise=initial_noise)
