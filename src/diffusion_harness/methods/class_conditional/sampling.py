"""Classifier-Free Guidance sampler.

Extends BaseSampler with guided prediction: runs both conditional and
unconditional model passes, then interpolates for guided generation.
"""

import numpy as np
import keras.ops as ops

from diffusion_harness.base.sampling import BaseSampler


class CFGSampler(BaseSampler):
    """Classifier-Free Guidance sampler.

    At each reverse step, computes both conditional and unconditional
    noise predictions and interpolates: eps = (1+w)*cond - w*uncond.
    """

    def __init__(self, model, schedule, num_timesteps,
                 guidance_scale=3.0, num_classes=10):
        super().__init__(model, schedule, num_timesteps)
        self.guidance_scale = guidance_scale
        self.null_class_id = num_classes  # Null class is one past last real class

    def model_predict(self, x_tensor, t_tensor, class_ids=None, **kwargs):
        """Guided prediction: interpolate conditional and unconditional."""
        if class_ids is None:
            raise ValueError("CFGSampler requires class_ids for sampling")

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
        """Generate class-conditional images.

        Args:
            shape: (batch, h, w, c) output shape.
            class_ids: Class label(s). Scalar or array of length batch.
                       If scalar, all samples use the same class.
            seed: Random seed.
            initial_noise: Optional pre-generated initial noise.

        Returns:
            Numpy array of generated images, shape (batch, h, w, c), range [-1, 1].
        """
        import jax

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

        for t_val in reversed(range(self.num_timesteps)):
            t_batch = np.full((shape[0],), t_val, dtype="int32")

            x_tensor = ops.convert_to_tensor(x)
            t_tensor = ops.convert_to_tensor(t_batch)
            c_tensor = ops.convert_to_tensor(class_ids)

            eps_pred = self.model_predict(x_tensor, t_tensor, class_ids=c_tensor)
            eps_pred = np.array(eps_pred)

            rng, subkey = jax.random.split(rng)
            noise = np.array(jax.random.normal(subkey, shape).astype("float32"))
            if t_val == 0:
                noise = np.zeros_like(noise)

            from diffusion_harness.base.sampling import _p_sample_step
            x = _p_sample_step(x, eps_pred, t_batch, self.schedule, noise)

        return np.array(x)


def cfg_sample(model, schedule, num_timesteps, shape, class_ids,
               guidance_scale=3.0, num_classes=10, seed=0,
               initial_noise=None):
    """Convenience function for class-conditional sampling.

    Args:
        model: Trained conditional denoiser.
        schedule: Dict from compute_schedule().
        num_timesteps: Total diffusion timesteps T.
        shape: (batch, h, w, c) output shape.
        class_ids: Class label(s).
        guidance_scale: Guidance strength (w). Higher = stronger conditioning.
        num_classes: Total number of classes.
        seed: Random seed.
        initial_noise: Optional pre-generated initial noise.

    Returns:
        Numpy array of generated images, shape (batch, h, w, c), range [-1, 1].
    """
    sampler = CFGSampler(model, schedule, num_timesteps,
                         guidance_scale=guidance_scale, num_classes=num_classes)
    return sampler.sample(shape, class_ids, seed=seed, initial_noise=initial_noise)
