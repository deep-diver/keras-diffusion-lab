"""Unconditional DDPM sampling.

Convenience wrapper around BaseSampler that matches the original
ddpm_sample() function signature.
"""

import numpy as np
from diffusion_harness.base.sampling import BaseSampler


def unconditional_sample(model, schedule, num_timesteps, shape, seed=0, initial_noise=None):
    """Generate images using the full DDPM reverse process.

    Args:
        model: Trained denoiser model.
        schedule: Dict from compute_schedule().
        num_timesteps: Total diffusion timesteps T.
        shape: (batch, h, w, c) output shape.
        seed: Random seed.
        initial_noise: Optional pre-generated initial noise.

    Returns:
        Numpy array of generated images, shape (batch, h, w, c), range [-1, 1].
    """
    sampler = BaseSampler(model, schedule, num_timesteps)
    return sampler.sample(shape, seed=seed, initial_noise=initial_noise)
