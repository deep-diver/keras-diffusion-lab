"""Noise schedules for diffusion models.

Implements standard beta schedules from the DDPM literature:
- Linear schedule (Ho et al. 2020)
- Cosine schedule (Nichol & Dhariwal 2021)

Each schedule function returns a numpy array of beta values.
`compute_schedule` precomputes all derived quantities needed for
forward/reverse diffusion steps.
"""

import numpy as np


def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> np.ndarray:
    """Linear noise schedule from Ho et al. 2020."""
    return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> np.ndarray:
    """Cosine noise schedule from Nichol & Dhariwal 2021.

    Avoids destroying information at high timesteps by using a
    smoother schedule based on cosine interpolation.
    """
    steps = np.arange(num_timesteps + 1, dtype=np.float64)
    alpha_bar = np.cos((steps / num_timesteps + s) / (1 + s) * np.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return np.clip(betas, 0.0, 0.999)


def compute_schedule(betas: np.ndarray) -> dict:
    """Precompute all schedule quantities from a beta array.

    Returns a dict with keys used by forward/reverse diffusion:
      betas, alphas, alphas_cumprod, sqrt_alphas_cumprod,
      sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas_cumprod,
      sqrt_recipm1_alphas_cumprod, posterior_variance,
      posterior_log_variance_clipped, posterior_mean_coef1,
      posterior_mean_coef2
    """
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)

    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus = np.sqrt(1.0 - alphas_cumprod)
    sqrt_recip = np.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1 = np.sqrt(1.0 / alphas_cumprod - 1.0)

    # Posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = (
        betas * (1.0 - np.concatenate([[1.0], alphas_cumprod[:-1]]))
        / (1.0 - alphas_cumprod)
    )
    # Log variance clipped for numerical stability
    posterior_log_variance_clipped = np.log(
        np.maximum(posterior_variance, 1e-20)
    )

    # Posterior mean coefficients: mu = coef1 * x0_hat + coef2 * x_t
    posterior_mean_coef1 = (
        betas * np.sqrt(np.concatenate([[1.0], alphas_cumprod[:-1]]))
        / (1.0 - alphas_cumprod)
    )
    posterior_mean_coef2 = (
        (1.0 - np.concatenate([[1.0], alphas_cumprod[:-1]]))
        * np.sqrt(alphas)
        / (1.0 - alphas_cumprod)
    )

    return {
        "betas": betas.astype(np.float32),
        "alphas": alphas.astype(np.float32),
        "alphas_cumprod": alphas_cumprod.astype(np.float32),
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod.astype(np.float32),
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus.astype(np.float32),
        "sqrt_recip_alphas_cumprod": sqrt_recip.astype(np.float32),
        "sqrt_recipm1_alphas_cumprod": sqrt_recipm1.astype(np.float32),
        "posterior_variance": posterior_variance.astype(np.float32),
        "posterior_log_variance_clipped": posterior_log_variance_clipped.astype(np.float32),
        "posterior_mean_coef1": posterior_mean_coef1.astype(np.float32),
        "posterior_mean_coef2": posterior_mean_coef2.astype(np.float32),
    }
