"""Base sampler for diffusion models and image saving utilities.

Provides the reverse diffusion loop as a reusable class with an
overridable model_predict() hook. Methods like CFG override this
to implement guided prediction.
"""

import numpy as np
import keras
import keras.ops as ops
import jax


class BaseSampler:
    """Standard DDPM reverse sampler.

    Subclasses may override model_predict() to modify how the model
    is called (e.g., classifier-free guidance).
    """

    def __init__(self, model, schedule, num_timesteps):
        self.model = model
        self.schedule = schedule
        self.num_timesteps = num_timesteps

    def model_predict(self, x_tensor, t_tensor, **kwargs):
        """Run model prediction. Override for guided sampling."""
        return self.model([x_tensor, t_tensor], training=False)

    def sample(self, shape, seed=0, initial_noise=None):
        """Generate images using the full DDPM reverse process.

        Args:
            shape: (batch, h, w, c) output shape.
            seed: Random seed.
            initial_noise: Optional pre-generated initial noise.

        Returns:
            Numpy array of generated images, shape (batch, h, w, c), range [-1, 1].
        """
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
            eps_pred = self.model_predict(x_tensor, t_tensor)
            eps_pred = np.array(eps_pred)

            rng, subkey = jax.random.split(rng)
            noise = np.array(jax.random.normal(subkey, shape).astype("float32"))
            if t_val == 0:
                noise = np.zeros_like(noise)

            x = _p_sample_step(x, eps_pred, t_batch, self.schedule, noise)

        return np.array(x)


def _p_sample_step(x_t, eps_pred, t_batch, schedule, noise):
    """Single reverse diffusion step."""
    sqrt_recip = schedule["sqrt_recip_alphas_cumprod"][t_batch]
    sqrt_recipm1 = schedule["sqrt_recipm1_alphas_cumprod"][t_batch]
    coef1 = schedule["posterior_mean_coef1"][t_batch]
    coef2 = schedule["posterior_mean_coef2"][t_batch]
    log_var = schedule["posterior_log_variance_clipped"][t_batch]

    x_0_hat = sqrt_recip.reshape(-1, 1, 1, 1) * x_t - sqrt_recipm1.reshape(-1, 1, 1, 1) * eps_pred
    x_0_hat = np.clip(x_0_hat, -1.0, 1.0)
    mean = coef2.reshape(-1, 1, 1, 1) * x_t + coef1.reshape(-1, 1, 1, 1) * x_0_hat

    nonzero_mask = (t_batch != 0).reshape(-1, 1, 1, 1).astype("float32")
    x_prev = mean + nonzero_mask * np.exp(0.5 * log_var.reshape(-1, 1, 1, 1)) * noise
    return x_prev


def save_image_grid(images, path, nrow=4):
    """Save a grid of images to a file.

    Args:
        images: Numpy array (N, H, W, C) in [-1, 1].
        path: Output file path.
        nrow: Number of images per row.
    """
    from diffusion_harness.data import denormalize

    images_uint8 = denormalize(images)
    n = len(images_uint8)
    h, w = images_uint8.shape[1], images_uint8.shape[2]
    c = images_uint8.shape[3] if images_uint8.ndim == 4 else 1
    ncol = (n + nrow - 1) // nrow

    grid = np.zeros((nrow * h, ncol * w, c), dtype=np.uint8)
    for idx in range(n):
        row = idx // ncol
        col = idx % ncol
        if row < nrow and col < ncol:
            grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = images_uint8[idx]

    if c == 1:
        grid = grid.squeeze(-1)

    try:
        from PIL import Image
        Image.fromarray(grid).save(path)
    except ImportError:
        np.save(path.replace(".png", ".npy"), grid)


def save_annotated_grid(images, path, step=None, loss=None, nrow=4):
    """Save image grid with step/loss annotation."""
    from diffusion_harness.data import denormalize

    images_uint8 = denormalize(images)
    n = len(images_uint8)
    h, w, c = images_uint8.shape[1:]
    ncol = (n + nrow - 1) // nrow

    grid = np.zeros((nrow * h, ncol * w, c), dtype=np.uint8)
    for idx in range(n):
        row = idx // ncol
        col = idx % ncol
        if row < nrow and col < ncol:
            grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = images_uint8[idx]

    try:
        from PIL import Image, ImageDraw
        img = Image.fromarray(grid)
        draw = ImageDraw.Draw(img)
        label_parts = []
        if step is not None:
            label_parts.append(f"step {step}")
        if loss is not None:
            label_parts.append(f"loss {loss:.4f}")
        if label_parts:
            label = " | ".join(label_parts)
            bbox = draw.textbbox((0, 0), label)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([4, 2, text_w + 8, text_h + 4], fill=(0, 0, 0))
            draw.text((6, 2), label, fill=(255, 255, 255))
        img.save(path)
    except ImportError:
        save_image_grid(images, path, nrow)
