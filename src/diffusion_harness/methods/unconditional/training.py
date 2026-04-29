"""Unconditional DDPM trainer.

Implements epsilon-prediction MSE loss with JAX value_and_grad,
extending the shared BaseTrainer infrastructure.
"""

import numpy as np
import jax
import keras.ops as ops

from diffusion_harness.base.training import BaseTrainer
from diffusion_harness.base.models import build_unet


class UnconditionalTrainer(BaseTrainer):
    """Standard unconditional DDPM trainer.

    Implements epsilon-prediction training: sample noise, apply forward
    diffusion, predict noise with U-Net, minimize MSE.
    """

    def __init__(self, config):
        # Build model
        model = build_unet(
            image_size=config["image_size"],
            channels=config["image_channels"],
            base_filters=config["base_filters"],
            num_levels=config["num_levels"],
            channel_multipliers=config["channel_multipliers"],
            attention_resolutions=config["attention_resolutions"],
        )

        # Build with dummy input
        dummy_x = np.zeros((1, config["image_size"], config["image_size"],
                            config["image_channels"]), dtype="float32")
        dummy_t = np.zeros((1,), dtype="int32")
        model([dummy_x, dummy_t])

        super().__init__(config, model)

    def train_step(self, batch):
        """Execute one unconditional DDPM training step.

        Args:
            batch: Image batch array of shape (B, H, W, C).

        Returns:
            Dict with loss, grad_norm, nan_detected.
        """
        config = self.config
        schedule = config["schedule"]
        num_timesteps = config["num_timesteps"]
        batch_size = batch.shape[0]

        # Sample random timesteps and noise
        key = jax.random.PRNGKey(self.step)
        _, t_key, noise_key = jax.random.split(key, 3)
        t = np.array(jax.random.randint(t_key, (batch_size,), 0, num_timesteps))
        noise = np.array(jax.random.normal(noise_key, batch.shape).astype("float32"))

        # Forward diffusion: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        alpha_bar_t = schedule["alphas_cumprod"][t]
        sqrt_alpha = np.sqrt(alpha_bar_t).reshape(-1, 1, 1, 1)
        sqrt_one_minus = np.sqrt(1.0 - alpha_bar_t).reshape(-1, 1, 1, 1)
        x_t = (sqrt_alpha * batch + sqrt_one_minus * noise).astype("float32")

        x_tensor = ops.convert_to_tensor(x_t)
        t_tensor = ops.convert_to_tensor(t.astype("int32"))
        noise_tensor = ops.convert_to_tensor(noise)

        # Pure loss function for gradient computation
        def _loss_fn(var_values):
            original_values = []
            for i, v in enumerate(self.model.trainable_variables):
                original_values.append(np.array(v.value))
                v.assign(var_values[i])

            eps_pred = self.model([x_tensor, t_tensor], training=True)
            loss = ops.mean((eps_pred - noise_tensor) ** 2)

            for i, v in enumerate(self.model.trainable_variables):
                v.assign(original_values[i])

            return loss

        var_values = [v.value for v in self.model.trainable_variables]
        loss_val, grads = jax.value_and_grad(_loss_fn)(var_values)

        grad_norm = float(np.sqrt(sum(float(np.sum(g ** 2)) for g in grads)))
        nan_detected = any(np.any(np.isnan(g)) for g in grads)

        if not nan_detected:
            self.optimizer.apply(grads, self.model.trainable_variables)

            if self.ema_weights is not None:
                for i, v in enumerate(self.model.trainable_variables):
                    self.ema_weights[i] = (
                        self.ema_decay * self.ema_weights[i]
                        + (1.0 - self.ema_decay) * np.array(v.value)
                    )

        self.step += 1
        loss_float = float(loss_val)
        self.loss_history.append(loss_float)

        return {
            "loss": loss_float,
            "grad_norm": grad_norm,
            "nan_detected": nan_detected,
        }
