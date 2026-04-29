"""Classifier-Free Guidance (CFG) trainer.

Extends BaseTrainer with class-conditional training. During training,
class labels are randomly dropped (replaced with null class) with
probability class_dropout_prob. This teaches the model to denoise both
conditioned and unconditioned, enabling guided sampling.
"""

import numpy as np
import jax
import keras.ops as ops

from diffusion_harness.base.training import BaseTrainer
from diffusion_harness.methods.class_conditional.models import build_cond_unet


class CFGTrainer(BaseTrainer):
    """Classifier-Free Guidance DDPM trainer.

    Adds class conditioning with random null-class dropout during training.
    """

    def __init__(self, config):
        num_classes = config.get("num_classes", 10)
        self.num_classes = num_classes
        self.null_class_id = num_classes  # Null class is one past the last real class
        self.class_dropout_prob = config.get("class_dropout_prob", 0.1)

        # Build conditional model
        model = build_cond_unet(
            image_size=config["image_size"],
            channels=config["image_channels"],
            base_filters=config["base_filters"],
            num_levels=config["num_levels"],
            channel_multipliers=config["channel_multipliers"],
            attention_resolutions=config["attention_resolutions"],
            num_classes=num_classes,
        )

        # Build with dummy input (3 inputs: image, timestep, class_id)
        dummy_x = np.zeros((1, config["image_size"], config["image_size"],
                            config["image_channels"]), dtype="float32")
        dummy_t = np.zeros((1,), dtype="int32")
        dummy_c = np.zeros((1,), dtype="int32")
        model([dummy_x, dummy_t, dummy_c])

        super().__init__(config, model)

    def train_step(self, batch):
        """Execute one CFG training step.

        Args:
            batch: Tuple (images, labels) where images is (B, H, W, C)
                   and labels is (B,) int array.

        Returns:
            Dict with loss, grad_norm, nan_detected.
        """
        images, labels = batch

        config = self.config
        schedule = config["schedule"]
        num_timesteps = config["num_timesteps"]
        batch_size = images.shape[0]

        # Class dropout: randomly replace some labels with null class
        class_ids = labels.copy()
        if self.class_dropout_prob > 0:
            drop_mask = np.random.random(batch_size) < self.class_dropout_prob
            class_ids[drop_mask] = self.null_class_id

        # Sample random timesteps and noise
        key = jax.random.PRNGKey(self.step)
        _, t_key, noise_key = jax.random.split(key, 3)
        t = np.array(jax.random.randint(t_key, (batch_size,), 0, num_timesteps))
        noise = np.array(jax.random.normal(noise_key, images.shape).astype("float32"))

        # Forward diffusion
        alpha_bar_t = schedule["alphas_cumprod"][t]
        sqrt_alpha = np.sqrt(alpha_bar_t).reshape(-1, 1, 1, 1)
        sqrt_one_minus = np.sqrt(1.0 - alpha_bar_t).reshape(-1, 1, 1, 1)
        x_t = (sqrt_alpha * images + sqrt_one_minus * noise).astype("float32")

        x_tensor = ops.convert_to_tensor(x_t)
        t_tensor = ops.convert_to_tensor(t.astype("int32"))
        c_tensor = ops.convert_to_tensor(class_ids.astype("int32"))
        noise_tensor = ops.convert_to_tensor(noise)

        # Loss function
        def _loss_fn(var_values):
            original_values = []
            for i, v in enumerate(self.model.trainable_variables):
                original_values.append(np.array(v.value))
                v.assign(var_values[i])

            eps_pred = self.model([x_tensor, t_tensor, c_tensor], training=True)
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
