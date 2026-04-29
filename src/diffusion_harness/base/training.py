"""Base trainer class for diffusion models.

Provides shared infrastructure: EMA weight management, checkpoint
save/load, train loop with logging hooks. Subclasses implement
train_step() for method-specific loss computation.
"""

import os
import json
import time
import numpy as np
import jax
import keras
import keras.ops as ops


class BaseTrainer:
    """Base trainer with EMA, checkpointing, and monitoring.

    Subclasses must implement train_step(batch) -> dict.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.step = 0
        self.loss_history = []
        self.ema_decay = config.get("ema_decay", 0.999)
        self.ema_weights = None

        self.optimizer = keras.optimizers.Adam(learning_rate=config["learning_rate"])

        if self.ema_decay > 0:
            self.ema_weights = [np.array(v.value) for v in self.model.trainable_variables]

    def train_step(self, batch):
        """Execute one training step. Must be overridden by subclasses.

        Args:
            batch: Training data (format depends on method).

        Returns:
            Dict with loss, grad_norm, nan_detected.
        """
        raise NotImplementedError("Subclasses must implement train_step()")

    def train(self, data_iter, num_steps, checkpoint_dir=None,
              sample_fn=None, event_log=None):
        """Run the training loop."""
        config = self.config
        log_loss_every = config.get("log_loss_every", 10)
        log_health_every = config.get("log_health_every", 100)
        checkpoint_every = config.get("checkpoint_every", 500)
        sample_every = config.get("sample_every", 250)

        start_time = time.time()

        for i in range(num_steps):
            batch = data_iter()
            metrics = self.train_step(batch)
            current_step = self.step

            if current_step % 100 == 0:
                elapsed = time.time() - start_time
                sps = (i + 1) / max(elapsed, 0.001)
                print(f"  step {current_step}: loss={metrics['loss']:.4f} "
                      f"grad_norm={metrics['grad_norm']:.2f} "
                      f"({sps:.1f} steps/s)")

            if current_step % log_loss_every == 0 and event_log:
                ema_loss = (np.mean(self.loss_history[-100:])
                            if len(self.loss_history) >= 10
                            else metrics["loss"])
                event_log.log_loss(
                    step=current_step, loss=metrics["loss"],
                    ema_loss=float(ema_loss),
                )

            if current_step % log_health_every == 0 and event_log:
                event_log.log_health(
                    step=current_step,
                    grad_norm=metrics["grad_norm"],
                    nan_detected=metrics["nan_detected"],
                )

            if checkpoint_dir and current_step % checkpoint_every == 0:
                self.save_checkpoint(checkpoint_dir, current_step)

            if sample_fn and current_step % sample_every == 0:
                sample_fn(current_step, self.model)

        return self.loss_history

    def get_ema_model(self):
        """Load EMA weights into model."""
        if self.ema_weights is None:
            return self.model
        self._training_weights = [np.array(v.value) for v in self.model.trainable_variables]
        for i, v in enumerate(self.model.trainable_variables):
            v.assign(self.ema_weights[i])
        return self.model

    def restore_training_weights(self):
        """Restore training weights after using EMA model."""
        if hasattr(self, "_training_weights"):
            for i, v in enumerate(self.model.trainable_variables):
                v.assign(self._training_weights[i])
            del self._training_weights

    def save_checkpoint(self, directory, step):
        """Save model, EMA, and optimizer state."""
        os.makedirs(directory, exist_ok=True)

        model_path = os.path.join(directory, f"model_step{step}.weights.h5")
        self.model.save_weights(model_path)

        if self.ema_weights is not None:
            ema_path = os.path.join(directory, f"ema_step{step}.weights.h5")
            training_backup = [np.array(v.value) for v in self.model.trainable_variables]
            for i, v in enumerate(self.model.trainable_variables):
                v.assign(self.ema_weights[i])
            self.model.save_weights(ema_path)
            for i, v in enumerate(self.model.trainable_variables):
                v.assign(training_backup[i])

        opt_vars = {}
        for v in self.optimizer.variables:
            opt_vars[v.name] = np.array(v.value)
        opt_path = os.path.join(directory, f"optimizer_step{step}.npz")
        np.savez(opt_path, **opt_vars)

        state = {"step": step, "loss_history_length": len(self.loss_history)}
        state_path = os.path.join(directory, f"state_step{step}.json")
        with open(state_path, "w") as f:
            json.dump(state, f)

        print(f"    Checkpoint saved: step {step}")

    def load_checkpoint(self, path):
        """Load model, EMA, and optimizer state."""
        self.model.load_weights(path)

        directory = os.path.dirname(path)
        basename = os.path.basename(path)

        step_str = ""
        for part in basename.split("_"):
            if part.startswith("step"):
                step_str = part.replace("step", "").split(".")[0]
                break
        if not step_str:
            return
        step = int(step_str)
        self.step = step

        ema_path = os.path.join(directory, f"ema_step{step}.weights.h5")
        if os.path.exists(ema_path) and self.ema_weights is not None:
            training_backup = [np.array(v.value) for v in self.model.trainable_variables]
            self.model.load_weights(ema_path)
            self.ema_weights = [np.array(v.value) for v in self.model.trainable_variables]
            for i, v in enumerate(self.model.trainable_variables):
                v.assign(training_backup[i])

        opt_path = os.path.join(directory, f"optimizer_step{step}.npz")
        if os.path.exists(opt_path):
            data = np.load(opt_path, allow_pickle=True)
            for v in self.optimizer.variables:
                if v.name in data:
                    v.assign(data[v.name])
