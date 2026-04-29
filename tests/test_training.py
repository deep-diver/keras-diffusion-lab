"""Tests for the training loop."""

import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

from diffusion_harness.core import make_config
from diffusion_harness.training import DiffusionTrainer


def test_trainer_creates():
    config = make_config(
        dataset="fashion_mnist",
        base_filters=16,
        num_levels=2,
        channel_multipliers=(1, 2),
        attention_resolutions=(),
        num_train_steps=10,
    )
    trainer = DiffusionTrainer(config)
    assert trainer.model is not None
    assert trainer.step == 0
    assert trainer.ema_weights is not None


def test_train_step_runs():
    config = make_config(
        dataset="fashion_mnist",
        base_filters=16,
        num_levels=2,
        channel_multipliers=(1, 2),
        attention_resolutions=(),
        num_timesteps=100,
        num_train_steps=5,
        batch_size=2,
    )
    trainer = DiffusionTrainer(config)
    batch = np.random.randn(2, 28, 28, 1).astype("float32")
    metrics = trainer.train_step(batch)
    assert "loss" in metrics
    assert "grad_norm" in metrics
    assert "nan_detected" in metrics
    assert metrics["nan_detected"] is False
    assert trainer.step == 1


def test_train_step_loss_decreases():
    """Over a few steps, loss should generally decrease."""
    config = make_config(
        dataset="fashion_mnist",
        base_filters=16,
        num_levels=2,
        channel_multipliers=(1, 2),
        attention_resolutions=(),
        num_timesteps=100,
        num_train_steps=20,
        batch_size=4,
        learning_rate=1e-3,
    )
    trainer = DiffusionTrainer(config)
    losses = []
    for _ in range(20):
        batch = np.random.randn(4, 28, 28, 1).astype("float32")
        metrics = trainer.train_step(batch)
        losses.append(metrics["loss"])
    # Loss should not be NaN
    assert all(np.isfinite(l) for l in losses)
    # Average of last 5 should be less than average of first 5
    assert np.mean(losses[-5:]) < np.mean(losses[:5])


def test_ema_updates():
    config = make_config(
        dataset="fashion_mnist",
        base_filters=16,
        num_levels=2,
        channel_multipliers=(1, 2),
        attention_resolutions=(),
        ema_decay=0.9,  # Low decay so EMA changes visibly
    )
    trainer = DiffusionTrainer(config)
    initial_ema = [w.copy() for w in trainer.ema_weights]
    batch = np.random.randn(2, 28, 28, 1).astype("float32")
    trainer.train_step(batch)
    # EMA should have changed (with decay=0.9, 10% of the step moves)
    any_changed = False
    for i in range(len(initial_ema)):
        if not np.allclose(initial_ema[i], trainer.ema_weights[i], atol=1e-6):
            any_changed = True
    assert any_changed, "EMA weights did not change after a training step"
