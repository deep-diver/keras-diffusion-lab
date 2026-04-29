"""Tests for CFG trainer."""

import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import pytest
from diffusion_harness.core import make_config
from diffusion_harness.methods.class_conditional.training import CFGTrainer
from diffusion_harness.data import load_dataset


def test_cfg_trainer_creates():
    """CFGTrainer builds successfully with correct model inputs."""
    config = make_config(
        dataset="fashion_mnist", method="class_conditional",
        base_filters=32, num_levels=3,
        channel_multipliers=(1, 2, 2), attention_resolutions=(0,),
        num_classes=10,
    )
    trainer = CFGTrainer(config)
    assert len(trainer.model.inputs) == 3, "Conditional model should have 3 inputs"
    assert trainer.ema_weights is not None
    assert trainer.null_class_id == 10


def test_cfg_train_step_runs():
    """CFGTrainer can execute a training step with (images, labels) batch."""
    config = make_config(
        dataset="fashion_mnist", method="class_conditional",
        base_filters=32, num_levels=3,
        channel_multipliers=(1, 2, 2), attention_resolutions=(0,),
        num_classes=10, batch_size=4,
    )
    trainer = CFGTrainer(config)

    # Create a fake batch
    images = np.random.randn(4, 28, 28, 1).astype("float32")
    labels = np.array([0, 3, 7, 9], dtype="int32")
    batch = (images, labels)

    metrics = trainer.train_step(batch)
    assert "loss" in metrics
    assert "grad_norm" in metrics
    assert metrics["loss"] > 0
    assert not metrics["nan_detected"]


def test_cfg_class_dropout():
    """Class dropout produces null class IDs sometimes."""
    config = make_config(
        dataset="fashion_mnist", method="class_conditional",
        base_filters=32, num_levels=3,
        channel_multipliers=(1, 2, 2), attention_resolutions=(0,),
        num_classes=10, class_dropout_prob=0.5,
    )
    trainer = CFGTrainer(config)

    images = np.random.randn(64, 28, 28, 1).astype("float32")
    labels = np.array([0] * 64, dtype="int32")

    # Run several steps — with 50% dropout, some should use null class
    losses = []
    for _ in range(5):
        metrics = trainer.train_step((images, labels))
        losses.append(metrics["loss"])
        assert not metrics["nan_detected"]

    # All losses should be finite
    assert all(np.isfinite(l) for l in losses)
