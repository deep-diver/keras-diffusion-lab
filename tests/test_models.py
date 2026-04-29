"""Tests for the U-Net model."""

import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

from diffusion_harness.models import build_unet


def test_build_unet_fashion_mnist():
    """Test model build for 28x28x1 input."""
    model = build_unet(image_size=28, channels=1, base_filters=32, num_levels=3,
                       channel_multipliers=(1, 2, 2), attention_resolutions=(1,))
    dummy_x = np.zeros((2, 28, 28, 1), dtype="float32")
    dummy_t = np.zeros((2,), dtype="int32")
    out = model([dummy_x, dummy_t])
    assert out.shape == (2, 28, 28, 1)


def test_build_unet_cifar10():
    """Test model build for 32x32x3 input."""
    model = build_unet(image_size=32, channels=3, base_filters=64, num_levels=4,
                       channel_multipliers=(1, 2, 2, 2), attention_resolutions=(1, 2))
    dummy_x = np.zeros((2, 32, 32, 3), dtype="float32")
    dummy_t = np.zeros((2,), dtype="int32")
    out = model([dummy_x, dummy_t])
    assert out.shape == (2, 32, 32, 3)


def test_model_output_range():
    """Model output should be finite for valid input."""
    model = build_unet(image_size=28, channels=1, base_filters=32, num_levels=3,
                       channel_multipliers=(1, 2, 2))
    x = np.random.randn(4, 28, 28, 1).astype("float32")
    t = np.array([0, 100, 500, 999], dtype="int32")
    out = model([x, t])
    assert np.all(np.isfinite(out))


def test_model_different_timesteps():
    """Model should give different outputs for different timesteps."""
    model = build_unet(image_size=28, channels=1, base_filters=32, num_levels=3,
                       channel_multipliers=(1, 2, 2))
    x = np.random.randn(1, 28, 28, 1).astype("float32")
    out_t0 = model([x, np.array([0], dtype="int32")])
    out_t500 = model([x, np.array([500], dtype="int32")])
    assert not np.allclose(out_t0, out_t500)
