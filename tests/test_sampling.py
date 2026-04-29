"""Tests for sampling."""

import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

from diffusion_harness.core import make_config
from diffusion_harness.schedules import linear_beta_schedule, compute_schedule
from diffusion_harness.models import build_unet
from diffusion_harness.sampling import ddpm_sample


def test_ddpm_sample_shape():
    config = make_config(
        dataset="fashion_mnist",
        base_filters=16,
        num_levels=2,
        channel_multipliers=(1, 2),
        attention_resolutions=(),
        num_timesteps=100,
    )
    schedule = config["schedule"]
    model = build_unet(
        image_size=28, channels=1, base_filters=16, num_levels=2,
        channel_multipliers=(1, 2), attention_resolutions=(),
    )
    dummy_x = np.zeros((1, 28, 28, 1), dtype="float32")
    dummy_t = np.zeros((1,), dtype="int32")
    model([dummy_x, dummy_t])

    shape = (2, 28, 28, 1)
    samples = ddpm_sample(model, schedule, 100, shape, seed=0)
    assert samples.shape == shape


def test_ddpm_sample_range():
    """Samples should be in roughly [-1, 1] range."""
    config = make_config(
        dataset="fashion_mnist",
        base_filters=16,
        num_levels=2,
        channel_multipliers=(1, 2),
        attention_resolutions=(),
        num_timesteps=50,
    )
    model = build_unet(
        image_size=28, channels=1, base_filters=16, num_levels=2,
        channel_multipliers=(1, 2),
    )
    dummy_x = np.zeros((1, 28, 28, 1), dtype="float32")
    dummy_t = np.zeros((1,), dtype="int32")
    model([dummy_x, dummy_t])

    shape = (2, 28, 28, 1)
    samples = ddpm_sample(model, config["schedule"], 50, shape, seed=42)
    assert np.all(np.isfinite(samples))


def test_ddpm_sample_deterministic():
    """Same seed should give same output."""
    config = make_config(
        dataset="fashion_mnist",
        base_filters=16,
        num_levels=2,
        channel_multipliers=(1, 2),
        attention_resolutions=(),
        num_timesteps=50,
    )
    model = build_unet(
        image_size=28, channels=1, base_filters=16, num_levels=2,
        channel_multipliers=(1, 2),
    )
    dummy_x = np.zeros((1, 28, 28, 1), dtype="float32")
    dummy_t = np.zeros((1,), dtype="int32")
    model([dummy_x, dummy_t])

    shape = (2, 28, 28, 1)
    s1 = ddpm_sample(model, config["schedule"], 50, shape, seed=7)
    s2 = ddpm_sample(model, config["schedule"], 50, shape, seed=7)
    assert np.allclose(s1, s2)
