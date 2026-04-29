"""Tests for CFG sampler."""

import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import pytest
from diffusion_harness.core import make_config
from diffusion_harness.methods.class_conditional.models import build_cond_unet
from diffusion_harness.methods.class_conditional.sampling import cfg_sample


def _make_config_and_model():
    config = make_config(
        dataset="fashion_mnist", method="class_conditional",
        base_filters=32, num_levels=3,
        channel_multipliers=(1, 2, 2), attention_resolutions=(0,),
        num_classes=10,
    )
    model = build_cond_unet(
        image_size=28, channels=1, base_filters=32, num_levels=3,
        channel_multipliers=(1, 2, 2), attention_resolutions=(0,),
        num_classes=10,
    )
    # Build model
    dummy_x = np.zeros((1, 28, 28, 1), dtype="float32")
    dummy_t = np.zeros((1,), dtype="int32")
    dummy_c = np.zeros((1,), dtype="int32")
    model([dummy_x, dummy_t, dummy_c])
    return config, model


def test_cfg_sample_shape():
    """CFG sampling produces correct output shape."""
    config, model = _make_config_and_model()
    shape = (2, 28, 28, 1)
    class_ids = np.array([0, 5], dtype="int32")
    samples = cfg_sample(
        model, config["schedule"], config["num_timesteps"],
        shape, class_ids=class_ids, num_classes=10,
        guidance_scale=3.0, seed=42,
    )
    assert samples.shape == shape, f"Expected {shape}, got {samples.shape}"


def test_cfg_guidance_effect():
    """Different guidance scales should produce different outputs."""
    config, model = _make_config_and_model()
    shape = (1, 28, 28, 1)
    class_ids = np.array([3], dtype="int32")

    # Use same noise for fair comparison
    import jax
    rng = jax.random.PRNGKey(42)
    noise = np.array(jax.random.normal(rng, shape).astype("float32"))

    s_low = cfg_sample(model, config["schedule"], config["num_timesteps"],
                       shape, class_ids=class_ids, num_classes=10,
                       guidance_scale=1.0, seed=42, initial_noise=noise)

    s_high = cfg_sample(model, config["schedule"], config["num_timesteps"],
                        shape, class_ids=class_ids, num_classes=10,
                        guidance_scale=5.0, seed=42, initial_noise=noise)

    diff = np.abs(s_low - s_high).mean()
    assert diff > 1e-6, "Different guidance scales should produce different outputs"


def test_cfg_sample_deterministic():
    """Same seed produces identical output."""
    config, model = _make_config_and_model()
    shape = (2, 28, 28, 1)
    class_ids = np.array([0, 5], dtype="int32")

    s1 = cfg_sample(model, config["schedule"], config["num_timesteps"],
                    shape, class_ids=class_ids, num_classes=10,
                    guidance_scale=3.0, seed=123)

    s2 = cfg_sample(model, config["schedule"], config["num_timesteps"],
                    shape, class_ids=class_ids, num_classes=10,
                    guidance_scale=3.0, seed=123)

    np.testing.assert_array_equal(s1, s2, "Same seed should produce identical output")
