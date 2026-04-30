"""Tests for DDIM sampler.

Tests DDIMSampler and CFGDDIMSampler with small models.
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np
import pytest


@pytest.fixture
def unet_small():
    """Build a small unconditional U-Net for testing."""
    from diffusion_harness.base.models import build_unet
    model = build_unet(
        image_size=28, channels=1, base_filters=16,
        num_levels=2, channel_multipliers=(1, 2),
        attention_resolutions=(1,)
    )
    return model


@pytest.fixture
def cond_unet_small():
    """Build a small conditional U-Net for testing."""
    from diffusion_harness.methods.class_conditional.models import build_cond_unet
    model = build_cond_unet(
        image_size=28, channels=1, base_filters=16,
        num_levels=2, channel_multipliers=(1, 2),
        attention_resolutions=(1,), num_classes=10
    )
    return model


@pytest.fixture
def schedule_small():
    """Create a small schedule for testing (50 timesteps)."""
    from diffusion_harness.schedules import linear_beta_schedule, compute_schedule
    betas = linear_beta_schedule(50)
    return compute_schedule(betas)


class TestDDIMSampler:
    def test_ddim_sample_shape(self, unet_small, schedule_small):
        """DDIM produces correct output shape."""
        from diffusion_harness.base.ddim_sampling import DDIMSampler
        sampler = DDIMSampler(unet_small, schedule_small, num_timesteps=50,
                              eta=0.0, subsequence_size=10)
        samples = sampler.sample(shape=(4, 28, 28, 1), seed=42)
        assert samples.shape == (4, 28, 28, 1)

    def test_ddim_sample_finite(self, unet_small, schedule_small):
        """DDIM output is finite (no NaN/Inf)."""
        from diffusion_harness.base.ddim_sampling import DDIMSampler
        sampler = DDIMSampler(unet_small, schedule_small, num_timesteps=50,
                              eta=0.0, subsequence_size=10)
        samples = sampler.sample(shape=(2, 28, 28, 1), seed=0)
        assert np.all(np.isfinite(samples))

    def test_ddim_deterministic_eta0(self, unet_small, schedule_small):
        """DDIM with eta=0 is deterministic (same seed gives same result)."""
        from diffusion_harness.base.ddim_sampling import DDIMSampler
        sampler = DDIMSampler(unet_small, schedule_small, num_timesteps=50,
                              eta=0.0, subsequence_size=10)
        s1 = sampler.sample(shape=(2, 28, 28, 1), seed=123)
        s2 = sampler.sample(shape=(2, 28, 28, 1), seed=123)
        np.testing.assert_array_equal(s1, s2)

    def test_ddim_various_subsequence_sizes(self, unet_small, schedule_small):
        """DDIM works with various subsequence sizes."""
        from diffusion_harness.base.ddim_sampling import DDIMSampler
        for size in [5, 10, 25, 50]:
            sampler = DDIMSampler(unet_small, schedule_small, num_timesteps=50,
                                  eta=0.0, subsequence_size=size)
            samples = sampler.sample(shape=(2, 28, 28, 1), seed=42)
            assert samples.shape == (2, 28, 28, 1)
            assert np.all(np.isfinite(samples))

    def test_ddim_convenience_function(self, unet_small, schedule_small):
        """ddim_sample convenience function works."""
        from diffusion_harness.base.ddim_sampling import ddim_sample
        samples = ddim_sample(unet_small, schedule_small, num_timesteps=50,
                              shape=(2, 28, 28, 1), seed=0, subsequence_size=10)
        assert samples.shape == (2, 28, 28, 1)


class TestCFGDDIMSampler:
    def test_cfg_ddim_sample_shape(self, cond_unet_small, schedule_small):
        """CFG DDIM produces correct output shape."""
        from diffusion_harness.methods.class_conditional.ddim_sampling import CFGDDIMSampler
        sampler = CFGDDIMSampler(cond_unet_small, schedule_small, num_timesteps=50,
                                 guidance_scale=3.0, num_classes=10,
                                 eta=0.0, subsequence_size=10)
        class_ids = np.array([0, 1, 2, 3], dtype=np.int32)
        samples = sampler.sample(shape=(4, 28, 28, 1), class_ids=class_ids, seed=42)
        assert samples.shape == (4, 28, 28, 1)

    def test_cfg_ddim_guidance_effect(self, cond_unet_small, schedule_small):
        """Different guidance scales produce different results."""
        from diffusion_harness.methods.class_conditional.ddim_sampling import CFGDDIMSampler
        class_ids = np.array([0, 1, 2, 3], dtype=np.int32)

        sampler_w1 = CFGDDIMSampler(cond_unet_small, schedule_small, num_timesteps=50,
                                    guidance_scale=1.0, num_classes=10,
                                    eta=0.0, subsequence_size=10)
        sampler_w5 = CFGDDIMSampler(cond_unet_small, schedule_small, num_timesteps=50,
                                    guidance_scale=5.0, num_classes=10,
                                    eta=0.0, subsequence_size=10)

        s1 = sampler_w1.sample(shape=(4, 28, 28, 1), class_ids=class_ids, seed=42)
        s5 = sampler_w5.sample(shape=(4, 28, 28, 1), class_ids=class_ids, seed=42)

        # Different guidance scales should produce different outputs
        assert not np.allclose(s1, s5, atol=1e-4)

    def test_cfg_ddim_convenience_function(self, cond_unet_small, schedule_small):
        """cfg_ddim_sample convenience function works."""
        from diffusion_harness.methods.class_conditional.ddim_sampling import cfg_ddim_sample
        class_ids = np.array([0, 1], dtype=np.int32)
        samples = cfg_ddim_sample(
            cond_unet_small, schedule_small, num_timesteps=50,
            shape=(2, 28, 28, 1), class_ids=class_ids,
            guidance_scale=3.0, seed=0, subsequence_size=10
        )
        assert samples.shape == (2, 28, 28, 1)
