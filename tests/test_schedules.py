"""Tests for noise schedules."""

import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

from diffusion_harness.schedules import linear_beta_schedule, cosine_beta_schedule, compute_schedule


def test_linear_beta_schedule_shape():
    betas = linear_beta_schedule(1000)
    assert betas.shape == (1000,)


def test_linear_beta_schedule_range():
    betas = linear_beta_schedule(1000)
    assert betas[0] == pytest.approx(1e-4, rel=1e-3)
    assert betas[-1] == pytest.approx(0.02, rel=1e-3)


def test_cosine_beta_schedule_shape():
    betas = cosine_beta_schedule(1000)
    assert betas.shape == (1000,)


def test_cosine_beta_schedule_bounded():
    betas = cosine_beta_schedule(1000)
    assert np.all(betas >= 0) and np.all(betas < 1.0)


def test_compute_schedule_no_nan():
    for schedule_fn in [linear_beta_schedule, cosine_beta_schedule]:
        betas = schedule_fn(1000)
        schedule = compute_schedule(betas)
        for key, val in schedule.items():
            assert not np.any(np.isnan(val)), f"NaN in {key}"
            assert not np.any(np.isinf(val)), f"Inf in {key}"


def test_compute_schedule_keys():
    betas = linear_beta_schedule(1000)
    schedule = compute_schedule(betas)
    expected_keys = [
        "betas", "alphas", "alphas_cumprod",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        "sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod",
        "posterior_variance", "posterior_log_variance_clipped",
        "posterior_mean_coef1", "posterior_mean_coef2",
    ]
    for key in expected_keys:
        assert key in schedule, f"Missing key: {key}"


def test_alphas_cumprod_decreasing():
    betas = linear_beta_schedule(1000)
    schedule = compute_schedule(betas)
    ac = schedule["alphas_cumprod"]
    assert np.all(np.diff(ac) <= 0), "alphas_cumprod should be monotonically decreasing"
