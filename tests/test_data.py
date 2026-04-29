"""Tests for data loading."""

import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

from diffusion_harness.data import load_dataset, denormalize, make_dataset, get_dataset_info


def test_load_fashion_mnist():
    images = load_dataset("fashion_mnist")
    assert images.dtype == np.float32
    assert images.min() >= -1.0 and images.max() <= 1.0
    assert images.shape == (60000, 28, 28, 1)


def test_load_mnist():
    images = load_dataset("mnist")
    assert images.shape == (60000, 28, 28, 1)


def test_load_cifar10():
    images = load_dataset("cifar10")
    assert images.shape == (50000, 32, 32, 3)


def test_load_subset():
    images = load_dataset("fashion_mnist", subset_size=100, seed=42)
    assert images.shape[0] == 100


def test_denormalize():
    images = np.array([-1.0, 0.0, 1.0], dtype=np.float32).reshape(1, 1, 1, 3)
    result = denormalize(images)
    assert result[0, 0, 0, 0] == 0
    assert result[0, 0, 0, 1] == 127
    assert result[0, 0, 0, 2] == 255
    assert result.dtype == np.uint8


def test_make_dataset():
    images = np.random.randn(32, 8, 8, 3).astype("float32")
    ds = make_dataset(images, batch_size=4, shuffle=False)
    batch = ds()
    assert batch.shape == (4, 8, 8, 3)


def test_get_dataset_info():
    info = get_dataset_info("cifar10")
    assert info["image_size"] == 32
    assert info["channels"] == 3

    info = get_dataset_info("fashion_mnist")
    assert info["image_size"] == 28
    assert info["channels"] == 1
