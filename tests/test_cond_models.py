"""Tests for class-conditional U-Net model."""

import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import pytest
from diffusion_harness.methods.class_conditional.models import (
    build_cond_unet, ClassEmbedding,
)


def test_build_cond_unet_fashion_mnist():
    """Conditional U-Net builds and produces correct output shape."""
    model = build_cond_unet(
        image_size=28, channels=1, base_filters=32, num_levels=3,
        channel_multipliers=(1, 2, 2), attention_resolutions=(0,),
        num_classes=10,
    )
    dummy_x = np.zeros((2, 28, 28, 1), dtype="float32")
    dummy_t = np.zeros((2,), dtype="int32")
    dummy_c = np.zeros((2,), dtype="int32")
    output = model([dummy_x, dummy_t, dummy_c])
    assert output.shape == (2, 28, 28, 1), f"Expected (2,28,28,1), got {output.shape}"


def test_cond_unet_differs_by_class():
    """Different class IDs should produce different outputs."""
    model = build_cond_unet(
        image_size=28, channels=1, base_filters=32, num_levels=3,
        channel_multipliers=(1, 2, 2), attention_resolutions=(0,),
        num_classes=10,
    )
    x = np.random.randn(1, 28, 28, 1).astype("float32")
    t = np.array([500], dtype="int32")
    c0 = np.array([0], dtype="int32")
    c5 = np.array([5], dtype="int32")

    out0 = model([x, t, c0])
    out5 = model([x, t, c5])
    diff = np.abs(np.array(out0) - np.array(out5)).mean()
    assert diff > 1e-6, "Different class IDs should produce different outputs"


def test_null_class_embedding():
    """Null class ID (num_classes) runs without error."""
    model = build_cond_unet(
        image_size=28, channels=1, base_filters=32, num_levels=3,
        channel_multipliers=(1, 2, 2), attention_resolutions=(0,),
        num_classes=10,
    )
    dummy_x = np.zeros((1, 28, 28, 1), dtype="float32")
    dummy_t = np.zeros((1,), dtype="int32")
    null_c = np.array([10], dtype="int32")  # null class = num_classes
    output = model([dummy_x, dummy_t, null_c])
    assert output.shape == (1, 28, 28, 1)


def test_class_embedding_shape():
    """ClassEmbedding output dimensions match time_dim."""
    emb = ClassEmbedding(num_classes=10, embedding_dim=64)
    class_ids = np.array([0, 3, 9], dtype="int32")
    out = emb(class_ids)
    assert out.shape == (3, 64), f"Expected (3, 64), got {out.shape}"
