"""Tests for evaluation metrics module.

Tests classifier building, feature extraction, FID computation,
and classification accuracy.
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np
import pytest


@pytest.fixture
def small_classifier():
    """Build a small classifier for testing."""
    from diffusion_harness.metrics.classifier import build_classifier
    return build_classifier(image_size=28, channels=1, num_classes=10)


@pytest.fixture
def small_feature_extractor(small_classifier):
    """Build feature extractor from small classifier."""
    from diffusion_harness.metrics.classifier import build_feature_extractor
    return build_feature_extractor(small_classifier)


class TestClassifier:
    def test_build_classifier_shape(self, small_classifier):
        """Classifier produces correct output shape."""
        x = np.random.randn(4, 28, 28, 1).astype('float32')
        out = small_classifier(x, training=False)
        assert np.array(out).shape == (4, 10)

    def test_classifier_output_is_softmax(self, small_classifier):
        """Classifier outputs sum to 1 (softmax)."""
        x = np.random.randn(4, 28, 28, 1).astype('float32')
        out = np.array(small_classifier(x, training=False))
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)

    def test_feature_extractor_shape(self, small_feature_extractor):
        """Feature extractor outputs (batch, 128)."""
        x = np.random.randn(4, 28, 28, 1).astype('float32')
        features = small_feature_extractor(x, training=False)
        assert np.array(features).shape == (4, 128)

    def test_train_classifier(self):
        """Classifier trains and achieves >50% accuracy on Fashion-MNIST subset."""
        from diffusion_harness.metrics.classifier import train_classifier
        import keras
        (x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
        x_train = (x_train[:1000].astype('float32') / 127.5) - 1.0
        x_train = x_train[..., np.newaxis]
        y_train = y_train[:1000]

        classifier, feat_ext, val_acc = train_classifier(
            x_train, y_train, epochs=3, batch_size=64, validation_split=0.2
        )
        assert val_acc > 0.5, f"Classifier accuracy too low: {val_acc}"
        assert np.array(feat_ext(x_train[:2], training=False)).shape == (2, 128)


class TestFID:
    def test_fid_identical_is_near_zero(self, small_feature_extractor):
        """FID between identical distributions is near 0."""
        from diffusion_harness.metrics.fid import compute_fid
        images = np.random.randn(100, 28, 28, 1).astype('float32')
        features = np.array(small_feature_extractor(images, training=False))
        fid = compute_fid(features, features)
        assert fid < 1.0, f"FID for identical distributions should be ~0, got {fid}"

    def test_fid_different_is_positive(self, small_feature_extractor):
        """FID between different distributions is positive."""
        from diffusion_harness.metrics.fid import compute_fid
        images1 = np.random.randn(200, 28, 28, 1).astype('float32') * 0.5
        images2 = np.random.randn(200, 28, 28, 1).astype('float32') + 2.0
        f1 = np.array(small_feature_extractor(images1, training=False))
        f2 = np.array(small_feature_extractor(images2, training=False))
        fid = compute_fid(f1, f2)
        assert fid > 0, f"FID for different distributions should be >0, got {fid}"

    def test_extract_features_shape(self, small_feature_extractor):
        """Feature extraction returns correct shape."""
        from diffusion_harness.metrics.fid import extract_features
        images = np.random.randn(10, 28, 28, 1).astype('float32')
        features = extract_features(images, small_feature_extractor, batch_size=4)
        assert features.shape == (10, 128)


class TestClassifierAccuracy:
    def test_accuracy_with_labels(self, small_classifier):
        """Classification accuracy returns correct dict keys."""
        from diffusion_harness.metrics.fid import compute_classifier_accuracy
        images = np.random.randn(20, 28, 28, 1).astype('float32')
        labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2)
        result = compute_classifier_accuracy(images, small_classifier, labels=labels)
        assert 'accuracy' in result
        assert 'per_class_accuracy' in result
        assert 'predictions' in result
        assert 'class_distribution' in result
        assert len(result['predictions']) == 20
        assert 0 <= result['accuracy'] <= 1

    def test_accuracy_without_labels(self, small_classifier):
        """Without labels, returns distribution but not accuracy."""
        from diffusion_harness.metrics.fid import compute_classifier_accuracy
        images = np.random.randn(10, 28, 28, 1).astype('float32')
        result = compute_classifier_accuracy(images, small_classifier)
        assert 'accuracy' not in result
        assert 'predictions' in result
        assert 'class_distribution' in result
