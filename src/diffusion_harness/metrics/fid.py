"""FID computation and classification accuracy evaluation.

Uses a trained classifier's penultimate features for FID — appropriate
for small grayscale images where InceptionV3 is impractical.

FID = ||mu_1 - mu_2||^2 + Tr(sigma_1 + sigma_2 - 2 * sqrtm(sigma_1 @ sigma_2))
"""

import numpy as np
from scipy.linalg import sqrtm


def extract_features(images, feature_extractor, batch_size=256):
    """Extract features from images in batches.

    Args:
        images: (N, H, W, C) array in [-1, 1].
        feature_extractor: keras.Model outputting (batch, D) features.
        batch_size: Batch size for inference.

    Returns:
        (N, D) numpy array of features.
    """
    all_features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        features = feature_extractor(batch, training=False)
        all_features.append(np.array(features))
    return np.concatenate(all_features, axis=0)


def compute_fid(features_real, features_gen):
    """Compute Frechet Inception Distance between two feature sets.

    Args:
        features_real: (N, D) array of real image features.
        features_gen: (M, D) array of generated image features.

    Returns:
        FID score (float). Lower is better. 0 = identical distributions.
    """
    mu_real = np.mean(features_real, axis=0)
    mu_gen = np.mean(features_gen, axis=0)
    sigma_real = np.cov(features_real, rowvar=False)
    sigma_gen = np.cov(features_gen, rowvar=False)

    # Handle 1-d case (single feature dimension)
    if sigma_real.ndim == 0:
        sigma_real = np.array([[sigma_real]])
        sigma_gen = np.array([[sigma_gen]])

    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real @ sigma_gen)

    # Numerical stability: discard imaginary part if tiny
    if np.iscomplexobj(covmean):
        if np.allclose(np.imag(covmean), 0, atol=1e-3):
            covmean = np.real(covmean)

    fid = np.dot(diff, diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return float(fid)


def compute_fid_from_stats(mu_real, sigma_real, mu_gen, sigma_gen):
    """Compute FID from pre-computed statistics.

    Useful when real-data statistics are computed once and reused.

    Args:
        mu_real: (D,) mean of real features.
        sigma_real: (D, D) covariance of real features.
        mu_gen: (D,) mean of generated features.
        sigma_gen: (D, D) covariance of generated features.

    Returns:
        FID score (float).
    """
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real @ sigma_gen)

    if np.iscomplexobj(covmean):
        if np.allclose(np.imag(covmean), 0, atol=1e-3):
            covmean = np.real(covmean)

    fid = np.dot(diff, diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return float(fid)


def compute_classifier_accuracy(images, classifier, labels=None,
                                batch_size=256):
    """Compute classification accuracy on generated images.

    Args:
        images: (N, H, W, C) array in [-1, 1].
        classifier: Trained keras classifier.
        labels: Optional (N,) int array of intended classes.
                If provided, computes accuracy and per-class accuracy.
        batch_size: Batch size for inference.

    Returns:
        Dict with:
          - 'predictions': (N,) predicted class indices
          - 'class_distribution': dict of {class_id: count}
          - 'accuracy': float (only if labels provided)
          - 'per_class_accuracy': dict {class_id: accuracy} (only if labels provided)
    """
    predictions = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        probs = classifier(batch, training=False)
        preds = np.argmax(probs, axis=-1)
        predictions.append(preds)
    predictions = np.concatenate(predictions, axis=0)

    # Class distribution
    unique, counts = np.unique(predictions, return_counts=True)
    class_distribution = dict(zip(unique.tolist(), counts.tolist()))

    result = {
        'predictions': predictions,
        'class_distribution': class_distribution,
    }

    if labels is not None:
        labels = np.asarray(labels)
        correct = predictions == labels
        result['accuracy'] = float(np.mean(correct))

        # Per-class accuracy
        per_class = {}
        for c in np.unique(labels):
            mask = labels == c
            per_class[int(c)] = float(np.mean(predictions[mask] == c))
        result['per_class_accuracy'] = per_class

    return result
