"""Evaluation metrics for diffusion model quality assessment.

Provides FID computation using a trained classifier's features
(domain-appropriate for small grayscale images), and classification
accuracy for conditional models.
"""

from diffusion_harness.metrics.classifier import (
    build_classifier,
    build_feature_extractor,
    train_classifier,
)
from diffusion_harness.metrics.fid import (
    compute_fid,
    compute_fid_from_stats,
    compute_classifier_accuracy,
    extract_features,
)
