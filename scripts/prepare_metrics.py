"""Prepare evaluation metrics: train classifier and compute real data statistics.

One-time setup script. Trains a Fashion-MNIST classifier, extracts features
from the full training set, and saves:
  - metrics_classifier.weights.h5  (classifier weights)
  - metrics_real_stats.npz         (mu, sigma of real feature distribution)

Usage:
    KERAS_BACKEND=jax python scripts/prepare_metrics.py \
        --dataset fashion_mnist \
        --output-dir artifacts/metrics
"""

import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Prepare evaluation metrics")
    parser.add_argument("--dataset", default="fashion_mnist",
                        choices=["fashion_mnist", "mnist", "cifar10"])
    parser.add_argument("--output-dir", default="artifacts/metrics")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    from diffusion_harness.data import load_dataset
    images, labels = load_dataset(args.dataset, return_labels=True)
    print(f"Dataset: {images.shape}, labels: {labels.shape}")

    # Train classifier
    from diffusion_harness.metrics.classifier import build_classifier, build_feature_extractor
    classifier_path = os.path.join(args.output_dir, "metrics_classifier.weights.h5")

    if os.path.exists(classifier_path):
        print(f"Loading existing classifier from {classifier_path}")
        classifier = build_classifier(
            image_size=images.shape[1],
            channels=images.shape[-1],
            num_classes=int(labels.max()) + 1
        )
        classifier.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        # Build model by running a forward pass
        classifier.predict(images[:1], verbose=0)
        classifier.load_weights(classifier_path)
        feature_extractor = build_feature_extractor(classifier)
        print("  Loaded.")
    else:
        print("Training classifier...")
        from diffusion_harness.metrics.classifier import train_classifier
        classifier, feature_extractor, val_acc = train_classifier(
            images, labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=classifier_path,
        )
        print(f"  Validation accuracy: {val_acc:.4f}")
        print(f"  Saved to {classifier_path}")

    # Extract features from full training set
    print("Extracting features from training set...")
    from diffusion_harness.metrics.fid import extract_features
    features = extract_features(images, feature_extractor, batch_size=512)
    print(f"  Features shape: {features.shape}")

    # Compute and save statistics
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    print(f"  Feature mean range: [{mu.min():.4f}, {mu.max():.4f}]")
    print(f"  Feature cov shape: {sigma.shape}")

    stats_path = os.path.join(args.output_dir, "metrics_real_stats.npz")
    np.savez(stats_path, mu=mu, sigma=sigma)
    print(f"  Saved real stats to {stats_path}")

    print("Done! Metrics ready for evaluation.")


if __name__ == "__main__":
    main()
