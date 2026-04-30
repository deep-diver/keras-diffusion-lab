"""Guidance scale sweep: generate samples and evaluate at multiple w values.

Loads conditional 30K checkpoint, generates samples for w in [1, 3, 5, 7.5],
computes FID and classification accuracy for each, and produces a report.

Usage:
    KERAS_BACKEND=jax python scripts/guidance_sweep.py \
        --checkpoint artifacts/cfg-run/checkpoints/ema_step30000.weights.h5 \
        --output-dir artifacts/guidance_sweep \
        --samples-per-class 100 \
        --guidance-scales 1.0 3.0 5.0 7.5
"""

import argparse
import os
import time
import numpy as np


CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def main():
    parser = argparse.ArgumentParser(description="Guidance scale sweep")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to EMA checkpoint weights")
    parser.add_argument("--output-dir", default="artifacts/guidance_sweep")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--guidance-scales", type=float, nargs="+",
                        default=[1.0, 3.0, 5.0, 7.5])
    parser.add_argument("--metrics-dir", default="artifacts/metrics",
                        help="Directory with classifier weights and real stats")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Generation batch size per class")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load metrics infrastructure
    from diffusion_harness.data import load_dataset
    from diffusion_harness.metrics.classifier import build_classifier, build_feature_extractor
    from diffusion_harness.metrics.fid import (
        extract_features, compute_fid_from_stats, compute_classifier_accuracy
    )

    # Load classifier
    images_real, labels_real = load_dataset("fashion_mnist", return_labels=True)
    num_classes = 10
    classifier_path = os.path.join(args.metrics_dir, "metrics_classifier.weights.h5")
    stats_path = os.path.join(args.metrics_dir, "metrics_real_stats.npz")

    print("Loading classifier...")
    classifier = build_classifier(num_classes=num_classes)
    classifier.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    classifier.predict(images_real[:1], verbose=0)
    classifier.load_weights(classifier_path)
    feature_extractor = build_feature_extractor(classifier)
    print("  Loaded.")

    # Load real feature statistics
    real_stats = np.load(stats_path)
    mu_real = real_stats["mu"]
    sigma_real = real_stats["sigma"]
    print(f"  Real stats: mu={mu_real.shape}, sigma={sigma_real.shape}")

    # Build conditional model
    from diffusion_harness.methods.class_conditional.models import build_cond_unet
    from diffusion_harness.methods.class_conditional.sampling import CFGSampler
    from diffusion_harness.schedules import linear_beta_schedule, compute_schedule

    print("Building conditional model...")
    model = build_cond_unet(
        image_size=28, channels=1, base_filters=128,
        num_levels=3, channel_multipliers=(1, 2, 2, 2),
        attention_resolutions=(1, 2), num_classes=num_classes
    )
    model.load_weights(args.checkpoint)
    print(f"  Loaded checkpoint: {args.checkpoint}")

    betas = linear_beta_schedule(1000)
    schedule = compute_schedule(betas)

    # Run sweep
    results = []
    all_samples = {}

    for w in args.guidance_scales:
        print(f"\n=== Guidance scale w={w} ===")
        sampler = CFGSampler(model, schedule, num_timesteps=1000,
                             guidance_scale=w, num_classes=num_classes)

        # Generate samples for all classes
        all_gen = []
        all_labels = []
        gen_batch_size = args.batch_size

        for c in range(num_classes):
            class_samples = []
            for start in range(0, args.samples_per_class, gen_batch_size):
                n = min(gen_batch_size, args.samples_per_class - start)
                seed = int(c * 10000 + start)
                samples = sampler.sample(
                    shape=(n, 28, 28, 1),
                    class_ids=np.full(n, c, dtype=np.int32),
                    seed=seed
                )
                class_samples.append(samples)
            class_gen = np.concatenate(class_samples, axis=0)
            all_gen.append(class_gen)
            all_labels.extend([c] * len(class_gen))
            print(f"  Class {c} ({CLASS_NAMES[c]}): {len(class_gen)} samples")

        all_gen = np.concatenate(all_gen, axis=0)
        all_labels = np.array(all_labels, dtype=np.int32)
        all_samples[w] = all_gen

        # Compute metrics
        print("  Computing FID...")
        features_gen = extract_features(all_gen, feature_extractor, batch_size=512)
        mu_gen = np.mean(features_gen, axis=0)
        sigma_gen = np.cov(features_gen, rowvar=False)
        fid = compute_fid_from_stats(mu_real, sigma_real, mu_gen, sigma_gen)

        print("  Computing classification accuracy...")
        acc_result = compute_classifier_accuracy(
            all_gen, classifier, labels=all_labels
        )
        accuracy = acc_result['accuracy']
        per_class = acc_result['per_class_accuracy']

        result = {
            'guidance_scale': w,
            'fid': fid,
            'accuracy': accuracy,
            'per_class_accuracy': per_class,
            'n_samples': len(all_gen),
        }
        results.append(result)

        print(f"  FID: {fid:.2f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Per-class: {per_class}")

        # Save sample grid
        from diffusion_harness.base.sampling import save_image_grid
        grid_path = os.path.join(args.output_dir, f"samples_w{w:.1f}.png")
        # 10 samples per class for grid
        grid_samples = []
        for c in range(num_classes):
            start = c * args.samples_per_class
            grid_samples.append(all_gen[start:start + 10])
        grid_images = np.concatenate(grid_samples, axis=0)
        save_image_grid(grid_images, grid_path, nrow=10)
        print(f"  Saved grid to {grid_path}")

    # Save results
    print("\n=== Summary ===")
    print(f"{'w':>5} {'FID':>10} {'Accuracy':>10} {'N':>8}")
    print("-" * 35)
    for r in results:
        print(f"{r['guidance_scale']:5.1f} {r['fid']:10.2f} "
              f"{r['accuracy']:10.4f} {r['n_samples']:8d}")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "sweep_results.csv")
    with open(csv_path, 'w') as f:
        f.write("guidance_scale,fid,accuracy,n_samples\n")
        for r in results:
            f.write(f"{r['guidance_scale']:.1f},{r['fid']:.4f},"
                    f"{r['accuracy']:.4f},{r['n_samples']}\n")
    print(f"\nSaved CSV to {csv_path}")

    # Save results dict
    np.savez(
        os.path.join(args.output_dir, "sweep_results.npz"),
        guidance_scales=np.array([r['guidance_scale'] for r in results]),
        fids=np.array([r['fid'] for r in results]),
        accuracies=np.array([r['accuracy'] for r in results]),
    )

    # Generate plot
    _plot_results(results, args.output_dir)

    print("Done!")


def _plot_results(results, output_dir):
    """Generate summary plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in (ax1, ax2):
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#333366')

    ws = [r['guidance_scale'] for r in results]
    fids = [r['fid'] for r in results]
    accs = [r['accuracy'] for r in results]

    ax1.bar(range(len(ws)), fids, color='#4fc3f7')
    ax1.set_xticks(range(len(ws)))
    ax1.set_xticklabels([f'{w:.1f}' for w in ws])
    ax1.set_xlabel('Guidance Scale (w)')
    ax1.set_ylabel('FID (lower is better)')
    ax1.set_title('FID vs Guidance Scale')

    ax2.bar(range(len(ws)), accs, color='#e91e63')
    ax2.set_xticks(range(len(ws)))
    ax2.set_xticklabels([f'{w:.1f}' for w in ws])
    ax2.set_xlabel('Guidance Scale (w)')
    ax2.set_ylabel('Classification Accuracy')
    ax2.set_title('Accuracy vs Guidance Scale')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sweep_summary.png'),
                dpi=150, facecolor=fig.get_facecolor())
    plt.close()


if __name__ == "__main__":
    main()
