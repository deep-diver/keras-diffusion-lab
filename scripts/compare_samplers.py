"""Compare DDPM vs DDIM sampling quality and speed.

Generates samples using both samplers from the same initial noise,
computes metrics, and produces comparison grids.

Usage:
    KERAS_BACKEND=jax python scripts/compare_samplers.py \
        --checkpoint artifacts/cfg-run/checkpoints/ema_step30000.weights.h5 \
        --method class_conditional \
        --ddim-steps 50 100 200 500
"""

import argparse
import os
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare DDPM vs DDIM samplers")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", default="class_conditional",
                        choices=["unconditional", "class_conditional"])
    parser.add_argument("--ddim-steps", type=int, nargs="+",
                        default=[50, 100, 200, 500])
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Number of samples to generate per run")
    parser.add_argument("--metrics-dir", default="artifacts/metrics")
    parser.add_argument("--output-dir", default="artifacts/ddim_comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from diffusion_harness.data import load_dataset
    from diffusion_harness.schedules import linear_beta_schedule, compute_schedule
    from diffusion_harness.base.sampling import save_image_grid

    betas = linear_beta_schedule(1000)
    schedule = compute_schedule(betas)

    # Load metrics
    classifier_path = os.path.join(args.metrics_dir, "metrics_classifier.weights.h5")
    stats_path = os.path.join(args.metrics_dir, "metrics_real_stats.npz")
    has_metrics = os.path.exists(classifier_path) and os.path.exists(stats_path)

    if has_metrics:
        from diffusion_harness.metrics.classifier import build_classifier, build_feature_extractor
        from diffusion_harness.metrics.fid import extract_features, compute_fid_from_stats

        images_real, _ = load_dataset("fashion_mnist", return_labels=True)
        classifier = build_classifier()
        classifier.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        classifier.predict(images_real[:1], verbose=0)
        classifier.load_weights(classifier_path)
        feature_extractor = build_feature_extractor(classifier)

        real_stats = np.load(stats_path)
        mu_real, sigma_real = real_stats["mu"], real_stats["sigma"]
        print("Metrics loaded.")
    else:
        print("WARNING: Metrics not prepared. Run scripts/prepare_metrics.py first.")
        print("Continuing without FID evaluation...")

    if args.method == "class_conditional":
        _run_conditional(args, schedule, has_metrics,
                         feature_extractor if has_metrics else None,
                         mu_real if has_metrics else None,
                         sigma_real if has_metrics else None)
    else:
        _run_unconditional(args, schedule)


def _run_conditional(args, schedule, has_metrics,
                     feature_extractor, mu_real, sigma_real):
    """Run comparison for conditional model."""
    from diffusion_harness.methods.class_conditional.models import build_cond_unet
    from diffusion_harness.methods.class_conditional.sampling import CFGSampler
    from diffusion_harness.methods.class_conditional.ddim_sampling import CFGDDIMSampler
    from diffusion_harness.base.sampling import save_image_grid
    from diffusion_harness.metrics.fid import extract_features, compute_fid_from_stats

    model = build_cond_unet(
        image_size=28, channels=1, base_filters=128,
        num_levels=3, channel_multipliers=(1, 2, 2, 2),
        attention_resolutions=(1, 2), num_classes=10
    )
    model.load_weights(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Use same class IDs across all runs
    class_ids = np.arange(10, dtype=np.int32)
    if args.n_samples > 10:
        class_ids = np.repeat(np.arange(10), args.n_samples // 10 + 1)[:args.n_samples]
    elif args.n_samples <= 10:
        class_ids = np.arange(args.n_samples, dtype=np.int32)

    shape = (len(class_ids), 28, 28, 1)
    seed = 42

    results = []

    # DDPM baseline (1000 steps)
    print("\n=== DDPM (1000 steps) ===")
    sampler = CFGSampler(model, schedule, num_timesteps=1000,
                         guidance_scale=3.0, num_classes=10)
    t0 = time.time()
    samples_ddpm = sampler.sample(shape, class_ids=class_ids, seed=seed)
    dt = time.time() - t0
    print(f"  Time: {dt:.1f}s ({dt/len(class_ids):.2f}s/sample)")

    grid_path = os.path.join(args.output_dir, "ddpm_1000steps.png")
    save_image_grid(samples_ddpm, grid_path, nrow=10)
    print(f"  Saved: {grid_path}")

    result_ddpm = {'method': 'DDPM', 'steps': 1000, 'time': dt, 'samples': samples_ddpm}

    if has_metrics:
        feat_gen = extract_features(samples_ddpm, feature_extractor, batch_size=256)
        mu_gen = np.mean(feat_gen, axis=0)
        sigma_gen = np.cov(feat_gen, rowvar=False)
        fid = compute_fid_from_stats(mu_real, sigma_real, mu_gen, sigma_gen)
        result_ddpm['fid'] = fid
        print(f"  FID: {fid:.2f}")

    results.append(result_ddpm)

    # DDIM at various step counts
    for n_steps in args.ddim_steps:
        print(f"\n=== DDIM ({n_steps} steps, eta=0) ===")
        sampler = CFGDDIMSampler(model, schedule, num_timesteps=1000,
                                 guidance_scale=3.0, num_classes=10,
                                 eta=0.0, subsequence_size=n_steps)
        t0 = time.time()
        samples_ddim = sampler.sample(shape, class_ids=class_ids, seed=seed)
        dt = time.time() - t0
        print(f"  Time: {dt:.1f}s ({dt/len(class_ids):.2f}s/sample)")

        grid_path = os.path.join(args.output_dir, f"ddim_{n_steps}steps.png")
        save_image_grid(samples_ddim, grid_path, nrow=10)
        print(f"  Saved: {grid_path}")

        result = {'method': 'DDIM', 'steps': n_steps, 'time': dt,
                  'samples': samples_ddim}

        if has_metrics:
            feat_gen = extract_features(samples_ddim, feature_extractor, batch_size=256)
            mu_gen = np.mean(feat_gen, axis=0)
            sigma_gen = np.cov(feat_gen, rowvar=False)
            fid = compute_fid_from_stats(mu_real, sigma_real, mu_gen, sigma_gen)
            result['fid'] = fid
            print(f"  FID: {fid:.2f}")

        results.append(result)

    # Generate comparison grid
    _make_comparison_grid(results, args.output_dir)

    # Print summary
    print("\n=== Summary ===")
    if has_metrics:
        print(f"{'Method':>8} {'Steps':>6} {'Time':>8} {'FID':>10}")
        print("-" * 34)
        for r in results:
            print(f"{r['method']:>8} {r['steps']:>6d} {r['time']:>7.1f}s "
                  f"{r.get('fid', float('nan')):>10.2f}")
    else:
        print(f"{'Method':>8} {'Steps':>6} {'Time':>8}")
        print("-" * 24)
        for r in results:
            print(f"{r['method']:>8} {r['steps']:>6d} {r['time']:>7.1f}s")

    # Save timing results
    np.savez(
        os.path.join(args.output_dir, "sampler_comparison.npz"),
        methods=[r['method'] for r in results],
        steps=[r['steps'] for r in results],
        times=[r['time'] for r in results],
        fids=[r.get('fid', float('nan')) for r in results],
    )

    print("Done!")


def _run_unconditional(args, schedule):
    """Run comparison for unconditional model."""
    from diffusion_harness.base.models import build_unet
    from diffusion_harness.base.sampling import BaseSampler, save_image_grid
    from diffusion_harness.base.ddim_sampling import DDIMSampler

    model = build_unet(
        image_size=28, channels=1, base_filters=128,
        num_levels=3, channel_multipliers=(1, 2, 2, 2),
        attention_resolutions=(1, 2)
    )
    model.load_weights(args.checkpoint)

    shape = (args.n_samples, 28, 28, 1)
    seed = 42

    print("\n=== DDPM (1000 steps) ===")
    sampler = BaseSampler(model, schedule, num_timesteps=1000)
    t0 = time.time()
    samples = sampler.sample(shape, seed=seed)
    dt = time.time() - t0
    print(f"  Time: {dt:.1f}s")
    save_image_grid(samples, os.path.join(args.output_dir, "ddpm_1000steps.png"))

    for n_steps in args.ddim_steps:
        print(f"\n=== DDIM ({n_steps} steps) ===")
        sampler = DDIMSampler(model, schedule, num_timesteps=1000,
                              eta=0.0, subsequence_size=n_steps)
        t0 = time.time()
        samples = sampler.sample(shape, seed=seed)
        dt = time.time() - t0
        print(f"  Time: {dt:.1f}s")
        save_image_grid(samples,
                        os.path.join(args.output_dir, f"ddim_{n_steps}steps.png"))


def _make_comparison_grid(results, output_dir):
    """Generate side-by-side comparison grid for all methods."""
    from diffusion_harness.base.sampling import save_image_grid

    # Take first 8 samples from each method
    comparison = []
    for r in results:
        comparison.append(r['samples'][:8])
    all_images = np.concatenate(comparison, axis=0)

    grid_path = os.path.join(output_dir, "comparison_grid.png")
    save_image_grid(all_images, grid_path, nrow=8)
    print(f"\nComparison grid saved to {grid_path}")


if __name__ == "__main__":
    main()
