"""Remote diffusion training via Keras Kinetic with GCS artifact persistence.

Features:
  - Standard DDPM baseline (configurable architecture)
  - GCS-based checkpoint persistence (survives preemption and job timeout)
  - Resume from GCS checkpoint (model + EMA + optimizer state)
  - Structured event logging (JSONL) for remote monitoring
  - Fixed-seed snapshot sampling with EMA model
  - Staged training with evidence-based continuation decisions
  - Dataset-agnostic: supports CIFAR-10, Fashion-MNIST, MNIST

Usage:
    # Provision Spot TPU pool
    kinetic pool add --accelerator v5litepod-4 --spot \\
        --project YOUR_PROJECT --zone us-west4-a

    # Quick local test (Fashion-MNIST)
    KERAS_BACKEND=jax python remote_train.py \\
        --dataset fashion_mnist --steps 100 --batch-size 32

    # Remote TPU training
    KERAS_BACKEND=jax KERAS_REMOTE_PROJECT=YOUR_PROJECT python remote_train.py \\
        --gcs-bucket gs://YOUR_BUCKET/runs/run01 \\
        --zone us-west4-a --stage warmup

    # Resume from latest checkpoint
    KERAS_BACKEND=jax KERAS_REMOTE_PROJECT=YOUR_PROJECT python remote_train.py \\
        --gcs-bucket gs://YOUR_BUCKET/runs/run01 \\
        --zone us-west4-a --steps 2000 --resume

    # Download artifacts
    python remote_train.py --gcs-bucket gs://YOUR_BUCKET/runs/run01 \\
        --download-only --local-dir artifacts
"""

import argparse
import os

os.environ["KERAS_BACKEND"] = "jax"

STAGES = {
    "warmup": {
        "steps": 1000,
        "checkpoint_every": 200,
        "sample_every": 200,
        "description": "Initial warmup — validate architecture and loss convergence",
    },
    "early": {
        "steps": 4000,
        "checkpoint_every": 500,
        "sample_every": 250,
        "description": "Early training — loss should be decreasing",
    },
    "mid": {
        "steps": 15000,
        "checkpoint_every": 1000,
        "sample_every": 500,
        "description": "Mid training — structure should be emerging",
    },
    "late": {
        "steps": 80000,
        "checkpoint_every": 2000,
        "sample_every": 1000,
        "description": "Late training — objects should become recognizable",
    },
    "extended": {
        "steps": 700000,
        "checkpoint_every": 5000,
        "sample_every": 2500,
        "description": "Extended training — approaching convergence",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Remote diffusion training via Keras Kinetic")
    # GCS
    parser.add_argument("--gcs-bucket", default=None,
                        help="GCS path for artifacts (required for remote)")
    parser.add_argument("--download-only", action="store_true",
                        help="Download artifacts from GCS instead of training")
    parser.add_argument("--local-dir", default="artifacts",
                        help="Local directory for --download-only")
    # Kinetic
    parser.add_argument("--project", default=os.environ.get("KERAS_REMOTE_PROJECT", "gcp-ml-172005"))
    parser.add_argument("--zone", default="us-west4-a")
    parser.add_argument("--accelerator", default="v5litepod-4")
    parser.add_argument("--container-image", default=None)
    parser.add_argument("--no-spot", action="store_true")
    # Dataset
    parser.add_argument("--dataset", default="fashion_mnist",
                        choices=["cifar10", "fashion_mnist", "mnist"],
                        help="Dataset to train on")
    # Architecture
    parser.add_argument("--base-filters", type=int, default=128)
    parser.add_argument("--num-levels", type=int, default=3)
    # Diffusion
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--schedule-type", default="linear", choices=["linear", "cosine"])
    # Training
    parser.add_argument("--stage", choices=list(STAGES.keys()), default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=None)
    parser.add_argument("--snapshot-seed", type=int, default=123)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--method", default="unconditional",
                        choices=["unconditional", "class_conditional"],
                        help="Training method (default: unconditional)")
    args = parser.parse_args()

    # Resolve stage
    if args.stage:
        sc = STAGES[args.stage]
        if args.steps is None:
            args.steps = sc["steps"]
        if args.checkpoint_every is None:
            args.checkpoint_every = sc["checkpoint_every"]
        if args.sample_every is None:
            args.sample_every = sc["sample_every"]
    else:
        if args.steps is None:
            args.steps = 500
        if args.checkpoint_every is None:
            args.checkpoint_every = 100
        if args.sample_every is None:
            args.sample_every = 100

    # Download-only mode
    if args.download_only:
        _download_artifacts(args)
        return

    # Local or remote
    if args.gcs_bucket:
        _run_remote(args)
    else:
        _run_local(args)


def _run_local(args):
    """Run training locally (no TPU, no GCS)."""
    from diffusion_harness.core import make_config
    from diffusion_harness.data import load_dataset, make_dataset, make_dataset_with_labels
    from diffusion_harness.methods import get_method
    from diffusion_harness.sampling import ddpm_sample, save_image_grid
    from diffusion_harness.monitoring import EventLog
    import jax
    import numpy as np

    method = get_method(args.method)

    config = make_config(
        dataset=args.dataset,
        method=args.method,
        base_filters=args.base_filters,
        num_levels=args.num_levels,
        num_timesteps=args.num_timesteps,
        schedule_type=args.schedule_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_steps=args.steps,
        checkpoint_every=args.checkpoint_every,
        sample_every=args.sample_every,
        num_samples=args.num_samples,
        snapshot_seed=args.snapshot_seed,
    )
    if args.no_ema:
        config["ema_decay"] = 0.0

    print(f"=== Local Training ===")
    print(f"  Method: {args.method}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")

    # Load data — conditional methods need labels
    if args.method == "class_conditional":
        images, labels = load_dataset(args.dataset, subset_size=args.subset,
                                      seed=42, return_labels=True)
        print(f"  Dataset shape: {images.shape}, labels: {labels.shape}")
        data_iter = make_dataset_with_labels(images, labels, config["batch_size"],
                                             shuffle=True, seed=42)
    else:
        images = load_dataset(args.dataset, subset_size=args.subset, seed=42)
        print(f"  Dataset shape: {images.shape}")
        data_iter = make_dataset(images, config["batch_size"], shuffle=True, seed=42)

    trainer = method.build_trainer(config)
    print(f"  Model params: {trainer.model.count_params():,}")

    event_log = EventLog()

    out_dir = "artifacts/local_run"
    os.makedirs(out_dir, exist_ok=True)

    def sample_fn(step, model):
        snap_model = trainer.get_ema_model() if trainer.ema_weights else model
        shape = (config["num_samples"], config["image_size"],
                 config["image_size"], config["image_channels"])
        rng = jax.random.PRNGKey(config["snapshot_seed"])
        _, key = jax.random.split(rng)
        noise = jax.random.normal(key, shape)

        if args.method == "class_conditional":
            from diffusion_harness.methods.class_conditional.sampling import cfg_sample
            # Sample a mix of classes for visualization
            class_ids = np.array([i % config["num_classes"] for i in range(config["num_samples"])], dtype="int32")
            samples = cfg_sample(snap_model, config["schedule"],
                                 config["num_timesteps"], shape,
                                 class_ids=class_ids,
                                 guidance_scale=config["guidance_scale"],
                                 num_classes=config["num_classes"],
                                 seed=config["snapshot_seed"],
                                 initial_noise=np.array(noise))
        else:
            samples = ddpm_sample(snap_model, config["schedule"],
                                  config["num_timesteps"], shape,
                                  seed=config["snapshot_seed"],
                                  initial_noise=np.array(noise))

        if trainer.ema_weights:
            trainer.restore_training_weights()
        path = os.path.join(out_dir, f"samples_step{step:06d}.png")
        save_image_grid(samples, path, nrow=4)
        print(f"    Snapshot saved: {path}")

    loss_history = trainer.train(
        data_iter, args.steps,
        checkpoint_dir=os.path.join(out_dir, "checkpoints"),
        sample_fn=sample_fn,
        event_log=event_log,
    )

    print(f"\n=== Training Complete ===")
    print(f"  Final loss: {loss_history[-1]:.4f}")


def _download_artifacts(args):
    """Download all training artifacts from GCS."""
    from diffusion_harness.utils.gcs import (
        download_file, download_json, download_numpy, download_bytes, list_blobs,
    )
    import json as _json
    import numpy as _np

    gcs = args.gcs_bucket
    local = args.local_dir
    os.makedirs(local, exist_ok=True)

    print(f"=== Downloading {gcs} -> {local} ===")

    # Manifest
    manifest = download_json(f"{gcs}/snapshots/manifest.json")
    if manifest:
        path = os.path.join(local, "snapshots", "manifest.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            _json.dump(manifest, f, indent=2)
        print(f"  Manifest: {len(manifest)} entries")

    # Events
    raw = download_bytes(f"{gcs}/logs/events.jsonl")
    if raw:
        path = os.path.join(local, "logs", "events.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(raw.decode("utf-8"))
        print(f"  Events: {len(raw.decode().strip().splitlines())} lines")

    # Loss history
    loss = download_numpy(f"{gcs}/logs/loss_history.npy")
    if loss is not None:
        path = os.path.join(local, "logs", "loss_history.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _np.save(path, loss)
        print(f"  Loss history: {len(loss)} entries")

    # Checkpoints
    ckpts = list_blobs(f"{gcs}/checkpoints", pattern=".weights.h5")
    for bp in ckpts:
        name = bp.split("/")[-1]
        download_file(bp, os.path.join(local, "checkpoints", name))
    print(f"  Checkpoints: {len(ckpts)} files")

    # Snapshots
    snaps = [b for b in list_blobs(f"{gcs}/snapshots") if ".npy" in b or ".png" in b]
    for bp in snaps:
        name = bp.split("/")[-1]
        download_file(bp, os.path.join(local, "snapshots", name))
    print(f"  Snapshots: {len(snaps)} files")

    print("=== Download complete ===")


def _run_remote(args):
    """Submit remote training job via Keras Kinetic."""
    import kinetic

    src_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    src_data = kinetic.Data(src_abs)

    # Capture args for closure
    gcs_bucket = args.gcs_bucket
    project = args.project
    zone = args.zone
    accelerator = args.accelerator
    container_image = args.container_image
    steps = args.steps
    batch_size = args.batch_size
    base_filters = args.base_filters
    num_levels = args.num_levels
    num_timesteps = args.num_timesteps
    schedule_type = args.schedule_type
    dataset_name = args.dataset
    checkpoint_every = args.checkpoint_every
    sample_every = args.sample_every
    snapshot_seed = args.snapshot_seed
    num_samples = args.num_samples
    subset = args.subset
    do_resume = args.resume
    resume_from = args.resume_from
    no_ema = args.no_ema
    stage_name = args.stage
    method_name = args.method

    @kinetic.run(
        accelerator=accelerator,
        project=project,
        zone=zone,
        volumes={"/tmp/diffusion_harness_src": src_data},
        container_image=container_image,
    )
    def remote_train():
        import sys
        os.environ["KERAS_BACKEND"] = "jax"
        src_mount = "/tmp/diffusion_harness_src"
        if os.path.isdir(src_mount) and src_mount not in sys.path:
            sys.path.insert(0, src_mount)

        import json, time
        import numpy as np
        import jax
        import keras

        from diffusion_harness.core import make_config
        from diffusion_harness.data import load_dataset, make_dataset, make_dataset_with_labels
        from diffusion_harness.sampling import ddpm_sample, save_image_grid, save_annotated_grid
        from diffusion_harness.monitoring import EventLog, parse_jsonl
        from diffusion_harness.utils.gcs import (
            upload_file, upload_json, upload_numpy, upload_bytes,
            download_file, download_json, download_bytes, list_blobs,
            find_latest_checkpoint,
        )

        devices = jax.devices()
        gcs = gcs_bucket
        gcs_ckpt = f"{gcs}/checkpoints"
        gcs_snap = f"{gcs}/snapshots"
        gcs_logs = f"{gcs}/logs"

        print(f"=== Remote Training ===")
        print(f"  Devices: {[str(d) for d in devices]}")
        print(f"  Dataset: {dataset_name}")
        print(f"  Steps: {steps}")
        if stage_name:
            print(f"  Stage: {stage_name}")

        # Config
        config = make_config(
            dataset=dataset_name,
            method=method_name,
            base_filters=base_filters,
            num_levels=num_levels,
            num_timesteps=num_timesteps,
            schedule_type=schedule_type,
            batch_size=batch_size,
            num_train_steps=steps,
            checkpoint_every=checkpoint_every,
            sample_every=sample_every,
            num_samples=num_samples,
            snapshot_seed=snapshot_seed,
        )
        if no_ema:
            config["ema_decay"] = 0.0

        # Data
        if method_name == "class_conditional":
            images, labels = load_dataset(dataset_name, subset_size=subset,
                                          seed=42, return_labels=True)
            print(f"  Dataset: {images.shape}, labels: {labels.shape}")
            data_iter = make_dataset_with_labels(images, labels, config["batch_size"],
                                                 shuffle=True, seed=42)
        else:
            images = load_dataset(dataset_name, subset_size=subset, seed=42)
            print(f"  Dataset: {images.shape}")
            data_iter = make_dataset(images, config["batch_size"], shuffle=True, seed=42)

        # Trainer
        from diffusion_harness.methods import get_method
        method_module = get_method(method_name)
        trainer = method_module.build_trainer(config)
        print(f"  Model params: {trainer.model.count_params():,}")

        # Resume
        start_step = 0
        if do_resume or resume_from:
            ckpt_gcs = resume_from or find_latest_checkpoint(gcs_ckpt)
            if ckpt_gcs:
                print(f"  Resuming from: {ckpt_gcs}")
                orig_name = ckpt_gcs.split("/")[-1]
                local_ckpt = os.path.join("/tmp", orig_name)
                if download_file(ckpt_gcs, local_ckpt):
                    step_str = orig_name.split("step")[1].split(".")[0]
                    for suffix in [f"optimizer_step{step_str}.npz",
                                   f"ema_step{step_str}.weights.h5"]:
                        download_file(f"{gcs_ckpt}/{suffix}", os.path.join("/tmp", suffix))
                    trainer.load_checkpoint(local_ckpt)
                    start_step = trainer.step
                    print(f"  Resumed at step {start_step}")

        # Event log
        event_log = EventLog()
        existing_raw = download_bytes(f"{gcs_logs}/events.jsonl")
        if existing_raw:
            event_log.events = parse_jsonl(existing_raw.decode("utf-8"))
            print(f"  Loaded {len(event_log.events)} existing events")

        # Manifest
        manifest = download_json(f"{gcs_snap}/manifest.json") or []

        # Loss history
        import io as _io
        raw_loss = download_bytes(f"{gcs_logs}/loss_history.npy")
        if raw_loss:
            trainer.loss_history = np.load(_io.BytesIO(raw_loss)).tolist()
            print(f"  Loaded {len(trainer.loss_history)} loss entries")

        # Fixed noise for snapshots
        snap_shape = (num_samples, config["image_size"],
                      config["image_size"], config["image_channels"])
        rng = jax.random.PRNGKey(snapshot_seed)
        _, key = jax.random.split(rng)
        fixed_noise = np.array(jax.random.normal(key, snap_shape).astype("float32"))

        # Temp dirs
        for d in ["/tmp/checkpoints", "/tmp/snapshots"]:
            os.makedirs(d, exist_ok=True)

        # Snapshot callback
        def generate_snapshot(step, model):
            loss_val = trainer.loss_history[-1] if trainer.loss_history else 0.0
            print(f"  Snapshot at step {step} (loss={loss_val:.4f})...")

            snap_model = trainer.get_ema_model() if trainer.ema_weights else model

            if method_name == "class_conditional":
                from diffusion_harness.methods.class_conditional.sampling import cfg_sample
                class_ids = np.array([i % config["num_classes"] for i in range(config["num_samples"])], dtype="int32")
                samples = cfg_sample(
                    snap_model, config["schedule"], config["num_timesteps"],
                    snap_shape, class_ids=class_ids,
                    guidance_scale=config["guidance_scale"],
                    num_classes=config["num_classes"],
                    seed=snapshot_seed, initial_noise=fixed_noise,
                )
            else:
                samples = ddpm_sample(
                    snap_model, config["schedule"], config["num_timesteps"],
                    snap_shape, seed=snapshot_seed, initial_noise=fixed_noise,
                )
            if trainer.ema_weights:
                trainer.restore_training_weights()

            # Save numpy
            npy_name = f"samples_step{step:06d}.npy"
            np.save(f"/tmp/snapshots/{npy_name}", samples)
            upload_file(f"/tmp/snapshots/{npy_name}", f"{gcs_snap}/{npy_name}")

            # Save PNG
            try:
                png_name = f"samples_step{step:06d}.png"
                png_path = f"/tmp/snapshots/{png_name}"
                save_annotated_grid(samples, png_path, step=step, loss=loss_val)
                if os.path.exists(png_path):
                    upload_file(png_path, f"{gcs_snap}/{png_name}")
            except Exception:
                pass

            manifest.append({"step": step, "loss": loss_val,
                             "path": f"{gcs_snap}/{npy_name}"})
            upload_json(manifest, f"{gcs_snap}/manifest.json")
            event_log.log_snapshot(step=step, gcs_path=f"{gcs_snap}/{npy_name}", loss=loss_val)

        # Checkpoint callback
        original_save = trainer.save_checkpoint
        def save_and_upload(directory, step):
            original_save(directory, step)
            for suffix in [f"model_step{step}.weights.h5",
                          f"ema_step{step}.weights.h5",
                          f"optimizer_step{step}.npz",
                          f"state_step{step}.json"]:
                local_path = os.path.join(directory, suffix)
                if os.path.exists(local_path):
                    upload_file(local_path, f"{gcs_ckpt}/{suffix}")
            try:
                event_log.upload(f"{gcs_logs}/events.jsonl")
                upload_numpy(np.array(trainer.loss_history), f"{gcs_logs}/loss_history.npy")
            except Exception as e:
                print(f"    Warning: event upload failed: {e}")
        trainer.save_checkpoint = save_and_upload

        # Train
        print(f"\n=== Training ({steps} steps from step {start_step}) ===")
        start_time = time.time()
        trainer.train(
            data_iter, steps,
            checkpoint_dir="/tmp/checkpoints",
            sample_fn=generate_snapshot,
            event_log=event_log,
        )
        elapsed = time.time() - start_time

        # Final snapshot
        generate_snapshot(trainer.step, trainer.model)

        # Final upload
        event_log.upload(f"{gcs_logs}/events.jsonl")
        upload_numpy(np.array(trainer.loss_history), f"{gcs_logs}/loss_history.npy")

        result = {
            "steps_completed": steps,
            "total_steps": trainer.step,
            "final_loss": trainer.loss_history[-1] if trainer.loss_history else None,
            "elapsed_seconds": elapsed,
            "num_params": int(trainer.model.count_params()),
            "dataset": dataset_name,
            "gcs_bucket": gcs,
            "stage": stage_name,
        }

        print(f"\n=== Complete ===")
        for k, v in result.items():
            print(f"  {k}: {v}")

        return result

    print(f"\n=== Submitting job ===")
    print(f"  Accelerator: {accelerator}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Steps: {steps}")
    if stage_name:
        print(f"  Stage: {stage_name} — {STAGES[stage_name]['description']}")
    result = remote_train()
    print(f"\n=== Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
