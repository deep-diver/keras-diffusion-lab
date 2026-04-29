# Repository Migration Summary

**Date:** 2026-04-22
**From:** keras-diffusion (DDPM/DDIM one-off project)
**To:** diffusion-harness (extensible research harness)

---

## What Was Removed

### Source code
- `src/keras_diffusion/` (entire old package) — replaced by `src/diffusion_harness/`
- `src/keras_diffusion/improved/` — legacy Improved DDPM (known divergences)
- `src/keras_diffusion/improved_canonical/` — canonical Improved DDPM variant
- `src/keras_diffusion/backbones/` — DDPM-specific U-Net with `tiny_unet` compat wrapper
- `src/keras_diffusion/config/` — DDPM-specific configs (ddpm_config, tiny_config, production_config)
- `src/keras_diffusion/paths/` — DDPM forward/reverse diffusion path functions
- `src/keras_diffusion/targets/` — epsilon_target function
- `src/keras_diffusion/trainers/` — DDPMTrainer class
- `src/keras_diffusion/sampling/ddim.py` — DDIM sampler
- All DDPM/DDIM-specific scripts (train_ddpm, sample_ddpm, sample_compare, visualize_trajectory, etc.)

### Top-level scripts
- `remote_train_canonical.py` — canonical Improved DDPM Kinetic job
- `remote_train_improved.py` — legacy Improved DDPM Kinetic job

### Artifacts
- `artifacts/checkpoints/` — 169MB of DDPM/DDIM checkpoints
- `artifacts/ddim_comparison/` — DDIM comparison outputs (grids, GIFs, timing)
- `artifacts/comparison/` — plain vs improved comparison outputs
- `artifacts/*.gif`, `artifacts/*.png` — training evolution animations and loss curves
- `artifacts_run02/` — second run artifacts (snapshots, trajectories, checkpoints)
- All DDPM/DDIM milestone reports (6 report files)

### Tests
- All method-specific tests (test_backbones, test_trainer, test_paths, test_targets,
  test_sampling, test_improved_canonical, test_integration)

## What Was Preserved

### Knowledge documentation
- `artifacts/reports/keras_kinetic_tpu_field_notes.md` — 12 sections of TPU/Kinetic operational knowledge
- `artifacts/reports/engineering_tips.md` — 27 actionable tips
- `artifacts/reports/diffusion_research_axes_2026.md` — Research landscape survey with prioritized directions

### Reusable code patterns (rewritten, not copied)
- **GCS utilities** (`utils/gcs.py`): upload/download for files, bytes, JSON, numpy, list_blobs,
  find_latest_checkpoint. Model-agnostic.
- **EventLog** (`monitoring/__init__.py`): JSONL structured logging. Generalized from DDPM-specific
  to model-agnostic (removed x0_hat, logvar fields as required; they remain as optional kwargs).
- **Noise schedules** (`schedules/__init__.py`): linear_beta_schedule, cosine_beta_schedule, compute_schedule.
  These are standard DDPM infrastructure, preserved exactly.
- **Image utilities** (`sampling/__init__.py`): save_image_grid, save_annotated_grid. Model-agnostic.
- **Data loading** (`data/__init__.py`): Generalized from CIFAR-10 only to support cifar10/fashion_mnist/mnist.
  Added get_dataset_info() for automatic architecture sizing.

## What Was Generalized

| Old (DDPM-specific) | New (generalized) |
|---------------------|-------------------|
| `keras_diffusion` package name | `diffusion_harness` |
| `ddpm_unet()` function | `build_unet()` with configurable parameters |
| `DDPMTrainer` class | `DiffusionTrainer` class |
| `ddpm_config()`, `production_config()`, `tiny_config()` | Single `make_config()` with defaults |
| `load_cifar10()` only | `load_dataset(name)` supporting 3 datasets |
| 3 remote_train scripts | Single `remote_train.py` with `--dataset` flag |
| EMA + checkpoint baked into trainer | Same, but documented as reusable pattern |
| EventLog with DDPM-specific fields | EventLog with optional extra fields |
| GCS `upload_checkpoint_set` hardcoded | Generic GCS primitives + checkpoint naming convention |

## Why These Decisions

1. **Package rename**: `keras_diffusion` tied the name to one method. `diffusion_harness` reflects the research purpose.

2. **Single config builder**: The old code had 3 config functions (ddpm_config, production_config, tiny_config) that differed in subtle ways. One builder with explicit parameters is clearer and less error-prone.

3. **Dataset abstraction**: Starting with Fashion-MNIST for fast iteration, but the code handles CIFAR-10 identically via `--dataset cifar10`. No rewrite needed to scale up.

4. **Methods directory**: Empty placeholder directories for `pruning/` and `distillation/` signal where future work goes without premature implementation.

5. **Preserved Kinetic patterns**: The GCS persistence, staged training, resume logic, and snapshot generation patterns from the old `remote_train.py` were preserved because they were battle-tested over 10+ TPU training jobs and directly apply to any diffusion training.
