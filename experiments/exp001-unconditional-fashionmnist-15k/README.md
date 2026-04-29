# Experiment 001: Unconditional Fashion-MNIST Baseline (15K Steps)

## Status
Completed

## Date
2026-04-23

## Method
unconditional

## Objective
Establish a working DDPM baseline on Fashion-MNIST. Verify the harness pipeline end-to-end: training, EMA, checkpointing, sampling, and visual quality.

## Configuration
```yaml
dataset: fashion_mnist
method: unconditional
base_filters: 128
num_levels: 3
channel_multipliers: (1, 2, 2)
attention_resolutions: (0,)
num_timesteps: 1000
batch_size: 64
learning_rate: 0.0002
ema_decay: 0.999
steps: 15000
schedule: linear
beta_start: 0.0001
beta_end: 0.02
```

## Results
- Final loss: 0.0361
- FID: Not computed
- Visual quality: Clear, recognizable Fashion-MNIST items (shirts, sneakers, bags, trousers). Some class blending at boundaries. Good global coherence but fine details (e.g., text on shirts) absent.
- Key observations:
  - EMA decay 0.999 (not 0.9999) was critical for 15K-step training. At 0.9999, EMA weights were still ~60% random initialization at 5K steps.
  - 3-level U-Net required for 28x28 (4 levels causes odd-dimension shape mismatch: 28→14→7→3).
  - Loss plateaued around 0.035-0.04 after ~10K steps.
  - ~21M parameters with base_filters=128.

## Artifacts
- GCS checkpoint: `gs://gcp-ml-172005-ddpm-training/harness_baseline/run02/`
- Local samples: `artifacts/harness_baseline/run02/snapshots/` (downloaded from GCS)
- Training evolution GIF: `artifacts/harness_baseline/run02/training_evolution.gif` (39 frames)

## Decisions
- EMA decay tuned to 0.999 for short runs (Decision 001).
- 3-level U-Net adopted for 28x28 images (Decision 002).
- 15K steps sufficient for recognizable results but not converged. Longer training (50-100K) likely needed for sharp details.

## Follow-ups
- [x] Decision 001: EMA decay tuning
- [x] Decision 002: 3-level U-Net for 28x28
- [x] Experiment 002: Class-conditional CFG on Fashion-MNIST
- [ ] Compute FID for quantitative baseline
- [ ] Longer training run (50K+ steps) to see quality ceiling
