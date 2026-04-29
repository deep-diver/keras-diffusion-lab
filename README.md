# Diffusion Research Harness

A clean, extensible diffusion model research framework built on **Keras 3 + JAX + Keras Kinetic** for TPU training.

Standard DDPM baseline with plug-in architecture for research methods (class-conditional generation, pruning, distillation, scaling studies, data efficiency experiments).

## Results

Two trained models on Fashion-MNIST:

| Model | Steps | Params | Method | Key Result |
|-------|-------|--------|--------|------------|
| Unconditional DDPM | 15,000 | ~20M | Epsilon-prediction DDPM | Recognizable fashion items, loss=0.038 |
| Class-Conditional CFG | 14,900 | ~21M | Classifier-Free Guidance | 10-class control with guidance scale w=3.0 |

See the detailed blog posts:
- [Training a DDPM on Fashion-MNIST](artifacts/reports/fashion-mnist-diffusion-2026-04-23/fashion_mnist_diffusion.md) — unconditional baseline walkthrough
- [Class-Conditional Generation with CFG](artifacts/reports/class-conditional-fashionmnist-2026-04-23/class_conditional_diffusion.md) — from "generate anything" to "generate a sneaker"

## Setup

```bash
pip install -e ".[dev]"
```

Requires Python 3.12+, Keras 3, JAX. Optional: `Pillow` for image output, `google-cloud-storage` for GCS persistence.

## Quick Start

### Run tests (~41 tests)

```bash
KERAS_BACKEND=jax pytest tests/ -v
```

### Local training

```bash
# Unconditional
KERAS_BACKEND=jax python remote_train.py --dataset fashion_mnist --steps 500 --batch-size 32

# Class-conditional (CFG)
KERAS_BACKEND=jax python remote_train.py --dataset fashion_mnist --method class_conditional --steps 500 --batch-size 32
```

### TPU training (Kinetic)

```bash
# Provision Spot TPU pool (91% cheaper)
kinetic pool add --accelerator v5litepod-4 --spot --project YOUR_PROJECT --zone us-west4-a

# Submit training job
KERAS_BACKEND=jax KERAS_REMOTE_PROJECT=YOUR_PROJECT python remote_train.py \
    --gcs-bucket gs://YOUR_BUCKET/runs/run01 \
    --zone us-west4-a --dataset fashion_mnist --stage warmup

# Class-conditional training
KERAS_BACKEND=jax KERAS_REMOTE_PROJECT=YOUR_PROJECT python remote_train.py \
    --gcs-bucket gs://YOUR_BUCKET/cfg-run01 \
    --zone us-west4-a --dataset fashion_mnist --method class_conditional --stage warmup

# Resume from checkpoint
KERAS_BACKEND=jax KERAS_REMOTE_PROJECT=YOUR_PROJECT python remote_train.py \
    --gcs-bucket gs://YOUR_BUCKET/runs/run01 \
    --zone us-west4-a --stage early --resume

# Download artifacts
python remote_train.py --gcs-bucket gs://YOUR_BUCKET/runs/run01 --download-only
```

### Generate visualizations

```bash
# Denoising GIF (10-class, requires EMA checkpoint)
KERAS_BACKEND=jax python scripts/denoising_gif.py \
    --checkpoint artifacts/cfg-run/checkpoints/ema_step14800.weights.h5 \
    --output denoising.gif --guidance-scale 3.0

# All report images (loss curve, architecture, milestones, etc.)
python scripts/generate_report_images.py
```

## What's Implemented

### Training Methods

| Method | Location | Description |
|--------|----------|-------------|
| **Unconditional DDPM** | `methods/unconditional/` | Standard epsilon-prediction (Ho et al. 2020) |
| **Class-Conditional CFG** | `methods/class_conditional/` | Classifier-Free Guidance (Ho & Salimans 2022) |
| Pruning | `methods/pruning/` | (TODO) |
| Distillation | `methods/distillation/` | (TODO) |

Adding a new method: create `methods/<name>/` with `build_model()`, `build_trainer()`, `build_sampler()`, then register in `methods/__init__.py`.

### Core Features

- **Configurable U-Net**: depth, width, attention placement, optional class embedding
- **Linear and cosine noise schedules**
- **EMA** (exponential moving average) for stable sample quality
- **Full reverse DDPM sampling** with CFG guided prediction
- **Checkpointing** (model + EMA + optimizer state) via GCS
- **Resume support** for chained training across TPU job timeouts
- **Structured monitoring** (JSONL event log)
- **Method registry** (`get_method("class_conditional")`) for extensibility

### Multi-Dataset Support

| Dataset | Size | Channels | Use case |
|---------|------|----------|----------|
| Fashion-MNIST | 28x28 | 1 (gray) | Fast iteration, debugging, 10 classes |
| MNIST | 28x28 | 1 (gray) | Simplest test, 10 classes |
| CIFAR-10 | 32x32 | 3 (RGB) | Standard benchmark, 10 classes |

## Architecture

```
src/diffusion_harness/
  base/              — Shared base classes
    models.py        — build_unet(), ResBlock, SelfAttention, FiLM conditioning
    training.py      — BaseTrainer: EMA, checkpointing, train loop (template method)
    sampling.py      — BaseSampler: reverse diffusion, save_image_grid
  core/              — make_config() builder (method, num_classes, guidance_scale)
  schedules/         — linear/cosine beta schedules, compute_schedule()
  models/            — Re-exports from base (backward compat)
  training/          — Re-exports UnconditionalTrainer as DiffusionTrainer
  sampling/          — Re-exports from base + methods
  data/              — load_dataset (return_labels), make_dataset_with_labels
  monitoring/        — EventLog (JSONL structured logging)
  utils/             — GCS helpers (upload/download/checkpoint finding)
  methods/
    __init__.py      — get_method(name) registry, list_methods()
    unconditional/
      training.py    — UnconditionalTrainer(BaseTrainer): epsilon-MSE loss
      sampling.py    — unconditional_sample()
      config.py      — Default hyperparams
    class_conditional/
      models.py      — ClassEmbedding, build_cond_unet() (3 inputs)
      training.py    — CFGTrainer(BaseTrainer): class dropout + epsilon-MSE
      sampling.py    — CFGSampler: guided prediction (1+w)*cond - w*uncond
      config.py      — CFG defaults (guidance_scale=3.0, dropout=0.1)
    pruning/         — (TODO) pruning research
    distillation/    — (TODO) knowledge distillation
scripts/
  denoising_gif.py            — 10-class denoising animation generator
  generate_report_images.py   — Report image generator
  generate_keras_diagrams.py  — Keras plot_model() architecture diagrams
tests/                        — ~41 tests (31 unconditional + 10 CFG)
remote_train.py               — Main entry point (local + TPU, --method flag)
decisions/                    — ADR-style decision records (6 decisions)
experiments/                  — Experiment cards (2 completed)
```

## Documentation

| Document | Location | Description |
|----------|----------|-------------|
| Unconditional DDPM blog post | `artifacts/reports/fashion-mnist-diffusion-2026-04-23/` | Full walkthrough: architecture, training, troubleshooting, critical analysis |
| CFG blog post | `artifacts/reports/class-conditional-fashionmnist-2026-04-23/` | Repo restructure, conditional U-Net, guided sampling, denoising GIF |
| Research survey | `artifacts/reports/diffusion_research_axes_2026.md` | Research directions and method comparisons |
| TPU field notes | `artifacts/reports/keras_kinetic_tpu_field_notes.md` | Kinetic/TPU operational knowledge |
| Engineering tips | `artifacts/reports/engineering_tips.md` | Practical development notes |

## Decision Records

| Decision | Choice | Why |
|----------|--------|-----|
| [001: EMA decay](decisions/001-ema-decay-tuning.md) | 0.999 (not 0.9999) | 0.9999 designed for 800K steps; too slow for 10K-50K runs |
| [002: U-Net levels](decisions/002-unet-3-levels-for-28x28.md) | 3 levels for 28x28 | 4 levels produces odd spatial dims (28→14→7→3) |
| [003: CFG](decisions/003-class-conditional-cfg.md) | Classifier-Free Guidance | Single model, no external classifier, industry standard |
| [004: Guidance scale](decisions/004-guidance-scale-choice.md) | w=3.0 without ablation | Moderate guidance, not validated with sweep |
| [005: Chained training](decisions/005-chained-resume-training.md) | Resume jobs for Kinetic timeout | 7 chained jobs to reach 15K steps; quality regression at boundaries |
| [006: Attention placement](decisions/006-attention-placement-inconsistency.md) | Known confound | Unconditional uses level 0, conditional uses levels 1,2 |
| [003: CFG](decisions/003-class-conditional-cfg.md) | Classifier-Free Guidance | Single model, no external classifier, industry standard |

## Key Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| Template method | `BaseTrainer.train()` calls `train_step()` | Subclasses implement training logic; loop/EMA/checkpointing shared |
| Registry | `get_method("class_conditional")` | New methods plug in by creating a package + registering |
| Inheritance | `CFGTrainer(BaseTrainer)` | Small codebase, clear override points |
| Backward compat | Original modules are thin re-exports | Existing tests pass unchanged |

## Baseline Recipe

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Timesteps | 1000 | Standard DDPM |
| Schedule | Linear (1e-4 to 0.02) | Matches Ho et al. 2020 |
| U-Net levels | 3 (28x28) / 4 (32x32) | Clean spatial halving |
| Base filters | 128 | ~20M params, fits TPU v5litepod-4 |
| EMA decay | 0.999 | 1000-step window for 10K-50K runs |
| Optimizer | Adam, lr=2e-4 | Standard |
| CFG guidance scale | 3.0 | Moderate, good class adherence |
| CFG class dropout | 0.1 | Standard from Ho & Salimans 2022 |

## References

1. Ho, Jain & Abbeel (2020). "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
2. Ho & Salimans (2022). "Classifier-Free Diffusion Guidance." NeurIPS 2021 Workshop.
3. Nichol & Dhariwal (2021). "Improved Denoising Diffusion Probabilistic Models." ICML 2021.
4. Perez et al. (2018). "FiLM: Visual Reasoning with a General Conditioning Layer." AAAI 2018.
5. Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022.
