# Experiment 002: Class-Conditional CFG on Fashion-MNIST (15K Steps)

## Status
Completed

## Date
2026-04-23 – 2026-04-29

## Method
class_conditional (Classifier-Free Guidance)

## Objective
Add class-conditional generation to the diffusion harness. Given a class label (e.g., "sneaker"), generate an image of that class. Validate the multi-method harness architecture (base classes + method registry) with a real second method. Evaluate CFG quality at 15K steps on Fashion-MNIST.

## Configuration
```yaml
dataset: fashion_mnist
method: class_conditional
base_filters: 128
num_levels: 3
channel_multipliers: (1, 2, 2, 2)
attention_resolutions: (1, 2)
num_timesteps: 1000
batch_size: 64
learning_rate: 0.0002
ema_decay: 0.999
steps: 14900
schedule: linear
beta_start: 0.0001
beta_end: 0.02
# CFG-specific:
guidance_scale: 3.0
class_dropout_prob: 0.1
num_classes: 10
null_class_id: 10
```

## Results
- Final loss: 0.0400 (100-step avg: 0.038)
- Best single-step loss: 0.0135 at step 14,503
- FID: Not computed
- Class differentiation: 145.4 pixel spread across 10 classes (step 14,800)
- Visual quality: Clear class-specific outputs visible in denoising GIF. T-shirts bright, dresses dark, bags rectangular. Recognizable but fine details still soft.
- Key observations:
  - Loss comparable to unconditional model (~0.038), confirming conditioning doesn't hurt convergence
  - Resume disruptions at steps 5,000 and 6,500 caused temporary quality collapse (diversity dropped from 0.347 to 0.103)
  - Distribution mean offset: -0.280 vs real -0.428 (delta +0.148) — worse than unconditional model's +0.032
  - Step 5,000 had better distribution stats than final step 14,900, suggesting chained training is not equivalent to continuous training
  - Training split across 7 chained TPU jobs due to 1-hour Kinetic timeout

## Artifacts
- GCS checkpoint: `gs://gcp-ml-172005-ddpm-training/cfg-fashionmnist-5k/run01/`
- Local EMA checkpoints: `artifacts/cfg-run/checkpoints/` (steps 1000, 5000, 6000, 9000, 14800)
- Final denoising GIF: `artifacts/cfg-run/denoising_step14800_10class_final.gif` (10 classes, 50 frames)
- Blog post: `artifacts/reports/class-conditional-fashionmnist-2026-04-23/class_conditional_diffusion.md`

## Decisions
- Decision 003: CFG over classifier guidance
- Decision 004: Guidance scale w=3.0 without ablation
- Decision 005: Chained resume training (7 jobs) to work around Kinetic timeout
- Decision 006: Attention placement at levels 1,2 (differs from unconditional's level 0) — noted as inconsistency

## Follow-ups
- [ ] Compute FID score on held-out test set
- [ ] Classification accuracy on generated samples (per-class accuracy metric)
- [ ] Guidance scale sweep (w=1, 3, 5, 7.5)
- [ ] Train to 50K+ steps for sharper details
- [ ] Run controlled comparison: same architecture, same steps, unconditional vs conditional (isolate CFG effect)
- [ ] Experiment 003: Unconditional vs CFG head-to-head comparison
