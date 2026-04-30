# Building a Diffusion Model Research Harness from Scratch

> How we built an extensible diffusion model framework on Keras 3 + JAX, trained two models on Google Cloud TPUs, and learned what the tutorials don't tell you.

**April 2026** | [GitHub](https://github.com/deep-diver/keras-diffusion-lab)

---

## The Starting Point

Diffusion models generate images by learning to reverse a gradual noising process. The concept is elegant — add noise step by step until an image becomes static, then train a neural network to undo each step. The math is clean, the papers are clear, and the results are stunning.

But when you actually sit down to build one from scratch — not using Hugging Face diffusers, not cloning a reference implementation — the gap between "understanding the paper" and "having a working model" turns out to be wide. Really wide.

This is the story of closing that gap: building a complete diffusion model research harness on Keras 3 + JAX, training it on Fashion-MNIST using Google Cloud TPUs, then extending it to support class-conditional generation with Classifier-Free Guidance. Along the way, we hit every pothole the tutorials skip over.

---

## Why Build From Scratch?

Three reasons:

**1. Research flexibility.** Pre-built pipelines are optimized for production use cases. Research requires poking at things — changing the loss mid-training, pruning parts of the network, distilling knowledge into smaller models. Frameworks like diffusers make standard workflows easy but non-standard workflows painful.

**2. Understanding.** There's a difference between knowing that DDPM uses epsilon-prediction and understanding what happens when your EMA decay is wrong and your model generates solid black rectangles for 5,000 steps. Building from scratch forces you to confront every detail.

**3. Keras 3 ecosystem.** Keras 3's multi-backend design (JAX, TensorFlow, PyTorch) combined with Keras Kinetic for TPU access is a genuinely interesting stack that hasn't been explored much for diffusion training. We wanted to see what works and what doesn't.

The goal: a clean, extensible codebase where adding a new research method means creating one directory, not restructuring the whole project.

---

## What We Built

Two trained models on Fashion-MNIST (28x28 grayscale, 10 classes):

| Model | Steps | Params | What It Does |
|-------|-------|--------|-------------|
| **Unconditional DDPM** | 15,000 | ~20M | Generates random fashion items — press "generate," get a surprise |
| **Class-Conditional CFG** | 14,900 | ~21M | Generates specific classes — ask for a sneaker, get a sneaker |

Both trained on Google Cloud TPU v5litepod-4 (4 TPU chips) using Keras Kinetic for remote execution. Both produce recognizable fashion items. The conditional model demonstrates class-level control through Classifier-Free Guidance.

```
Unconditional:  noise → [U-Net x1000 steps] → random fashion item
Conditional:    noise + "sneaker" → [Guided U-Net x1000 steps] → 👟 sneaker
```

The full harness includes:
- A method registry for plugging in new research approaches
- A template-method training loop with shared EMA, checkpointing, and logging
- GCS-based artifact persistence for TPU job survival
- 41 tests, 6 architecture decision records, and detailed experiment cards

---

## The Architecture

At the core is a U-Net with FiLM conditioning — the same architectural family used in DDPM, Improved DDPM, and many subsequent papers.

```
Input: noisy image (28x28x1)
  │
  ├── Down path (3 levels): 128 → 256 → 256 channels
  │     ResBlock + FiLM(time) + optional Self-Attention
  │
  ├── Bottleneck: 256 channels with self-attention
  │
  └── Up path (3 levels): 256 → 256 → 128 channels
        ResBlock + FiLM(time) + skip connections
  │
Output: predicted noise (28x28x1)
```

Each `ResBlock` receives a timestep embedding via FiLM (Feature-wise Linear Modulation) — essentially "here's how noisy this image is, adjust your processing accordingly." Self-attention is applied at the lowest spatial resolution (7x7) to capture global structure.

For class-conditional generation, we add a `ClassEmbedding` layer that produces a vector for each class label. This gets added to the time embedding, so the model receives combined "what timestep is this" + "what class should this be" information through the same FiLM pathway. No architectural gymnastics — just add the embeddings and go.

```
Time embedding:     [batch, 512]
+ Class embedding:  [batch, 512]    ← new for CFG
= Combined:         [batch, 512]    → FiLM conditioning throughout U-Net
```

### A Decision Worth Noting: 3 Levels, Not 4

Fashion-MNIST images are 28x28. A standard 4-level U-Net halves the spatial dimensions four times: 28 → 14 → 7 → 3 → 1. The problem: 3x3 is an awkward spatial size for convolutions, and the information loss at that resolution hurts quality. We use 3 levels instead (28 → 14 → 7 → bottleneck), which keeps clean spatial dimensions throughout. This is documented in [Decision 002](decisions/002-unet-3-levels-for-28x28.md).

---

## The Journey: What Actually Happened

### Phase 1: "It's All Black"

The first model trained for 5,000 steps. The loss went down beautifully. But every generated sample was a dark, amorphous blob — barely distinguishable from solid black.

The culprit: **EMA decay = 0.9999**.

The canonical DDPM paper uses EMA decay of 0.9999, but it trains for 800,000 steps. At that decay rate, the EMA weights have an effective averaging window of 10,000 steps. After only 5,000 steps, **60.7% of the EMA weights were still the initial random values**. The training model itself was generating decent samples, but we were sampling from the EMA model — which hadn't caught up.

Diagnosis was straightforward once we looked:
- Training model at step 5,000: mean=-0.30, std=0.48 (realistic)
- EMA model at step 5,000: mean=-0.86, std=0.19 (nearly black)

Fix: changed EMA decay to 0.999 (1,000-step window). Samples immediately improved. This is [Decision 001](decisions/001-ema-decay-tuning.md) — the kind of thing papers assume you'll just know.

### Phase 2: Chaining TPU Jobs

Keras Kinetic has a 1-hour timeout per job. At ~0.55 steps/second with batch size 64, each job covers roughly 2,000 steps. To reach 15,000 steps, we chained 8 jobs — each one resuming from the latest checkpoint:

```
Job 1: steps 0→2,000      (timed out)
Job 2: steps 2,000→4,000   (timed out)
  ...
Job 8: steps 14,000→15,000 (completed!)
```

This worked, but introduced a subtle problem: quality regressions at job boundaries. When resuming, the Adam optimizer's momentum buffers are restored, but the learning rate schedule and gradient statistics need a few hundred steps to re-stabilize. Loss spikes at resume points are visible in the training curve:

```
Step 4,900: loss = 0.031   ← end of Job 2
Step 5,100: loss = 0.144   ← start of Job 3 (resume bump)
Step 5,500: loss = 0.035   ← recovered
```

The fix isn't elegant — just be aware of it and don't evaluate quality at resume boundaries. [Decision 005](decisions/005-chained-resume-training.md) documents this in full.

### Phase 3: Adding Class Control

Once unconditional generation worked, the next question was obvious: can we control *what* gets generated? Classifier-Free Guidance (CFG) is the standard answer — train one model that can denoise both with and without a class label, then interpolate between the two predictions during sampling:

```
guided_prediction = (1 + w) * conditioned - w * unconditioned
```

Where `w` is the guidance scale. At w=3.0 (our default), this pushes the model toward the specified class while maintaining sample diversity.

Implementing CFG required restructuring the entire codebase from a single-experiment project into a multi-method harness. The original `models/`, `training/`, and `sampling/` modules became thin re-exports, while new `methods/unconditional/` and `methods/class_conditional/` packages held the actual implementations. A method registry (`get_method("class_conditional")`) dispatches to the right modules.

The restructure was the hardest engineering task in the project — touching 7 existing modules while keeping all 31 tests passing. But it paid off: the class-conditional implementation reused 90% of the unconditional code. `CFGTrainer` extends `BaseTrainer` and only overrides `train_step()` — the training loop, EMA, checkpointing, and logging are all inherited.

---

## Results

### Unconditional DDPM (15,000 steps)

The model generates recognizable fashion items: t-shirts, trousers, bags, sneakers, dresses. Some samples are crisp; others are blurry or show class blending (a shoe that's halfway between a sneaker and a sandal).

| Metric | Value |
|--------|-------|
| Final loss (100-step avg) | 0.038 |
| Best single-step loss | 0.0218 (step 12,000) |
| Training time | ~5 hours across 8 TPU jobs |
| Throughput | ~0.55 steps/sec, ~35 images/sec |

### Class-Conditional CFG (14,900 steps)

The model generates class-specific items. Asking for class 7 ("sneaker") produces sneakers; class 8 ("bag") produces bags. Class boundaries are reasonably sharp, though some cross-class bleeding remains.

| Metric | Value |
|--------|-------|
| Final loss (100-step avg) | 0.038 |
| Best single-step loss | 0.0135 (step 14,503) |
| Training time | ~7 hours across 7 TPU jobs |
| Guidance scale | 3.0 (not ablated) |

---

## What We'd Do Differently

Honest assessment of the limitations — things that worked, things that didn't, and things we should have done but didn't.

### No FID Scores

The single biggest gap. We evaluated sample quality visually and with pixel distribution statistics (mean, std, histogram comparison). Neither is a reliable proxy for perceptual quality. FID (Frechet Inception Distance) is the standard metric, and we should have computed it at regular intervals throughout training. Without FID:
- We can't objectively compare unconditional vs. conditional quality
- We can't determine the optimal stopping point
- We can't benchmark against published results

### No Ablation Studies

We used guidance scale w=3.0 because it's moderate and "looked good." We didn't sweep w=1, 2, 3, 5, 7.5 to find the optimal value. Similarly, we used class dropout probability 0.1 from the CFG paper without testing 0.05 or 0.2. These are free parameters that directly affect quality, and we cargo-culted them from the literature.

### Unfair Comparison

The unconditional and conditional models have different attention placements (unconditional: level 0; conditional: levels 1, 2). This means any quality difference between the two could be due to attention configuration rather than the conditional training method. [Decision 006](decisions/006-attention-placement-inconsistency.md) flags this as a known confound.

### No Learning Rate Schedule

We used a constant learning rate of 2e-4 throughout training. A cosine annealing or warmup schedule might have improved convergence, especially given the resume-boundary spikes in chained training.

### Missing Evaluation Suite

A proper evaluation pipeline would include:
- FID at every 1,000 steps
- Inception Score
- Classification accuracy of generated samples (does a "sneaker" sample actually look like a sneaker to a classifier?)
- Diversity metrics within each class

We have none of these. The evaluation is "look at the samples and check if they seem right."

---

## The Codebase

The final structure reflects the journey from single experiment to research harness:

```
src/diffusion_harness/
  base/                    # Shared: build_unet, BaseTrainer, BaseSampler
  methods/
    unconditional/         # DDPM epsilon-prediction
    class_conditional/     # CFG with guided sampling
    pruning/               # (TODO)
    distillation/          # (TODO)
  core/                    # make_config() builder
  schedules/               # Linear/cosine noise schedules
  data/                    # Dataset loading with optional labels
  monitoring/              # JSONL structured logging
  utils/                   # GCS helpers for TPU persistence

decisions/                 # 6 ADR-style decision records
experiments/               # 2 experiment cards with configs + results
tests/                     # 41 tests (31 original + 10 CFG)
```

Adding a new method means creating `methods/<name>/` with four functions — `build_model()`, `build_trainer()`, `build_sampler()`, and `config.py` — then registering it. The base classes handle the rest.

The pattern we landed on:
- **Template method** for the training loop (`BaseTrainer.train()` calls `train_step()` which subclasses implement)
- **Inheritance** for method-specific behavior (`CFGTrainer` extends `BaseTrainer`)
- **Registry** for dispatch (`get_method("class_conditional")`)

This is deliberately simple — no plugin framework, no dependency injection, no configuration DSL. Just classes and a dictionary lookup. For a research harness with a handful of methods, this is the right level of abstraction.

---

## TPU Operations: The Undocumented Parts

Running on Google Cloud TPUs via Keras Kinetic taught us things that aren't in any tutorial:

**Always use Spot instances.** Up to 91% cheaper. Spot preemption is rare for short jobs (< 1 hour), and our checkpoint-every-1,000-steps strategy means we lose at most a few minutes of training on preemption.

**Keras Kinetic has a context packaging bug** (as of v0.0.1). The `context.zip` it uploads to the TPU worker captures `kinetic/core/` instead of your project source. The workaround is to use the `volumes` parameter to explicitly ship your code:

```python
src_data = kinetic.Data("./src/")

@kinetic.run(accelerator="v5litepod-4", volumes={"/tmp/src": src_data})
def remote_train():
    import sys
    sys.path.insert(0, "/tmp/src")
    # now your imports work
```

**GCS is your only persistence.** Everything on the TPU worker dies with the job. We checkpoint every 1,000 steps: model weights, EMA weights, optimizer state, and a training state JSON. The checkpoint quartet is the minimum viable persistence.

### What If I Want a Larger TPU Pool?

All our experiments used `v5litepod-4` (4 TPU chips). But the harness supports larger pools out of the box — just change the `--accelerator` flag and scale the batch size.

Keras 3 + JAX handles data parallelism automatically. When JAX sees multiple TPU chips, it replicates the model across devices and splits each batch. No code changes needed — no mesh configuration, no sharding annotations for single-host setups.

| Accelerator | Chips | Batch Size | Throughput | Use Case |
|-------------|-------|-----------|------------|----------|
| `v5litepod-1` | 1 | 32 | ~0.14 steps/s | Debugging, local testing |
| `v5litepod-4` | 4 | 64 | ~0.55 steps/s | Research iteration (what we used) |
| `v5litepod-8` | 8 | 128 | ~1.1 steps/s | Faster training, larger models |

```bash
# 8-chip pool — 2x throughput, double the batch size
kinetic pool add --accelerator v5litepod-8 --spot --project YOUR_PROJECT --zone us-west4-a

KERAS_BACKEND=jax KERAS_REMOTE_PROJECT=YOUR_PROJECT python remote_train.py \
    --gcs-bucket gs://YOUR_BUCKET/runs/run01 \
    --accelerator v5litepod-8 \
    --zone us-west4-a --dataset fashion_mnist --batch-size 128 --steps 15000
```

The throughput scales roughly linearly because each chip processes its sub-batch independently — pure data parallelism. The same ~20M parameter model that fits on 4 chips also fits on 8, but each step processes twice as many images.

**When you need more than 8 chips**, you enter multi-host territory. TPU v5e supports up to 16 chips (4x4 topology) in a single slice, but this requires multiple VMs and explicit JAX mesh configuration. That's beyond what Kinetic handles automatically today, and would be the next infrastructure project if scaling studies demand it.

**When you need a larger model**, not just faster training — increase `--base-filters` (e.g., 128 to 256) and possibly `--num-levels` (3 to 4 for images larger than 28x28). The larger model needs more memory per chip, which may require the 8-chip pool even at the same batch size.

### What About Newer TPU Generations?

All our experiments used TPU v5e (`v5litepod-4`). But Google Cloud offers significantly faster chips now, and the harness works with them out of the box — just change the `--accelerator` flag.

The current Google Cloud TPU landscape:

| Generation | Accelerator | Key Specs | Status |
|------------|-------------|-----------|--------|
| **v5e** | `v5litepod-4` | 16 GB HBM/chip | GA (what we used) |
| **v5p** | `v5p-8` | 95 GB HBM/chip, ~2x compute | GA |
| **v6e (Trillium)** | `v6e-8` | 32 GB HBM/chip, 4.7x compute vs v5e | GA |
| **7th Gen (Ironwood)** | — | 10x vs v5p | GA 2025 |
| **8th Gen (TPU 8t/8i)** | — | 121 Exaflops superpod | Announced 2026 |

For a 20M-parameter diffusion model on Fashion-MNIST, the v5litepod-4 is already more than sufficient — our bottleneck was the 1-hour job timeout, not compute. But for scaling studies (larger models, higher-resolution datasets like CIFAR-10 or ImageNet), newer generations become compelling:

- **v5p** gives you 95 GB HBM per chip — enough to train models in the 100M+ parameter range without worrying about memory
- **Trillium (v6e)** delivers 4.7x the compute per chip at 67% better energy efficiency, making it the best performance-per-dollar option for training
- **Ironwood and beyond** are aimed at frontier-scale training (think Gemini-class models) and are overkill for a research harness — but the code would work on them unchanged

The key insight: because Keras 3 + JAX handles device abstraction, and Kinetic handles job submission, the harness code is completely TPU-generation-agnostic. You don't rewrite anything — you just provision a different pool.

```bash
# Upgrade from v5e to Trillium — no code changes
kinetic pool add --accelerator v6e-8 --spot --project YOUR_PROJECT --zone REGION

KERAS_BACKEND=jax KERAS_REMOTE_PROJECT=YOUR_PROJECT python remote_train.py \
    --gcs-bucket gs://YOUR_BUCKET/runs/run01 \
    --accelerator v6e-8 \
    --zone REGION --dataset cifar10 --base-filters 256 --steps 50000
```

These lessons are documented in [TPU field notes](artifacts/reports/keras_kinetic_tpu_field_notes.md) and [engineering tips](artifacts/reports/engineering_tips.md).

---

## What's Next

The harness is designed to support the research directions that matter most for diffusion models in 2026:

**Model pruning** — How small can a diffusion U-Net get before quality degrades? Is uniform pruning sufficient, or do different timesteps need different network capacity?

**Knowledge distillation** — Can a 4M-parameter student match a 20M-parameter teacher on Fashion-MNIST? What distillation loss works best for diffusion?

**Scaling studies** — How does sample quality scale with model size, training compute, and dataset size on this stack?

**FID evaluation pipeline** — The most urgent missing piece. Until we can measure quality objectively, every other experiment is running blind.

Each of these is a new directory under `methods/`, a new set of experiments under `experiments/`, and new decisions under `decisions/`. The harness is ready.

---

## Deep Dives

This post covers the project at a high level. For the full technical details, see:

| Topic | Deep Dive |
|-------|-----------|
| Unconditional DDPM training | [Training a DDPM on Fashion-MNIST](artifacts/reports/fashion-mnist-diffusion-2026-04-23/fashion_mnist_diffusion.md) |
| Class-conditional CFG training | [Class-Conditional Generation with CFG](artifacts/reports/class-conditional-fashionmnist-2026-04-23/class_conditional_diffusion.md) |
| TPU operations | [Keras Kinetic TPU Field Notes](artifacts/reports/keras_kinetic_tpu_field_notes.md) |
| Practical tips | [Engineering Tips](artifacts/reports/engineering_tips.md) |
| Research directions | [Diffusion Research Axes 2026](artifacts/reports/diffusion_research_axes_2026.md) |

## References

1. Ho, Jain & Abbeel (2020). "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
2. Ho & Salimans (2022). "Classifier-Free Diffusion Guidance." NeurIPS 2021 Workshop.
3. Nichol & Dhariwal (2021). "Improved Denoising Diffusion Probabilistic Models." ICML 2021.
4. Perez et al. (2018). "FiLM: Visual Reasoning with a General Conditioning Layer." AAAI 2018.
5. Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022.

---

*Built with Keras 3, JAX, and Google Cloud TPUs. Available at [github.com/deep-diver/keras-diffusion-lab](https://github.com/deep-diver/keras-diffusion-lab).*
