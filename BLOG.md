# Building a Diffusion Model Research Harness from Scratch

> How we built an extensible diffusion model framework on Keras 3 + JAX, trained two models on Google Cloud TPUs, and discovered that Classifier-Free Guidance doesn't work at 30K steps — along with everything else the tutorials don't tell you.

**May 2026** | [GitHub](https://github.com/deep-diver/keras-diffusion-lab)

---

## The Starting Point

Diffusion models generate images by learning to reverse a gradual noising process. The concept is elegant — add noise step by step until an image becomes static, then train a neural network to undo each step. The math is clean, the papers are clear, and the results are stunning.

But when you actually sit down to build one from scratch — not using Hugging Face diffusers, not cloning a reference implementation — the gap between "understanding the paper" and "having a working model" turns out to be wide. Really wide.

This is the story of closing that gap: building a complete diffusion model research harness on Keras 3 + JAX, training it on Fashion-MNIST using Google Cloud TPUs, extending it to support class-conditional generation with Classifier-Free Guidance, then evaluating everything rigorously and discovering that the most important feature (CFG class control) doesn't work as expected.

---

## Why Build From Scratch?

Three reasons:

**1. Research flexibility.** Pre-built pipelines are optimized for production use cases. Research requires poking at things — changing the loss mid-training, pruning parts of the network, distilling knowledge into smaller models. Frameworks like diffusers make standard workflows easy but non-standard workflows painful.

**2. Understanding.** There's a difference between knowing that DDPM uses epsilon-prediction and understanding what happens when your EMA decay is wrong and your model generates solid black rectangles for 5,000 steps. Building from scratch forces you to confront every detail.

**3. Keras 3 ecosystem.** Keras 3's multi-backend design (JAX, TensorFlow, PyTorch) combined with Keras Kinetic for TPU access is a genuinely interesting stack that hasn't been explored much for diffusion training. We wanted to see what works and what doesn't.

The goal: a clean, extensible codebase where adding a new research method means creating one directory, not restructuring the whole project.

---

## What We Built

Two trained models on Fashion-MNIST (28x28 grayscale, 10 classes), plus three research experiments:

| Model | Steps | Params | What It Does |
|-------|-------|--------|-------------|
| **Unconditional DDPM** | 30,000 | ~20M | Generates random fashion items — press "generate," get a surprise |
| **Class-Conditional CFG** | 30,000 | ~21M | Generates specific classes — ask for a sneaker, get... well, that's the story |

Both trained on Google Cloud TPU v5litepod-4 (4 TPU chips) using Keras Kinetic for remote execution, across 15 chained TPU jobs (~13 hours total).

```
Unconditional:  noise → [U-Net x1000 steps] → random fashion item
Conditional:    noise + "sneaker" → [Guided U-Net x1000 steps] → should be 👟
```

The full harness includes:
- A method registry for plugging in new research approaches
- A template-method training loop with shared EMA, checkpointing, and logging
- DDIM deterministic sampling (20x faster than DDPM, no retraining)
- FID evaluation using a domain-specific classifier
- Guidance scale sweep infrastructure
- 58 tests, 6 architecture decision records, and detailed experiment cards

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

For class-conditional generation, we add a `ClassEmbedding` layer that produces a vector for each class label. This gets added to the time embedding, so the model receives combined "what timestep is this" + "what class should this be" information through the same FiLM pathway.

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

Keras Kinetic has a 1-hour timeout per job. At ~0.55 steps/second with batch size 64, each job covers roughly 2,000 steps. To reach 30,000 steps, we chained 15 jobs across three hardware phases:

```
# Phase 1: v5litepod-4 (steps 0–15K), ~7.5 hours
Job 1-8: steps 0→~16,200

# Phase 2: v5litepod-8 (steps 15K–20K), ~3 hours
Job 9-10: steps 16,200→20,200

# Phase 3: v5litepod-4 spot (steps 20K–30K), ~5 hours
Job 11-15: steps 20,000→30,100
```

Each job downloads the latest checkpoint from GCS, restores all state, and continues. Quality regressions at job boundaries are visible in the loss curve but recover within ~500 steps. [Decision 005](decisions/005-chained-resume-training.md) documents this in full.

### Phase 3: Adding Class Control

Once unconditional generation worked, the next question was obvious: can we control *what* gets generated? Classifier-Free Guidance (CFG) is the standard answer — train one model that can denoise both with and without a class label, then interpolate between the two predictions during sampling:

```
guided_prediction = (1 + w) * conditioned - w * unconditioned
```

Implementing CFG required restructuring the entire codebase from a single-experiment project into a multi-method harness. The restructure touched 7 existing modules while keeping all 31 tests passing. But it paid off: the class-conditional implementation reused 90% of the unconditional code. `CFGTrainer` extends `BaseTrainer` and only overrides `train_step()`.

### Phase 4: Evaluation — What We Wish We'd Done First

After training both models to 30K steps, we finally built the evaluation pipeline we should have had from the start:

**DDIM Sampling** — Standard DDPM needs 1000 forward passes per image (~52s each). DDIM reduces this to 50 steps (~2.6s) — a 20x speedup — by visiting only a subsequence of timesteps with a deterministic update rule. No retraining required. This made all subsequent evaluation feasible.

**FID Evaluation** — We trained a Fashion-MNIST classifier (91.52% accuracy) and used its penultimate layer as a feature extractor for FID. Domain-specific features instead of InceptionV3, which would require 10x upscaling from 28x28 to 299x299.

**Guidance Scale Sweep** — We swept w ∈ {1.0, 3.0, 5.0, 7.5} with 100 samples per scale, measuring both FID and classification accuracy.

---

## The Uncomfortable Finding: CFG Doesn't Work

Here's what we expected: higher guidance scale → stronger class signal → better accuracy. This is how CFG works in published models like Stable Diffusion.

Here's what we found:

| Guidance Scale (w) | FID | Accuracy | What's happening |
|---------------------|-----|----------|-------------------|
| 1.0 | ~93 | **29%** | Weak class signal, some correct |
| 3.0 (our default) | **~70** | **10%** | Best FID, but chance-level accuracy |
| 5.0 | ~75 | **1%** | Accuracy collapses |
| 7.5 | ~81 | **0%** | No sample classified correctly |

The CFG paradox: **w=3.0 produces the best-looking images but completely ignores the class label.** Classification accuracy at w=3.0 is exactly 10% — random chance for 10 classes.

### Why It Fails

CFG amplifies the difference between the conditional and unconditional predictions:

```
ε_guided = ε_cond + w × (ε_cond − ε_uncond)
                       ↑
                  The "class signal"
```

If the model hasn't learned to differentiate by class (ε_cond ≈ ε_uncond), the amplified signal is just noise. The model generates reasonable unconditional images → decent FID. But there's no class information → chance accuracy.

At only 30K training steps (3.75% of the typical 800K convergence budget), the model simply hasn't trained long enough for the conditional/unconditional gap to develop. Published CFG results use 200K-800K steps.

### What This Means

This is the most important finding about our conditional model. It doesn't mean CFG is broken — it means our model is undertrained. The priority for future work is clear: train longer and re-evaluate.

---

## The FID Lesson

Building the FID pipeline taught us something equally important: **128-dimensional features are useless at small sample sizes**.

FID requires estimating a 128×128 covariance matrix from generated samples. The rank of this estimate is min(n, d):

| Samples | Covariance Rank | FID Reliability |
|---------|----------------|-----------------|
| 4 (DDIM comparison) | 3% | Meaningless |
| 8 (initial evaluation) | 6% | Extremely noisy |
| 100 (guidance sweep) | 78% | Marginal |
| 500+ | 100% | Reliable |

Our initial FID evaluation (8 samples) produced numbers like "87.14" and "131.41" — false precision from a rank-deficient matrix. The matrix square root computation issued singularity warnings. We've since learned to treat FID at these sample sizes as a directional indicator only.

The fix for future work: reduce the feature dimension to 32 (reliable at n=100) or generate 500+ samples per evaluation.

---

## Results

### Unconditional DDPM (30,000 steps)

The model generates recognizable fashion items: t-shirts, trousers, bags, sneakers, dresses. Some samples are crisp; others show class blending.

| Metric | Value |
|--------|-------|
| Final loss (100-step avg) | 0.034 |
| Best single-step loss | 0.0107 (step 27,781) |
| FID (8 samples, unreliable) | ~87 |
| Training time | ~13 hours across 15 TPU jobs |

### Class-Conditional CFG (30,000 steps)

The model generates visually appealing fashion items, but class control is non-functional at all tested guidance scales.

| Metric | Value |
|--------|-------|
| Final loss (100-step avg) | 0.034 |
| Best single-step loss | 0.0135 (step 14,503) |
| FID at w=3.0 (100 samples) | ~70 |
| Classification accuracy at w=3.0 | 10% (chance) |
| Training time | ~13 hours across 15 TPU jobs |

### DDIM Sampling

| Metric | DDPM-1000 | DDIM-50 |
|--------|-----------|---------|
| Time per sample | 52.5s | 2.6s |
| Speedup | 1x | **20x** |
| FID (4 samples, unreliable) | ~274 | ~262 |

---

## What We'd Do Differently

Honest assessment — things that worked, things that didn't, and things we should have done but didn't.

### Should Have Built Evaluation First

We spent months training models without FID. By the time we added it, we'd already committed to 30K steps with a model whose class conditioning doesn't work. If we'd had FID + accuracy evaluation from step 1K, we'd have caught the CFG failure early and either trained longer or adjusted the approach.

### 128-dim Features Were Too Large

The Dense(128) feature layer was a poor match for our sample sizes. A Dense(32) layer would have made FID reliable at n=100 (our practical limit for sweeps). The extra 96 dimensions don't help if you can't estimate their covariance.

### Attention Placement Confound

The unconditional and conditional models have different attention placements (level 0 vs levels 1,2). This means any quality difference could be due to attention configuration rather than the conditional training method. [Decision 006](decisions/006-attention-placement-inconsistency.md) flags this.

### 30K Steps Is Not Enough for CFG

Published CFG results use 200K-800K steps. Our 30K steps is 3.75% of the typical budget. The class conditioning failure is almost certainly due to insufficient training, not an architectural problem.

---

## The Codebase

The final structure reflects the journey from single experiment to research harness:

```
src/diffusion_harness/
  base/                    # Shared: build_unet, BaseTrainer, BaseSampler, DDIMSampler
  methods/
    unconditional/         # DDPM epsilon-prediction
    class_conditional/     # CFG with guided sampling (DDPM + DDIM)
    pruning/               # (TODO)
    distillation/          # (TODO)
  core/                    # make_config() builder
  schedules/               # Linear/cosine noise schedules
  data/                    # Dataset loading with optional labels
  metrics/                 # Classifier, FID, classification accuracy
  monitoring/              # JSONL structured logging
  utils/                   # GCS helpers for TPU persistence

scripts/                   # Evaluation and visualization scripts
decisions/                 # 6 ADR-style decision records
experiments/               # 2 experiment cards with configs + results
tests/                     # 58 tests
```

The pattern we landed on:
- **Template method** for the training loop (`BaseTrainer.train()` calls `train_step()` which subclasses implement)
- **Inheritance** for method-specific behavior (`CFGTrainer` extends `BaseTrainer`)
- **Registry** for dispatch (`get_method("class_conditional")`)
- **Hook pattern** for sampling (`DDIMSampler.model_predict()` — override for method-specific prediction)

---

## What's Next

The three research experiments pointed to clear next steps:

**1. Train longer (50K-100K steps).** The #1 priority. CFG fails because the model is undertrained. Resume from the 30K checkpoint and see if the conditional/unconditional gap develops with more training.

**2. CFG diagnostic.** Directly measure ε_cond vs ε_uncond during inference. If they're nearly identical, it confirms the hypothesis. If they diverge, the problem is elsewhere.

**3. Reduce FID feature dimension to 32.** Quick fix that makes FID reliable at practical sample sizes.

**4. Unconditional baseline comparison.** Generate 500+ samples from both models with DDIM-50 and compare FID properly. If unconditional FID ≈ conditional FID at w=3.0, it confirms the model is ignoring the class label.

**5. Cosine schedule.** Implemented but never used. Published results show it helps at fewer training steps.

---

## Deep Dives

This post covers the project at a high level. For the full technical details, see:

| Topic | Deep Dive |
|-------|-----------|
| Unconditional DDPM training (30K steps) | [Training a DDPM on Fashion-MNIST](artifacts/reports/fashion-mnist-diffusion-2026-04-23/fashion_mnist_diffusion.md) |
| Class-conditional CFG training (30K steps) | [Class-Conditional Generation with CFG](artifacts/reports/class-conditional-fashionmnist-2026-04-23/class_conditional_diffusion.md) |
| DDIM 20x speedup | [DDIM Sampling Report](artifacts/reports/ddim-sampling-2026-05/ddim_sampling.md) |
| FID reliability analysis | [FID Evaluation Report](artifacts/reports/fid-evaluation-2026-05/fid_evaluation.md) |
| CFG failure analysis | [Guidance Sweep Report](artifacts/reports/guidance-sweep-2026-05/guidance_sweep.md) |
| TPU operations | [Keras Kinetic TPU Field Notes](artifacts/reports/keras_kinetic_tpu_field_notes.md) |
| Practical tips | [Engineering Tips](artifacts/reports/engineering_tips.md) |
| Research directions | [Diffusion Research Axes 2026](artifacts/reports/diffusion_research_axes_2026.md) |

## References

1. Ho, Jain & Abbeel (2020). "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
2. Ho & Salimans (2022). "Classifier-Free Diffusion Guidance." NeurIPS 2021 Workshop.
3. Song, Meng & Ermon (2021). "Denoising Diffusion Implicit Models." ICLR 2021.
4. Nichol & Dhariwal (2021). "Improved Denoising Diffusion Probabilistic Models." ICML 2021.
5. Heusel et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a local Nash equilibrium." NeurIPS 2017.
6. Perez et al. (2018). "FiLM: Visual Reasoning with a General Conditioning Layer." AAAI 2018.
7. Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022.

---

*Built with Keras 3, JAX, and Google Cloud TPUs. Available at [github.com/deep-diver/keras-diffusion-lab](https://github.com/deep-diver/keras-diffusion-lab).*
