# Diffusion Research Axes 2026

Survey of open research directions in diffusion models, prioritized for a Keras + TPU research harness focused on small-scale image generation (CIFAR-10, Fashion-MNIST).

---

## Landscape Overview

The diffusion model field has matured rapidly since DDPM (Ho et al. 2020). The core training recipe is well-established. The frontier has shifted to efficiency, compression, and understanding why these models work. Below are the most promising open research directions, prioritized by feasibility and impact for this harness.

---

## Priority 1: Model Pruning for Diffusion

**Status:** Active open question. No consensus method exists.

**Why it matters:** Diffusion models are over-parameterized. The standard DDPM U-Net for CIFAR-10 has ~35M parameters, but we don't know how much of this is redundant. Pruning could enable deployment on edge devices or faster sampling.

**Key open questions:**
- Does uniform pruning work, or does diffusion need timestep-aware pruning? (Different timesteps may rely on different network pathways.)
- Can we prune the model without retraining? (One-shot vs iterative pruning.)
- What is the minimum viable model size for reasonable CIFAR-10 generation?
- Does pruning interact with the noise schedule in unexpected ways?

**Key references:**
- Structure-Preserving Pruning for Diffusion Models (NPC, 2024) — timestep-aware structured pruning
- TAP (Timestep-Adaptive Pruning) — different sparsity per timestep
- Standard magnitude pruning applied to diffusion (baseline)

**Feasibility:** High. Pruning is a post-hoc modification — the baseline training loop doesn't change. Implementation involves: (1) train baseline, (2) apply pruning method, (3) optionally fine-tune, (4) evaluate FID/IS.

**Implementation difficulty:** Medium. Requires pruning utilities but not architectural changes.

---

## Priority 2: Knowledge Distillation for Diffusion

**Status:** Partially solved for step distillation, wide open for model compression.

**Why it matters:** Two distinct problems:
1. **Step distillation** — fewer sampling steps (Progressive Distillation, Consistency Models). This is relatively mature.
2. **Model distillation** — smaller student model. This is much less explored and the open question is more interesting.

**Key open questions:**
- Can a small student (e.g., 4M params) match a large teacher (35M params) on CIFAR-10?
- What is the best distillation loss? (Feature matching vs output matching vs distribution matching.)
- Does timestep-specific distillation help? (Different teachers for different timesteps.)
- How does distillation interact with EMA?

**Key references:**
- Progressive Distillation for Fast Sampling (Salimans & Ho, 2022)
- Consistency Models (Song et al., 2023)
- Guided Distillation (2024) — feature-based distillation for diffusion
- On дистилляция (ongoing work on model-size distillation)

**Feasibility:** High. Requires a trained teacher model (baseline) and a modified training loop for the student.

**Implementation difficulty:** Medium-High. The distillation training loop is different from the standard loop (needs teacher inference at each step).

---

## Priority 3: Small-Model Scaling Laws for Diffusion

**Status:** Surprisingly under-explored for diffusion on CIFAR-10.

**Why it matters:** We know scaling laws for language models (Chinchilla) and for large diffusion models (video/image generation). But the behavior of very small diffusion models (< 5M params) on small datasets is not well documented.

**Key open questions:**
- What is the minimum model size for recognizable CIFAR-10 generation?
- How does sample quality scale with model size in the small regime?
- Is there a phase transition where quality suddenly drops?
- Do architectural choices (attention placement, depth) matter more at small scale?

**Key references:**
- Scaling laws for neural language models (Kaplan et al., 2020) — the paradigm
- None specifically for small diffusion models — this IS the gap

**Feasibility:** Very high. Just train multiple model sizes and measure FID/IS. The harness naturally supports this via configurable model parameters.

**Implementation difficulty:** Low. Requires only sweep utilities on top of the baseline.

---

## Priority 4: Data Efficiency

**Status:** Significantly under-explored for diffusion.

**Why it matters:** Standard DDPM trains on the full 50K CIFAR-10 training set. What happens with less data? This has practical implications for specialized domains where data is scarce.

**Key open questions:**
- How few images can a diffusion model learn from while still generating reasonable samples?
- Does the noise schedule need adjustment for small datasets?
- Can data augmentation substitute for dataset size in diffusion?
- Is there a data efficiency phase transition similar to model size?

**Feasibility:** Very high. Requires only dataset subsetting, which is trivial to implement.

**Implementation difficulty:** Very low. Parameterize dataset size, run experiments.

---

## Priority 5: Quantization for Diffusion

**Status:** Active research area with recent progress.

**Why it matters:** Reducing precision (float32 -> int8) enables faster inference and deployment on resource-constrained hardware.

**Key open questions:**
- Can diffusion models be quantized to INT8 without quality loss?
- Does the timestep embedding need special treatment?
- Is quantization-aware training necessary, or is post-training quantization sufficient?

**Key references:**
- PQD: Post-training Quantization for Diffusion models (arXiv:2501.00124, Jan 2025)
- Q-DM (2024) — quantization-aware training for diffusion
- Efficient quantization techniques for U-Net (general)

**Feasibility:** Medium. Quantization interacts with JAX/TPU in non-obvious ways. JAX has some quantization support but it's not as mature as PyTorch.

**Implementation difficulty:** Medium-High. Requires understanding of JAX quantization primitives.

---

## Priority 6: Training Efficiency (Convergence Acceleration)

**Status:** Active area with multiple proposed methods, no consensus best approach.

**Why it matters:** DDPM requires 800K+ steps for CIFAR-10. Reducing this would lower TPU costs and iteration speed.

**Key open questions:**
- Can adaptive timestep sampling (training more on hard timesteps) converge faster?
- Does Min-SNR weighting help on small models?
- Are there better noise schedules that converge faster?
- Can warm-starting from a smaller model help?

**Key references:**
- Min-SNR weighting (Hang et al., 2023)
- Adaptive timestep sampling (various)
- Cosine schedule vs linear schedule comparison

**Feasibility:** Medium. Requires modifications to the training loop (loss weighting, timestep sampling distribution).

**Implementation difficulty:** Medium. The harness needs to support configurable loss weighting and timestep distributions.

---

## Priority 7: Architecture Variants (Token Merging, Efficient Attention)

**Status:** Active research with recent papers.

**Why it matters:** The U-Net self-attention layers are expensive. Making them cheaper enables larger models or faster training.

**Key references:**
- DaTo: Token pruning and caching for diffusion (arXiv:2501.00375, Jan 2025)
- DiC: Architecture redesign for diffusion (arXiv:2501.00603, Jan 2025)
- Token merging (ToMe) applied to diffusion

**Feasibility:** Medium. Requires architectural modifications to the U-Net.

---

## Recommendations

### First method to implement after baseline: **Pruning**

Rationale:
1. Lowest implementation complexity — doesn't change the training loop
2. Clear evaluation metric (FID/IS at various sparsity levels)
3. Directly answers an open question (how much redundancy exists?)
4. Natural stepping stone to distillation (pruned model = student candidate)

### Second method: **Scaling Laws Sweep**

Rationale:
1. Trivially implementable once the baseline is parameterized
2. Produces novel data (small-model scaling for diffusion is not well documented)
3. Informs all other research directions (how small can we go?)

### Long-term targets: **Knowledge Distillation**, **Data Efficiency**

These are high-value but require more infrastructure (teacher-student training loop, dataset subsetting with proper evaluation).

---

## Harness Design Implications

To support these research directions, the harness needs:

1. **Configurable model architecture** — parameterized width, depth, attention placement
2. **Plug-in training loop** — support for custom loss functions, timestep sampling, and distillation objectives
3. **Evaluation pipeline** — FID/IS computation on generated samples
4. **Experiment management** — easy sweep over hyperparameters
5. **Post-hoc model modification** — pruning, quantization applied to trained models
6. **Model-agnostic monitoring** — loss, gradients, sample quality at every stage
