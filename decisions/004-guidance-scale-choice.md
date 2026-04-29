# Decision 004: Guidance Scale w=3.0 Without Ablation

## Status
Accepted

## Date
2026-04-23

## Context
Classifier-Free Guidance requires choosing a guidance scale w that controls the trade-off between class adherence and diversity. The formula `eps = (1+w)*eps_cond - w*eps_uncond` amplifies the difference between conditional and unconditional predictions. Common values in the literature range from w=1.0 (no guidance) to w=7.5 (Stable Diffusion default) to w=10+ (aggressive, often artifacts).

## Decision
Use w=3.0 as the default guidance scale for all CFG experiments.

## Alternatives Considered
1. **w=1.0** (no guidance boost) — Maximum diversity but weak class control. The model might not reliably produce the target class.
2. **w=3.0** (chosen) — Moderate guidance. Used by many early CFG papers. Expected to produce strong class differentiation without artifacts on a small 28x28 model.
3. **w=5.0** — Strong guidance. Could produce sharper class features but risks artifacts (oversaturation, unnatural contrast).
4. **w=7.5** — Stable Diffusion's default. Designed for high-resolution latent diffusion; likely too aggressive for a 20M-param pixel-space model on 28x28.
5. **Systematic sweep** — Train once, evaluate at w=1, 3, 5, 7.5 with FID. The rigorous approach but requires FID evaluation infrastructure we don't have yet.

## Consequences
- We cannot confirm w=3.0 is optimal. It may be too conservative (weak class signal) or too aggressive (artifacts we can't detect without FID).
- The class differentiation in the denoising GIF looks convincing at w=3.0 (145.4 pixel spread), but this is a visual assessment, not a quantitative one.
- **Gap**: A guidance scale sweep should be a priority before the next experiment. This is a single-command experiment (generate samples with different w from the same checkpoint) that doesn't require retraining.
