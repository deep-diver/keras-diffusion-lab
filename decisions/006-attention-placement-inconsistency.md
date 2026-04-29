# Decision 006: Attention Placement Inconsistency Between Methods

## Status
Accepted (technical debt)

## Date
2026-04-29

## Context
The unconditional model and conditional model use different self-attention placements in their U-Net architectures:

| Model | Attention Resolutions | Spatial Sizes |
|-------|----------------------|---------------|
| Unconditional | `(0,)` | Level 0 only (28x28) |
| Conditional | `(1, 2)` | Levels 1 and 2 (14x14, 7x7) |

This was not a deliberate design choice. The unconditional model was built first with `attention_resolutions=(0,)` (attention at the highest resolution). When the conditional model was added, the default config changed to `attention_resolutions=(1, 2)` following a different convention.

## Why This Matters
Any comparison between unconditional and conditional results is confounded by this architectural difference. If the conditional model produces better/worse images, we cannot attribute this solely to CFG conditioning — the attention placement also changed.

Attention at level 0 (28x28) operates on 784 spatial positions (computationally expensive but captures global structure). Attention at levels 1,2 (14x14, 7x7) operates on 196+49 positions (cheaper but more local).

## Decision
Accept the inconsistency for now. Document it as a known confound. Prioritize shipping over running a controlled ablation.

## Alternatives Considered
1. **Retrain conditional with attention_resolutions=(0,)** — Controlled comparison. Con: requires another 15K steps of TPU time (~7 hours).
2. **Retrain unconditional with attention_resolutions=(1,2)** — Same issue, different direction.
3. **Accept and document (chosen)** — Move forward with the current models, note the confound in the experiment report.
4. **Run both configs and compare** — Most rigorous. Con: doubles compute cost.

## Consequences
- The CFG blog post's class differentiation results cannot be directly compared to the unconditional blog post's quality results. This should be noted whenever comparing the two models.
- For future experiments, the attention placement should be explicitly controlled. Consider standardizing on one placement or running ablations.
- The `make_config()` defaults should be reviewed to ensure `attention_resolutions` is intentionally chosen, not accidentally inherited.
