# Decision 006: Attention Placement — Resolved (Both Models Identical)

## Status
Resolved (was a documentation error, not an architectural difference)

## Date
2026-04-29 (identified), 2026-04-30 (resolved)

## Context
The unconditional model and conditional model were documented as using different self-attention placements:

| Model | Documented | Actual |
|-------|-----------|--------|
| Unconditional | `(0,)` (level 0 only) | `(1, 2)` (levels 1, 2) |
| Conditional | `(1, 2)` (levels 1, 2) | `(1, 2)` (levels 1, 2) |

Both models were actually trained with identical architecture: `attention_resolutions=(1, 2)`, `num_levels=3`, `channel_multipliers=(1, 2, 2)`, `base_filters=128`.

The perceived inconsistency arose because the post-restructure config defaults were changed to `attention_resolutions=(0,)` for the unconditional method, but this did NOT match the actual checkpoints. The checkpoints were created by the pre-restructure code which used `(1, 2)` for both.

## How It Was Discovered
When attempting to resume training on v5litepod-8, checkpoint loading failed with shape mismatches. Inspecting the checkpoint `.weights.h5` files revealed that both models have 5 self-attention layers (encoder × 2 + bottleneck × 1 + decoder × 2), all with 256 channels — exactly matching `attention_resolutions=(1, 2)` with `num_levels=3`.

## Resolution
1. Corrected config defaults to `attention_resolutions=(1, 2)` to match the actual trained models
2. Removed custom layer names from `build_unet` and `build_cond_unet` to match checkpoint naming (Keras auto-generated names)
3. Decision 006's original concern (unfair comparison) is moot — both models always had identical architecture

## Consequences
- The unconditional vs conditional comparison IS fair — same architecture, same attention placement
- The CFG effect can be attributed to the conditioning method alone, not to an architectural confound
- The `make_config()` defaults now correctly reflect the actual training configuration
