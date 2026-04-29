# Decision 001: EMA Decay Tuning for Short Training Runs

## Status
Accepted

## Date
2026-04-23

## Context
The canonical DDPM paper (Ho et al. 2020) recommends EMA decay=0.9999 for 800K step training. Our initial Fashion-MNIST training ran for only 5K steps. At step 5000 with decay=0.9999, generated samples were nearly black (mean=-0.86 vs real data mean=-0.43). Investigation showed 60.7% of EMA weights were still random initialization — the effective averaging window (10,000 steps) was larger than the entire training run.

## Decision
Changed default `ema_decay` from 0.9999 to 0.999 for the harness baseline.

The effective EMA window is `1 / (1 - decay)`:
- decay=0.9999 → 10,000 step window (needs 100K+ steps to converge)
- decay=0.999 → 1,000 step window (good for 10K-50K steps)
- decay=0.99 → 100 step window (good for < 5K steps)

## Alternatives Considered
1. **Keep 0.9999, train longer** — Would need 100K+ steps. Not practical for quick iteration.
2. **Disable EMA entirely** — Training model samples looked good but EMA is needed for stable sampling.
3. **Adaptive EMA decay** — Increase decay over training. More complex, unclear benefit.

## Consequences
- EMA weights converge faster, matching training model quality within ~1K steps.
- For very long training runs (100K+ steps), the slightly faster EMA may introduce minor noise. Can be overridden via config for those cases.
- The config default is now 0.999. Users training to 100K+ steps should consider 0.9999.
