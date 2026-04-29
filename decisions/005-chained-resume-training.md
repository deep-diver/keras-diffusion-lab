# Decision 005: Chained Resume Training for Long Runs

## Status
Accepted

## Date
2026-04-29

## Context
Keras Kinetic enforces a 1-hour timeout on GKE jobs. At ~0.5 steps/second with batch_size=64 on TPU v5litepod-4, each job covers ~1,800 steps. To train for 15K steps, we need ~8 hours of TPU time. The training harness already supports `--resume` which downloads the latest GCS checkpoint (model + EMA + optimizer state) and continues training.

## Decision
Chain multiple `--resume` jobs to reach target step counts. Each job runs for ~1 hour (~1,800 steps), then a new job is manually submitted with `--resume` to continue from the latest checkpoint.

For the CFG experiment (6,000 → 14,900 steps), we chained 7 jobs:
```
Job 1 (initial):  0 → 5,000
Job 2 (resume):   5,000 → 6,500
Job 3 (resume):   6,500 → 7,500
Job 4 (resume):   7,500 → 9,000
Job 5 (resume):   9,000 → 10,500
Job 6 (resume):   10,500 → 13,500
Job 7 (resume):   13,500 → 14,900
```

## Alternatives Considered
1. **Manual chaining (chosen)** — Submit resume jobs one at a time. Simple, predictable. Con: requires manual monitoring.
2. **Auto-retry wrapper script** — Script that detects timeout and auto-resubmits. Pro: no manual intervention. Con: adds complexity; hard to debug if something goes wrong mid-chain.
3. **Request Kinetic timeout extension** — Not currently supported; 1-hour limit is a platform constraint.
4. **Use smaller model/fewer steps** — Avoid the problem by training faster. Con: compromises research goals.

## Consequences
- Each resume boundary causes a temporary quality regression (~500-1000 steps) due to optimizer momentum reset. The EMA weights absorb some of this, but the sample quality at steps 5,500-6,500 was noticeably worse than at step 5,000.
- The final model at step 14,900 has worse distribution stats (mean=-0.280) than the step 5,000 checkpoint (mean=-0.404) did. Chained training is *not* equivalent to continuous training.
- GCS checkpoint persistence is critical — without it, each timeout would lose all progress. The checkpoint format (model + EMA + optimizer + state JSON) worked reliably across 7 job boundaries.
- **Recommendation**: For future long runs, implement automatic job chaining and evaluate whether saving/restoring full optimizer state eliminates the quality regression.
