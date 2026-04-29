# Experiment [NUMBER]: [TITLE]

## Status
[Planned | Running | Completed | Failed]

## Date
[YYYY-MM-DD]

## Method
[e.g., unconditional, class_conditional]

## Objective
What question does this experiment answer?

## Configuration
```yaml
dataset: ...
method: ...
base_filters: ...
num_levels: ...
num_timesteps: ...
batch_size: ...
learning_rate: ...
ema_decay: ...
steps: ...
# Method-specific:
guidance_scale: ...
class_dropout_prob: ...
```

## Results
- Final loss:
- FID (if computed):
- Visual quality assessment:
- Key observations:

## Artifacts
- Checkpoint: `gs://...`
- Sample images: `...`
- Logs: `...`

## Decisions
What did we learn? What should change for the next experiment?

## Follow-ups
- [ ] Next experiment based on findings
