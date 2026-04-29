# Keras Kinetic TPU Field Notes

Hard-won operational knowledge from running real TPU training jobs with Keras 3 + JAX + Keras Kinetic. This document captures everything we learned empirically, not just from official docs.

---

## 1. Kinetic Context Packaging Bug (Critical)

**Problem:** Kinetic 0.0.1's `context.zip` captures `kinetic/core/` instead of the project source directory. Your package code will NOT be available on the remote worker.

**Workaround:** Use the `volumes` parameter to explicitly upload your source:
```python
src_data = kinetic.Data("./src/")

@kinetic.run(
    accelerator="v5litepod-4",
    volumes={"/tmp/keras_diffusion_src": src_data},
)
def remote_train():
    import sys
    src_mount = "/tmp/keras_diffusion_src"
    if os.path.isdir(src_mount) and src_mount not in sys.path:
        sys.path.insert(0, src_mount)
```

**Lesson:** Always verify that your package imports work inside the remote function before trusting a run. The local environment is NOT the same as the remote environment.

---

## 2. Kinetic Timeout Behavior

**Problem:** Kinetic has a default job timeout of 3600 seconds (1 hour). If your training exceeds this, the job is killed — even if checkpoints were saved.

**What we observed:**
- Stage B (1500 steps) completed training but timed out during final snapshot upload
- The checkpoint data was safe in GCS; only the final upload of events/loss was lost
- Subsequent stages had to be split into smaller chunks (1500-2500 steps each)

**Mitigation:**
- Stay well under 3600s per job. For v5litepod-4 at ~2.5 steps/s with batch=64, max ~3000 steps per job
- Upload events and loss history on every checkpoint save, not just at the end
- Design for idempotent resume: always check GCS for existing state before starting

---

## 3. GCS-Based Persistence Pattern

**Problem:** Kinetic has no built-in artifact persistence. Everything dies with the job.

**Solution we developed:** Use GCS as the persistence layer for ALL training state.

**GCS directory convention:**
```
gs://bucket/run_name/
  checkpoints/
    model_stepN.weights.h5
    ema_stepN.weights.h5
    optimizer_stepN.npz
    state_stepN.json
  snapshots/
    manifest.json
    samples_stepNNNNNN.npy
    samples_stepNNNNNN.png
  logs/
    events.jsonl
    loss_history.npy
  trajectories/
    trajectory_stepNNNNNN.png
```

**Key implementation details:**
- Upload checkpoint set (model + EMA + optimizer + state JSON) atomically on each save
- Upload events JSONL periodically (every 100 steps) to prevent data loss on timeout
- Loss history uploaded as numpy array on every checkpoint
- Manifest is a JSON list of snapshot entries, uploaded after each snapshot

**Resume flow:**
1. Find latest checkpoint via `find_latest_checkpoint()` (parses step numbers from filenames)
2. Download model + EMA + optimizer + state files to `/tmp/`
3. Load into trainer, get `start_step` from state JSON
4. Load existing events JSONL and loss history from GCS
5. Continue training from `start_step`

---

## 4. TPU Provisioning

**Spot vs On-demand:**
- Spot instances are up to 91% cheaper
- v5litepod-4 Spot is the sweet spot for small-scale diffusion research
- Spot can be preempted but this is rare for short jobs (< 1 hour)

**Pool vs One-off:**
```bash
# Preferred: Spot pool (persistent, auto-replaces preempted nodes)
kinetic pool add --accelerator v5litepod-4 --spot --project PROJ --zone us-west4-a

# One-off (creates/destroys per job)
kinetic up --accelerator v5litepod-4 --zone us-west4-a --spot -y
```

**Tear down:**
```bash
kinetic down --zone us-west4-a -y
```

**Important:** `kinetic down` may report "encountered an issue" even when the cluster was actually deleted. Verify with `gcloud container clusters list` or check for 404.

**Zone:** `us-west4-a` has v5litepod-4 availability.

---

## 5. BFloat16 Numerical Instability

**Problem:** TPU uses bfloat16 by default. Log-space arithmetic in the DDPM schedule can produce NaN due to `log(0)` or underflow.

**Solution:** All schedule computations must use numerically stable formulas:
- Use `np.maximum(x, 1e-20)` before `np.log()`
- Precompute log-variances with clipping: `posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))`
- Test schedule values for NaN/Inf before training starts

**Lesson:** Always validate that schedule arrays contain no NaN/Inf. This is the #1 cause of silent training failure on TPU.

---

## 6. JAX + Keras Interoperability

**Key patterns that work:**
- Use `jax.value_and_grad` for gradient computation inside `loss_fn` closures
- Use `keras.utils.stateless_call` to apply gradients without mutating model state directly
- Convert between numpy and JAX arrays explicitly: `np.array(jax_array)`, `ops.convert_to_tensor(numpy_array)`
- Set `training=False` for model inference calls to disable dropout/batch norm updates

**Pattern for a train step:**
```python
def train_step(self, batch):
    def loss_fn(trainable_vars, x_batch):
        # Forward pass + loss computation
        return loss_value, aux_data

    (loss, (grads, metrics)) = jax.value_and_grad(loss_fn, has_aux=True)(
        self.model.trainable_variables, batch
    )
    # Apply grads via stateless_call
    self.model = keras.utils.stateless_call(
        self.model, self.model.trainable_variables, grads,
        lambda m: m.optimizer.apply(gradients=grads)
    )
    return loss, metrics
```

---

## 7. EMA (Exponential Moving Average)

**Implementation:**
- After model build, snapshot all trainable variable values as numpy arrays
- After each gradient step, update shadow weights: `ema_w = decay * ema_w + (1 - decay) * w`
- For sampling, swap model weights with EMA weights, sample, then restore
- Save EMA weights alongside model weights in checkpoints

**Why it matters:** EMA dramatically improves sample quality in diffusion models. Without EMA, samples from a 9K-step model are pure noise. With EMA, recognizable structure appears.

**Decay:** 0.9999 is standard. This means ~10K steps before the EMA fully "warms up".

---

## 8. Container Image Optimization

**Problem:** Default Kinetic behavior builds a container via Cloud Build on every job submission. This adds 5-10 minutes of startup time.

**Solution:** Use a prebuilt container image:
```python
@kinetic.run(
    container_image="prebuilt",  # or a custom URI
)
```

**Lesson:** For iterative development, the build overhead is painful. Pre-built images are worth the setup cost.

---

## 9. Monitoring During Remote Training

**Pattern we developed:**
1. **EventLog** (JSONL) records loss, health, snapshot, and checkpoint events
2. Events are uploaded to GCS every N steps (configurable, we used 100)
3. Local `monitor_run.py` script can sync and parse events from GCS
4. Fixed-seed snapshots provide visual progress tracking across steps
5. Loss history as numpy array enables loss curve plotting locally

**Practical observation:** The event log is the single most useful artifact for debugging. It captures loss trajectory, gradient norms, x0_hat statistics, and NaN detection — all queryable after the fact.

---

## 10. Fixed-Seed Snapshot Pattern

**Pattern:** Pre-generate a fixed noise tensor at the start of training. At each snapshot interval, sample from the EMA model using this same noise. This gives a directly comparable visual progression.

```python
fixed_noise = jax.random.normal(PRNGKey(seed), shape)
# At each snapshot:
samples = sample(model, schedule, ..., initial_noise=fixed_noise)
```

**Why:** This shows the model's denoising quality improving over training steps without confounding from different random noise.

---

## 11. Staged Training Protocol

**Pattern:** Instead of one long run, split training into stages:
- warmup: 1K steps (validate architecture)
- early: 5K steps (validate convergence)
- mid: 20K steps (see structure)
- late: 100K steps (recognizable images)
- extended: 800K steps (convergence)

Each stage:
1. Resumes from the previous stage's GCS checkpoint
2. Runs a fixed number of steps
3. Evaluates decision criteria (loss trending, no NaN, gradient norm stable)
4. Prints a continuation recommendation

**Why:** Works around the 1-hour Kinetic timeout. Also provides natural checkpoints for cost/benefit decisions.

---

## 12. Common Failure Modes

| Failure | Cause | Fix |
|---------|-------|-----|
| NaN loss | Schedule log(0), bfloat16 underflow | Clip before log, use float32 for schedule |
| Timeout | Job > 3600s | Split into stages, upload events frequently |
| Missing package | Kinetic context bug | Use `volumes` workaround |
| Silent bad results | No EMA | Always use EMA for sampling |
| Lost artifacts | Job killed before upload | Upload on every checkpoint |
| High cost | On-demand TPU | Use Spot instances |
| Checkpoint not found | Wrong GCS path | Verify `list_blobs()` output |
