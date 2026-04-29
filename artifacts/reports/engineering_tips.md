# Engineering Tips

Concise, actionable lessons from building and running a Keras 3 + JAX + Kinetic TPU training system.

---

## TPU / Kinetic

1. **Always use Spot instances.** 91% cheaper. `kinetic pool add --accelerator v5litepod-4 --spot`
2. **Kinetic 0.0.1 has a context bug.** Use `volumes={mount: kinetic.Data("./src/")}` to ship your code.
3. **Default timeout is 1 hour.** Design for it: split long runs into stages, upload artifacts early and often.
4. **Pre-built containers are faster.** `container_image="prebuilt"` skips Cloud Build.
5. **`kinetic down` may report errors even when it succeeds.** Verify with `gcloud`.
6. **Zone `us-west4-a`** has v5litepod-4 availability.

## Numerical Stability

7. **Always clip before log.** `np.maximum(x, 1e-20)` before `np.log()`. TPU bfloat16 will NaN otherwise.
8. **Validate schedule arrays** for NaN/Inf before training starts.
9. **Use float32 for schedule computation** even if the model runs in bfloat16.

## Training Patterns

10. **EMA is essential.** Without EMA, diffusion model samples are noise even after thousands of steps. Use decay=0.9999.
11. **Upload events on every checkpoint**, not just at the end. Jobs can timeout.
12. **Fixed-seed snapshots** let you visually compare model quality across training steps.
13. **Staged training** (warmup -> early -> mid -> late -> extended) works around timeout limits and provides natural go/no-go checkpoints.

## JAX + Keras 3

14. **Use `jax.value_and_grad`** with `has_aux=True` for gradient + metrics in one pass.
15. **Use `keras.utils.stateless_call`** to apply gradients without direct state mutation.
16. **Convert explicitly** between numpy and JAX: `np.array(jax_array)`, `ops.convert_to_tensor(np_array)`.
17. **Set `training=False`** for all inference/sampling calls.

## GCS Persistence

18. **GCS is your only persistence mechanism** with Kinetic. Everything on the worker dies with the job.
19. **Save checkpoint quartet:** model weights, EMA weights, optimizer state, training state JSON.
20. **Save events JSONL periodically** (every 100 steps) to prevent data loss.
21. **Save loss history as numpy** alongside events for quick plotting.
22. **Manifest JSON** tracks all snapshots with step/loss/path for easy lookup.

## Debugging

23. **First thing to check:** does the schedule have NaN/Inf?
24. **Second thing:** is EMA being used for sampling?
25. **Third thing:** is the context/package available on the remote worker? (Kinetic bug)
26. **Monitor gradient norms** — spikes > 5x average indicate instability.
27. **Track x0_hat statistics** (mean, std) — they reveal if the model is learning to denoise.
