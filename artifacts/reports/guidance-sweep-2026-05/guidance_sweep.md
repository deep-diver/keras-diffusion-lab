# Guidance Scale Sweep: The Trade-Off Between Distribution Quality and Class Correctness

> An unexpected finding: higher guidance scale improves distribution matching (lower FID) but **destroys classification accuracy** — revealing that our conditional model's class conditioning is weaker than expected.

**Date**: May 2026
**Checkpoint**: Conditional DDPM at step 30,000
**Sampling**: DDIM-50 (deterministic, 50 steps)
**Evaluation**: FID + classification accuracy (trained Fashion-MNIST classifier, 91.52% on real data)
**Samples**: 10 per class × 10 classes = 100 per guidance scale

---

## The Question

We trained the conditional model with CFG at a fixed guidance scale of w=3.0. But is that the right choice? The guidance scale controls how strongly the model pushes toward the specified class:

```
guided_noise = (1 + w) × eps_conditional - w × eps_unconditional
```

- w=1: Standard conditional (no boost)
- w=3: Our default (moderate push)
- w=5: Strong push
- w=7.5: Aggressive push (Stable Diffusion default)

We evaluate w ∈ {1.0, 3.0, 5.0, 7.5} on two metrics:
1. **FID** — how close is the generated distribution to real Fashion-MNIST?
2. **Classification accuracy** — does the model generate the *correct* class?

## Results

### Summary

| Guidance Scale (w) | FID (lower=better) | Accuracy | Trend |
|---------------------|---------------------|----------|-------|
| 1.0 | 93.41 | 29.0% | Best accuracy, worst FID |
| **3.0** | **69.65** | **10.0%** | **Best FID, chance accuracy** |
| 5.0 | 75.29 | 1.0% | FID degrades, accuracy collapses |
| 7.5 | 81.04 | 0.0% | All accuracy lost |

### FID vs Guidance Scale

![Sweep Summary](sweep_summary.png)

FID improves from w=1.0 (93.41) to w=3.0 (69.65), then degrades at higher scales. This matches the expected U-shaped curve — too little guidance produces incoherent outputs, too much produces artifacts.

### The Surprising Result: Accuracy Drops with Guidance

Classification accuracy **decreases monotonically** with increasing guidance scale:
- w=1.0: 29% (above chance level of 10%)
- w=3.0: 10% (exactly chance level)
- w=5.0: 1% (near zero)
- w=7.5: 0% (zero — no sample classified correctly)

This is the opposite of what we expected. Higher guidance should produce *more class-specific* outputs. Instead, it produces outputs that the classifier can't recognize as any Fashion-MNIST class.

### Per-Class Accuracy Heatmap

![Per-class Heatmap](per_class_heatmap.png)

### Per-Class Breakdown at w=1.0

The only guidance scale with above-chance accuracy:

| Class | Accuracy | Note |
|-------|----------|------|
| Shirt | 90% | Most reliably generated |
| Pullover | 80% | Good |
| Bag | 60% | Decent |
| Trouser | 30% | Weak |
| Coat | 30% | Weak |
| T-shirt | 0% | Not recognized |
| Dress | 0% | Not recognized |
| Sandal | 0% | Not recognized |
| Sneaker | 0% | Not recognized |
| Ankle boot | 0% | Not recognized |

Only 5 of 10 classes are generated with any class correctness. The model struggles particularly with footwear (Sandal, Sneaker, Ankle boot) and specific clothing items (T-shirt, Dress).

## Analysis: Why Does Higher Guidance Reduce Accuracy?

Three possible explanations:

### 1. Guidance Amplifies Model Errors

At w=3.0, the guided prediction is: `4 × eps_cond - 3 × eps_uncond`. If the model's conditional prediction is slightly wrong, the 4x amplification makes the error dominant. At w=7.5, it's `8.5 × eps_cond - 7.5 × eps_uncond` — any error is massively amplified.

This is the "undertrained model + high w" effect documented in the literature: high guidance requires a well-calibrated conditional/unconditional difference. Our model at 30K steps may not have a sufficiently accurate conditional prediction.

### 2. DDIM-50 May Not Work Well with High Guidance

This sweep used DDIM with 50 steps instead of DDPM with 1000 steps. The guidance effect may require more reverse steps to properly manifest — at only 50 steps, the accumulated guidance push may overshoot and create artifacts.

The DDIM comparison report showed that DDIM-50 produces good overall quality (low FID). But high guidance + few steps may be a problematic combination that needs investigation.

### 3. The Classifier Doesn't Generalize to Guided Images

The classifier was trained on real Fashion-MNIST images. Heavily guided outputs may have a different "style" — sharper edges, more contrast, unusual texture patterns — that the classifier hasn't seen. The classifier may be unable to recognize class features in these stylized outputs.

However, this doesn't explain the w=1.0 result (29% accuracy) being above chance — at w=1.0, the outputs should be closest to natural images.

## What This Means

### For Our Model

1. **The class conditioning is weak.** At w=1.0 (standard conditional, no boost), only 29% of generated images are classified correctly. This means the model hasn't fully learned the class-to-image mapping — it generates images that are "in the right neighborhood" but not distinctly the correct class.

2. **w=3.0 produces the best-looking images (lowest FID) but not the correct classes.** The guidance scale is optimizing for visual quality (distribution match) at the expense of class correctness.

3. **The model may be relying on the unconditional pathway.** At w=3.0, the guided prediction is `4 × cond - 3 × uncond`. If cond ≈ uncond (model doesn't differentiate), the result is approximately just the unconditional prediction. FID would be good (unconditional model generates reasonable images) but accuracy would be at chance (no class signal).

### For Research

This result raises important questions:
- Is 30K steps enough for strong class conditioning?
- Would longer training (50K, 100K steps) improve the conditional/unconditional gap?
- Does DDIM-50 vs DDPM-1000 affect the guidance scale tradeoff?
- Would a guidance scale schedule (lower w early, higher w late in sampling) help?

## Sample Grids

| w=1.0 | w=3.0 | w=5.0 | w=7.5 |
|-------|-------|-------|-------|
| ![w1](samples_w1.0.png) | ![w3](samples_w3.0.png) | ![w5](samples_w5.0.png) | ![w7.5](samples_w7.5.png) |

## Limitations

1. **Small sample size (10/class).** Both FID and accuracy estimates have high variance with n=10. The FID computation issued a singular matrix warning. Results should be confirmed with 100+ samples per class.

2. **DDIM-50 instead of DDPM-1000.** The sweep used DDIM for speed, which may interact differently with guidance than the DDPM sampling the model was trained with.

3. **Classifier accuracy ceiling (91.52%).** Even perfect generated images would only achieve ~91.5% accuracy. But 29% at w=1.0 is well below this ceiling, indicating genuine model weakness.

4. **No comparison with unconditional baseline.** We didn't evaluate the unconditional model's FID with the same sample count (100 images) for direct comparison.

5. **Single checkpoint.** Only evaluated at step 30K. The guidance-quality tradeoff may differ at earlier or later training stages.

## Files

| File | Description |
|------|-------------|
| `scripts/guidance_sweep.py` | Full DDPM-based sweep script (slow) |
| `artifacts/guidance_sweep/` | Sweep results (CSV, npz, images, plots) |
| `artifacts/metrics/` | Classifier weights + real data statistics |

## References

1. Ho, J., & Salimans, T. (2022). "Classifier-Free Diffusion Guidance." NeurIPS 2021 Workshop.
2. Dhariwal, P., & Nichol, A. (2021). "Diffusion Models Beat GANs on Image Synthesis." NeurIPS 2021.
3. Song, J., Meng, C., & Ermon, S. (2021). "Denoising Diffusion Implicit Models." ICLR 2021.
