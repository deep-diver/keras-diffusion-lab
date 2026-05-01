# FID Evaluation: Measuring Diffusion Model Quality with Classifier Features

> First quantitative quality assessment of the unconditional and conditional DDPM models using FID and classification accuracy — revealing what distribution statistics alone couldn't tell us.

**Date**: May 2026
**Models Evaluated**: Unconditional DDPM (step 30K), Conditional DDPM + CFG (step 30K)
**Feature Extractor**: Trained Fashion-MNIST CNN classifier (91.52% validation accuracy, Dense(128) features)
**Samples**: 8 per model (training snapshots)

---

## The Problem

After 30K training steps, we evaluated models using only **distribution statistics** (pixel mean, std, per-image diversity). These are poor quality proxies — a model could match real data statistics while generating blurry or incorrect images. We had no way to answer:

1. **How good are the images?** (FID — standard quality metric)
2. **Does CFG produce the correct class?** (Classification accuracy — direct correctness measure)

## Approach: Classifier-Based FID

Standard FID uses InceptionV3 features (299x299 RGB input). For Fashion-MNIST (28x28 grayscale), this requires a 10x upscale that introduces meaningless interpolation artifacts.

Instead, we train a domain-specific CNN classifier and use its penultimate layer (Dense, 128-dim) as the feature extractor. This gives features that are:
- **Meaningful** for the actual image domain (Fashion-MNIST items)
- **No upscale needed** — works directly on 28x28 grayscale
- **Complementary** — we get both FID (distribution quality) and classification accuracy (correctness)

### FID Formula

$$\text{FID} = \|\mu_{\text{real}} - \mu_{\text{gen}}\|^2 + \text{Tr}(\Sigma_{\text{real}} + \Sigma_{\text{gen}} - 2\sqrt{\Sigma_{\text{real}} \Sigma_{\text{gen}}})$$

Lower FID = generated distribution closer to real data distribution.

### Classifier Architecture

```
Input (28, 28, 1) [-1, 1]
  → (x + 1) / 2                     # rescale to [0, 1]
  → Conv2D(32, 3, relu) + MaxPool(2) # 14x14
  → Conv2D(64, 3, relu) + MaxPool(2) # 7x7
  → Flatten                          # 7*7*64 = 3136
  → Dense(128, relu)                 # ← FID feature layer
  → Dense(10, softmax)               # class probabilities
```

Training: 10 epochs, Adam, batch_size=128, 10% validation split.
**Validation accuracy: 91.52%** — strong enough to provide meaningful features and reliable class predictions.

## Results

### Quick FID Assessment (8 samples)

| Model | Step | FID | Note |
|-------|------|-----|------|
| Unconditional | 30,000 | 87.14 | 8 samples only |
| Conditional (CFG, w=3.0) | 30,000 | 131.41 | 8 samples only |

> **Important caveat**: FID with n=8 samples is unreliable. The covariance matrix is 128x128, but estimated from only 8 observations — far underdetermined (singular matrix warning during computation). These numbers should be treated as rough indicators, not definitive measurements. The guidance sweep report uses 10 samples/class (100 total) for more reliable FID.

### Classification Distribution (8 samples)

**Unconditional model** (no class control):
```
Class distribution: {0: 1, 1: 1, 5: 2, 6: 2, 7: 1, 9: 1}
```
8 samples cover 6 of 10 classes — roughly uniform distribution, as expected for an unconditional model.

**Conditional model** (CFG, w=3.0):
```
Class distribution: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}
```
8 samples from 8 different classes — each generated with the intended class label. The classifier recognized each as a distinct class, which is a basic sanity check that CFG is producing class-different outputs.

### What This Tells Us

1. **Both models produce recognizable Fashion-MNIST items** — the classifier can assign classes to generated samples rather than producing random predictions.

2. **FID is high even for "good" models** — this is expected with a domain-specific classifier. The 128-dim features are more discriminative than InceptionV3's 2048-dim features, so FID values are not comparable to published InceptionV3-based FID scores. They're only useful for relative comparisons within our own experiments.

3. **We cannot yet confirm CFG correctness** — the 8-sample test shows class differentiation, but not class *correctness*. We need the guidance sweep (100 samples/class with known labels) to measure accuracy.

## Limitations

1. **Small sample size** (n=8). The FID scores have high variance. The matrix square root computation issued a singularity warning. 100+ samples are needed for reliable FID.

2. **Domain-specific FID is not comparable to published work.** Our FID uses a Fashion-MNIST classifier's features, not InceptionV3. We cannot compare our FID scores to any published DDPM results. This metric is only useful for internal comparisons (between our models, or across guidance scales).

3. **Classifier accuracy ceiling**. At 91.52% accuracy, the classifier misclassifies ~8.5% of real images. This means the classification accuracy on generated images has an upper bound of ~91.5% — even if the generated images are perfect, the classifier will mislabel some of them. Per-class accuracy on generated samples should be interpreted relative to the classifier's own accuracy.

4. **No baseline comparison**. We don't have FID scores for the unconditional model at earlier checkpoints (1K, 5K, 10K) to show quality progression over training.

## Implementation

### Files

| File | Description |
|------|-------------|
| `src/diffusion_harness/metrics/__init__.py` | Re-exports |
| `src/diffusion_harness/metrics/classifier.py` | build_classifier(), train_classifier(), build_feature_extractor() |
| `src/diffusion_harness/metrics/fid.py` | extract_features(), compute_fid(), compute_fid_from_stats(), compute_classifier_accuracy() |
| `scripts/prepare_metrics.py` | One-time setup: train classifier, compute real feature statistics |
| `artifacts/metrics/` | Saved classifier weights + real data statistics |
| `tests/test_metrics.py` | 10 tests for classifier and FID |

### Usage

```bash
# One-time setup
KERAS_BACKEND=jax python scripts/prepare_metrics.py --dataset fashion_mnist

# Compute FID in code
from diffusion_harness.metrics import compute_fid, extract_features
from diffusion_harness.metrics import build_classifier, build_feature_extractor

classifier = build_classifier()
classifier.load_weights("artifacts/metrics/metrics_classifier.weights.h5")
feature_extractor = build_feature_extractor(classifier)

real_stats = np.load("artifacts/metrics/metrics_real_stats.npz")
features_gen = extract_features(generated_images, feature_extractor)
fid = compute_fid_from_stats(real_stats["mu"], real_stats["sigma"],
                             np.mean(features_gen, axis=0),
                             np.cov(features_gen, rowvar=False))
```

### Dependencies Added

- `scipy` — for `scipy.linalg.sqrtm` (matrix square root in FID computation)

## References

1. Heusel, M., et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a local Nash equilibrium." NeurIPS 2017. (Original FID paper)
2. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
3. Ho, J., & Salimans, T. (2022). "Classifier-Free Diffusion Guidance." NeurIPS 2021 Workshop.
