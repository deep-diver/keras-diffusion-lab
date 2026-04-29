# Decision 002: 3-Level U-Net for 28x28 Images

## Status
Accepted

## Date
2026-04-23

## Context
Fashion-MNIST and MNIST images are 28x28. The canonical DDPM uses a 4-level U-Net designed for 32x32 images (CIFAR-10). With 4 levels on 28x28, the spatial dimensions go: 28→14→7→3 (odd), causing the decoder's transposed convolutions to produce different spatial sizes (6x6, 12x12) than the encoder's skip connections (7x7, 14x14). This resulted in a Concatenate shape mismatch.

## Decision
Use `num_levels=3` for 28x28 images. The spatial progression is 28→14→7 (clean halving at every level).

## Alternatives Considered
1. **4 levels with padding** — Pad images to 32x32. Pro: matches canonical. Con: changes data distribution, wasteful.
2. **4 levels with odd-size handling** — Use resizing or cropping in skip connections. Con: introduces artifacts, adds complexity.
3. **3 levels (chosen)** — Clean, simple, works for 28x28. Pro: no hacks needed. Con: slightly less capacity than 4 levels.

## Consequences
- All 28x28 datasets (Fashion-MNIST, MNIST) must use `--num-levels 3`.
- CIFAR-10 (32x32) can use `--num-levels 4` without issues.
- The model has 3 resolution levels instead of 4, but still produces good results (~21M params with base_filters=128).
