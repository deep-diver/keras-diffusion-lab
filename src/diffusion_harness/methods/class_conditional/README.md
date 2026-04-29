# Class-Conditional DDPM with Classifier-Free Guidance

Generate images conditioned on class labels using CFG (Ho & Salimans, 2022).

## How It Works

1. **Training**: The U-Net takes `[noisy_image, timestep, class_id]`. With probability `class_dropout_prob` (default 10%), the class label is replaced with a null token. This teaches the model to denoise with and without class information.

2. **Sampling**: At each step, the model makes two predictions:
   - **Conditioned**: `eps_cond = model(x_t, t, target_class)`
   - **Unconditioned**: `eps_uncond = model(x_t, t, null_class)`
   - **Guided**: `eps = (1 + w) * eps_cond - w * eps_uncond`

   The guidance scale `w` controls how strongly the output follows the class condition.

## Usage

```bash
python remote_train.py --method class_conditional --dataset fashion_mnist --steps 5000
```

## Guidance Scale Guide

| Scale (w) | Effect |
|-----------|--------|
| 0 | Ignores class (same as unconditional) |
| 1 | Normal conditional sampling |
| 3-5 | Strong class adherence (recommended) |
| 7-10 | Very aggressive, may reduce diversity |

## Fashion-MNIST Class Names

| ID | Name |
|----|------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Reference

Ho, J., & Salimans, T. (2022). "Classifier-Free Diffusion Guidance." NeurIPS 2022 Workshop.
