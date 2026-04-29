# Unconditional DDPM

Standard Denoising Diffusion Probabilistic Model (Ho et al. 2020) with no conditioning.

## Architecture
- U-Net with FiLM timestep conditioning
- Self-attention at configurable resolutions
- Strided conv downsampling, transposed conv upsampling
- EMA weight averaging for stable sampling

## Training
- Epsilon-prediction MSE loss
- Adam optimizer (lr=2e-4)
- Linear beta schedule (1e-4 to 0.02)

## Sampling
- Full 1000-step DDPM reverse process
- Uses EMA model weights
