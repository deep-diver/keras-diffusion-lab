# Decision 003: Classifier-Free Guidance for Class-Conditional Generation

## Status
Accepted

## Date
2026-04-23

## Context
The baseline diffusion model generates unconditional Fashion-MNIST samples — random classes with no user control. For practical use and research, we need class-conditional generation: given a class label (e.g., "sneaker"), generate an image of that class. Two main approaches exist: classifier guidance (training a separate classifier on noisy data) and classifier-free guidance (CFG, Ho & Salimans 2022), which trains a single model with class conditioning and random class dropout.

## Decision
Implement Classifier-Free Guidance (CFG) as the first class-conditional method.

Architecture:
- **Class embedding**: A learnable embedding (num_classes + 1 slots, last = null/unconditional) added to the time embedding before FiLM injection into ResBlocks.
- **Training**: Random class dropout with p=0.1 replaces the class label with the null token. This teaches the model both conditional and unconditional denoising in a single network.
- **Sampling**: Guided prediction: `eps = (1 + w) * eps_cond - w * eps_uncond`, where w is the guidance scale (default 3.0).

## Alternatives Considered
1. **Classifier guidance** — Train a separate noise-aware classifier. Pro: can vary guidance strength post-training. Con: requires training and storing a second model; classifier must handle all noise levels.
2. **CFG (chosen)** — Single model, joint conditional/unconditional training. Pro: simpler, no extra model, widely adopted in modern diffusion models (DALL-E 2, Imagen). Con: requires two forward passes per sampling step.
3. **Conditioning via cross-attention** — Class embedding injected via cross-attention layers rather than additive to time embedding. Pro: more expressive for rich text conditions. Con: overkill for single-class-token conditioning; adds complexity.

## Consequences
- The `build_cond_unet()` takes 3 inputs: `[noisy_image, timestep, class_id]` vs `build_unet()`'s 2 inputs.
- Sampling requires two forward passes per step (conditional + unconditional), roughly doubling sampling time.
- Guidance scale w=1.0 recovers unconditional sampling. w=3-7.5 produces strong class adherence. Very high w (>10) may cause artifacts.
- This pattern generalizes to future text conditioning by replacing the class embedding with a text encoder output.
