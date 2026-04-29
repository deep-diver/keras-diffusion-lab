"""Base classes and shared building blocks for diffusion methods.

Provides the shared infrastructure that all methods (unconditional, CFG,
pruning, distillation) build upon: model building blocks, trainer base
class with EMA/checkpointing, and sampler base class.
"""

from diffusion_harness.base.models import (
    sinusoidal_time_embedding,
    ResBlock,
    SelfAttention,
    Downsample,
    Upsample,
    build_unet,
)
from diffusion_harness.base.training import BaseTrainer
from diffusion_harness.base.sampling import (
    BaseSampler,
    save_image_grid,
    save_annotated_grid,
)
