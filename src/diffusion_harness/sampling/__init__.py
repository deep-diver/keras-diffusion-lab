"""DDPM reverse sampler: generate images by iterating the reverse process.

Re-exports from base.sampling and methods.unconditional.sampling
for backward compatibility.
"""

from diffusion_harness.base.sampling import (
    save_image_grid,
    save_annotated_grid,
)
from diffusion_harness.base.ddim_sampling import ddim_sample
from diffusion_harness.methods.unconditional.sampling import unconditional_sample as ddpm_sample
