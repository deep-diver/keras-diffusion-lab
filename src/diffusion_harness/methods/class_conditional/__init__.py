"""Class-conditional DDPM with Classifier-Free Guidance (CFG).

Based on Ho & Salimans (2022). The denoiser accepts class labels
alongside timesteps. During training, class labels are randomly
dropped to a null class, teaching the model both conditioned and
unconditioned denoising. At inference, guided sampling interpolates
between the two for class-controllable generation.
"""

from diffusion_harness.methods.class_conditional.training import CFGTrainer
from diffusion_harness.methods.class_conditional.sampling import cfg_sample, CFGSampler
from diffusion_harness.methods.class_conditional.models import build_cond_unet
from diffusion_harness.methods.class_conditional.ddim_sampling import (
    cfg_ddim_sample,
    CFGDDIMSampler,
)


def build_model(config):
    """Build the class-conditional U-Net model."""
    return build_cond_unet(
        image_size=config["image_size"],
        channels=config["image_channels"],
        base_filters=config["base_filters"],
        num_levels=config["num_levels"],
        channel_multipliers=config["channel_multipliers"],
        attention_resolutions=config["attention_resolutions"],
        num_classes=config.get("num_classes", 10),
    )


def build_trainer(config):
    """Build the CFG trainer."""
    return CFGTrainer(config)


def build_sampler(model, config):
    """Build the CFG sampler."""
    return CFGSampler(
        model, config["schedule"], config["num_timesteps"],
        guidance_scale=config.get("guidance_scale", 3.0),
        num_classes=config.get("num_classes", 10),
    )
