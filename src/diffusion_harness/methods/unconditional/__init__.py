"""Unconditional DDPM method.

Standard epsilon-prediction DDPM (Ho et al. 2020) with no conditioning.
The baseline that other methods are compared against.
"""

from diffusion_harness.methods.unconditional.training import UnconditionalTrainer
from diffusion_harness.methods.unconditional.sampling import unconditional_sample
from diffusion_harness.base.models import build_unet


def build_model(config):
    """Build the unconditional U-Net model."""
    return build_unet(
        image_size=config["image_size"],
        channels=config["image_channels"],
        base_filters=config["base_filters"],
        num_levels=config["num_levels"],
        channel_multipliers=config["channel_multipliers"],
        attention_resolutions=config["attention_resolutions"],
    )


def build_trainer(config):
    """Build the unconditional DDPM trainer."""
    return UnconditionalTrainer(config)


def build_sampler(model, config):
    """Build the unconditional DDPM sampler."""
    from diffusion_harness.base.sampling import BaseSampler
    return BaseSampler(model, config["schedule"], config["num_timesteps"])
