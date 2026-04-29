"""Core configuration for the diffusion harness.

Provides a single config builder that bundles all hyperparameters,
schedule, and dataset info into one dict. No global state.
"""

from diffusion_harness.schedules import linear_beta_schedule, cosine_beta_schedule, compute_schedule
from diffusion_harness.data import get_dataset_info


def make_config(
    # Dataset
    dataset: str = "fashion_mnist",
    # Method
    method: str = "unconditional",
    # Architecture
    base_filters: int = 128,
    num_levels: int = 3,
    channel_multipliers: tuple = (1, 2, 2),
    attention_resolutions: tuple = (0,),
    # Diffusion
    num_timesteps: int = 1000,
    schedule_type: str = "linear",
    # Training
    batch_size: int = 64,
    learning_rate: float = 2e-4,
    num_train_steps: int = 5000,
    ema_decay: float = 0.999,
    # Class-conditional (CFG) parameters
    num_classes: int = None,
    class_dropout_prob: float = 0.1,
    guidance_scale: float = 3.0,
    # Checkpointing & monitoring
    checkpoint_every: int = 500,
    sample_every: int = 250,
    num_samples: int = 8,
    snapshot_seed: int = 123,
    log_loss_every: int = 10,
    log_health_every: int = 100,
    upload_events_every: int = 100,
    # Overrides
    **overrides,
) -> dict:
    """Build a complete training configuration dict.

    All parameters have sensible defaults for a standard DDPM setup.
    Pass overrides as kwargs to customize any field.
    """
    ds_info = get_dataset_info(dataset)

    # Build noise schedule
    if schedule_type == "linear":
        betas = linear_beta_schedule(num_timesteps)
    elif schedule_type == "cosine":
        betas = cosine_beta_schedule(num_timesteps)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    schedule = compute_schedule(betas)

    config = {
        # Dataset
        "dataset": dataset,
        "image_size": ds_info["image_size"],
        "image_channels": ds_info["channels"],
        # Method
        "method": method,
        # Architecture
        "base_filters": base_filters,
        "num_levels": num_levels,
        "channel_multipliers": channel_multipliers,
        "attention_resolutions": attention_resolutions,
        # Diffusion
        "num_timesteps": num_timesteps,
        "schedule_type": schedule_type,
        "schedule": schedule,
        # Training
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_train_steps": num_train_steps,
        "ema_decay": ema_decay,
        # Class-conditional (CFG)
        "num_classes": num_classes or ds_info.get("num_classes", 10),
        "class_dropout_prob": class_dropout_prob,
        "guidance_scale": guidance_scale,
        # Monitoring
        "checkpoint_every": checkpoint_every,
        "sample_every": sample_every,
        "num_samples": num_samples,
        "snapshot_seed": snapshot_seed,
        "log_loss_every": log_loss_every,
        "log_health_every": log_health_every,
        "upload_events_every": upload_events_every,
    }

    config.update(overrides)
    return config
