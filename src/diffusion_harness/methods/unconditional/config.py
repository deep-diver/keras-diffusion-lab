"""Default configuration for unconditional DDPM."""

UNCONDITIONAL_DEFAULTS = {
    "method": "unconditional",
    "base_filters": 128,
    "num_levels": 3,
    "channel_multipliers": (1, 2, 2),
    "attention_resolutions": (0,),
    "ema_decay": 0.999,
    "learning_rate": 2e-4,
    "num_timesteps": 1000,
    "schedule_type": "linear",
    "batch_size": 64,
    "checkpoint_every": 500,
    "sample_every": 250,
}
