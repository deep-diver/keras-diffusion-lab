"""Default configuration for class-conditional DDPM with CFG."""

CLASS_CONDITIONAL_DEFAULTS = {
    "method": "class_conditional",
    "base_filters": 128,
    "num_levels": 3,
    "channel_multipliers": (1, 2, 2, 2),
    "attention_resolutions": (1, 2),
    "ema_decay": 0.999,
    "learning_rate": 2e-4,
    "num_timesteps": 1000,
    "schedule_type": "linear",
    "batch_size": 64,
    "checkpoint_every": 500,
    "sample_every": 250,
    # CFG-specific
    "num_classes": 10,
    "class_dropout_prob": 0.1,
    "guidance_scale": 3.0,
}
