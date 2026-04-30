"""Class-conditional U-Net model.

Extends the base U-Net with a class embedding that is added to the
timestep embedding before FiLM conditioning. Supports a null class
for classifier-free guidance training.
"""

import keras
from keras import layers
import keras.ops as ops

from diffusion_harness.base.models import (
    sinusoidal_time_embedding,
    ResBlock,
    SelfAttention,
    Downsample,
    Upsample,
)


class ClassEmbedding(keras.layers.Layer):
    """Learnable class embedding with a null slot for CFG."""

    def __init__(self, num_classes, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        # Slot 0..num_classes-1 = real classes, slot num_classes = null
        self.embedding = layers.Embedding(num_classes + 1, embedding_dim)

    def call(self, class_ids):
        return self.embedding(class_ids)


def build_cond_unet(image_size=28, channels=1, base_filters=128,
                    num_levels=4, channel_multipliers=(1, 2, 2, 2),
                    attention_resolutions=(1, 2), num_classes=10):
    """Build a class-conditional DDPM U-Net denoiser.

    Same architecture as build_unet but accepts an additional class_id
    input. The class embedding is added to the time embedding, so all
    ResBlocks receive combined class+time conditioning via FiLM.

    Args:
        image_size: Input image spatial size.
        channels: Input/output channels.
        base_filters: Base filter count.
        num_levels: Number of resolution levels.
        channel_multipliers: Per-level channel multiplier.
        attention_resolutions: Which levels get self-attention.
        num_classes: Number of conditional classes.

    Returns:
        A Keras model that takes [noisy_image, timestep, class_id] and
        outputs predicted noise.
    """
    if len(channel_multipliers) < num_levels:
        channel_multipliers = channel_multipliers + (
            channel_multipliers[-1],) * (num_levels - len(channel_multipliers))

    time_dim = base_filters * 4

    noisy_img = layers.Input(shape=(image_size, image_size, channels), name="noisy_image")
    timestep = layers.Input(shape=(), dtype="int32", name="timestep")
    class_id = layers.Input(shape=(), dtype="int32", name="class_id")

    # Time embedding
    t_emb = sinusoidal_time_embedding(timestep, time_dim)
    t_mlp = layers.Dense(time_dim, activation="silu")(t_emb)
    t_mlp = layers.Dense(time_dim)(t_mlp)

    # Class embedding — add to time embedding
    c_emb = ClassEmbedding(num_classes, time_dim, name="class_embedding")(class_id)
    t_mlp = t_mlp + c_emb  # Combined conditioning

    # Entry conv
    h = layers.Conv2D(base_filters * channel_multipliers[0], 3, padding="same")(noisy_img)

    # Encoder
    skips = []
    for level in range(num_levels):
        filters = base_filters * channel_multipliers[level]
        h = ResBlock(filters)(h, t_mlp)
        h = ResBlock(filters)(h, t_mlp)

        if level in attention_resolutions:
            h = SelfAttention()(h)

        skips.append(h)

        if level < num_levels - 1:
            h = Downsample()(h)

    # Bottleneck
    bn_filters = base_filters * channel_multipliers[-1]
    h = ResBlock(bn_filters)(h, t_mlp)
    h = SelfAttention()(h)
    h = ResBlock(bn_filters)(h, t_mlp)

    # Decoder
    for level in reversed(range(num_levels)):
        filters = base_filters * channel_multipliers[level]
        skip = skips[level]
        h = layers.Concatenate()([h, skip])
        h = ResBlock(filters)(h, t_mlp)
        h = ResBlock(filters)(h, t_mlp)

        if level in attention_resolutions:
            h = SelfAttention()(h)

        if level > 0:
            h = Upsample(base_filters * channel_multipliers[level - 1])(h)

    # Output
    h = layers.GroupNormalization(8)(h)
    h = ops.silu(h)
    output = layers.Conv2D(channels, 3, padding="same")(h)

    model = keras.Model(
        inputs=[noisy_img, timestep, class_id],
        outputs=output,
        name="cond_ddpm_unet",
    )
    return model
