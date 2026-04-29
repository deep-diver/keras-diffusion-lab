"""Standard DDPM U-Net model.

Implements the canonical denoising architecture from Ho et al. 2020
with FiLM-style timestep conditioning and self-attention at specified
resolutions. Configurable depth, width, and attention placement.

These building blocks are shared across all diffusion methods.

References:
  - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
  - Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021
"""

import keras
from keras import layers
import keras.ops as ops


def sinusoidal_time_embedding(t, dim):
    """Standard sinusoidal positional embedding for timesteps."""
    half_dim = dim // 2
    emb = ops.log(ops.cast(10000.0, "float32")) / (half_dim - 1)
    emb = ops.exp(ops.cast(ops.arange(half_dim), "float32") * -emb)
    emb = ops.cast(t, "float32")[:, None] * emb[None, :]
    emb = ops.concatenate([ops.sin(emb), ops.cos(emb)], axis=-1)
    if dim % 2 == 1:
        emb = ops.pad(emb, [[0, 0], [0, 1]])
    return emb


class ResBlock(keras.layers.Layer):
    """Residual block with FiLM timestep conditioning."""

    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.conv1 = layers.Conv2D(self.out_channels, 3, padding="same")
        self.conv2 = layers.Conv2D(self.out_channels, 3, padding="same")
        self.norm1 = layers.GroupNormalization(8)
        self.norm2 = layers.GroupNormalization(8)
        self.time_proj = layers.Dense(self.out_channels * 2)
        self.skip = (
            layers.Conv2D(self.out_channels, 1, padding="same")
            if in_channels != self.out_channels else None
        )

    def call(self, x, t_emb=None):
        h = self.norm1(x)
        h = ops.silu(h)
        h = self.conv1(h)

        if t_emb is not None:
            tp = self.time_proj(ops.silu(t_emb))
            scale, shift = ops.split(tp, 2, axis=-1)
            h = h * (1 + scale[:, None, None, :]) + shift[:, None, None, :]

        h = self.norm2(h)
        h = ops.silu(h)
        h = self.conv2(h)

        if self.skip is not None:
            x = self.skip(x)
        return x + h


class SelfAttention(keras.layers.Layer):
    """Single-head spatial self-attention with residual connection."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-1]
        self.norm = layers.GroupNormalization(8)
        self.qkv = layers.Conv2D(channels * 3, 1, padding="same")
        self.proj = layers.Conv2D(channels, 1, padding="same")
        self.channels = channels

    def call(self, x):
        shape = ops.shape(x)
        B, H, W = shape[0], shape[1], shape[2]
        C = self.channels

        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = ops.split(qkv, 3, axis=-1)

        q = ops.reshape(q, (B, H * W, C))
        k = ops.reshape(k, (B, H * W, C))
        v = ops.reshape(v, (B, H * W, C))

        attn = ops.matmul(q, ops.transpose(k, (0, 2, 1))) / ops.sqrt(ops.cast(C, "float32"))
        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.reshape(out, (B, H, W, C))
        return x + self.proj(out)


class Downsample(keras.layers.Layer):
    """Strided convolution downsampling."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.conv = layers.Conv2D(input_shape[-1], 3, strides=2, padding="same")

    def call(self, x):
        return self.conv(x)


class Upsample(keras.layers.Layer):
    """Transposed convolution upsampling."""

    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels

    def build(self, input_shape):
        self.conv = layers.Conv2DTranspose(self.out_channels, 3, strides=2, padding="same")

    def call(self, x):
        return self.conv(x)


def build_unet(image_size=28, channels=1, base_filters=128,
               num_levels=4, channel_multipliers=(1, 2, 2, 2),
               attention_resolutions=(1, 2)):
    """Build a standard DDPM U-Net denoiser.

    Args:
        image_size: Input image spatial size.
        channels: Input/output channels.
        base_filters: Base filter count (multiplied by channel_multipliers).
        num_levels: Number of resolution levels.
        channel_multipliers: Per-level channel multiplier.
        attention_resolutions: Which levels get self-attention (0-indexed).

    Returns:
        A Keras model that takes [noisy_image, timestep] and outputs
        predicted noise (same shape as input).
    """
    if len(channel_multipliers) < num_levels:
        channel_multipliers = channel_multipliers + (
            channel_multipliers[-1],) * (num_levels - len(channel_multipliers))

    time_dim = base_filters * 4

    noisy_img = layers.Input(shape=(image_size, image_size, channels), name="noisy_image")
    timestep = layers.Input(shape=(), dtype="int32", name="timestep")

    # Time embedding
    t_emb = sinusoidal_time_embedding(timestep, time_dim)
    t_mlp = layers.Dense(time_dim, activation="silu")(t_emb)
    t_mlp = layers.Dense(time_dim)(t_mlp)

    # Entry conv
    h = layers.Conv2D(base_filters * channel_multipliers[0], 3, padding="same")(noisy_img)

    # Encoder
    skips = []
    for level in range(num_levels):
        filters = base_filters * channel_multipliers[level]
        h = ResBlock(filters, name=f"enc_{level}_0")(h, t_mlp)
        h = ResBlock(filters, name=f"enc_{level}_1")(h, t_mlp)

        if level in attention_resolutions:
            h = SelfAttention(name=f"enc_attn_{level}")(h)

        skips.append(h)

        if level < num_levels - 1:
            h = Downsample(name=f"down_{level}")(h)

    # Bottleneck
    bn_filters = base_filters * channel_multipliers[-1]
    h = ResBlock(bn_filters, name="bn_0")(h, t_mlp)
    h = SelfAttention(name="bn_attn")(h)
    h = ResBlock(bn_filters, name="bn_1")(h, t_mlp)

    # Decoder
    for level in reversed(range(num_levels)):
        filters = base_filters * channel_multipliers[level]
        skip = skips[level]
        h = layers.Concatenate(name=f"cat_{level}")([h, skip])
        h = ResBlock(filters, name=f"dec_{level}_0")(h, t_mlp)
        h = ResBlock(filters, name=f"dec_{level}_1")(h, t_mlp)

        if level in attention_resolutions:
            h = SelfAttention(name=f"dec_attn_{level}")(h)

        if level > 0:
            h = Upsample(base_filters * channel_multipliers[level - 1],
                         name=f"up_{level}")(h)

    # Output
    h = layers.GroupNormalization(8)(h)
    h = ops.silu(h)
    output = layers.Conv2D(channels, 3, padding="same")(h)

    model = keras.Model(inputs=[noisy_img, timestep], outputs=output, name="ddpm_unet")
    return model
