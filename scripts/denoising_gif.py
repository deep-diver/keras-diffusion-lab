"""Generate a denoising GIF showing CFG reverse diffusion for different class labels.

Downloads the latest EMA checkpoint from GCS, builds the conditional model,
and runs the full reverse diffusion process saving intermediate frames.
Uses the library's _p_sample_step for correct denoising.
"""

import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import jax
import keras.ops as ops
from PIL import Image

from diffusion_harness.core import make_config
from diffusion_harness.methods.class_conditional.models import build_cond_unet
from diffusion_harness.base.sampling import _p_sample_step


CLASS_NAMES = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot",
}


def denoise_with_frames(model, config, class_id, num_classes, guidance_scale,
                        seed=42, num_frames=50):
    """Run reverse diffusion and capture intermediate frames.

    Uses the same denoising logic as the library's cfg_sample to ensure
    correct outputs.
    """
    schedule = config["schedule"]
    T = config["num_timesteps"]
    shape = (1, 28, 28, 1)
    null_class_id = num_classes

    # Store frames at evenly spaced timesteps
    frame_indices = set(np.linspace(T - 1, 0, num_frames).astype(int))

    # Initialize
    rng = jax.random.PRNGKey(seed)
    rng, subkey = jax.random.split(rng)
    x = np.array(jax.random.normal(subkey, shape).astype("float32"))

    class_ids = np.array([class_id], dtype="int32")

    frames = []

    # Reverse diffusion — same logic as CFGSampler.sample()
    for t_val in reversed(range(T)):
        t_batch = np.full((shape[0],), t_val, dtype="int32")

        x_tensor = ops.convert_to_tensor(x)
        t_tensor = ops.convert_to_tensor(t_batch)
        c_tensor = ops.convert_to_tensor(class_ids)

        # Conditional prediction
        eps_cond = model([x_tensor, t_tensor, c_tensor], training=False)
        eps_cond = np.array(eps_cond)

        # Unconditional prediction (null class)
        null_ids = ops.convert_to_tensor(
            np.full((shape[0],), null_class_id, dtype="int32")
        )
        eps_uncond = model([x_tensor, t_tensor, null_ids], training=False)
        eps_uncond = np.array(eps_uncond)

        # Guided interpolation
        eps_pred = (1.0 + guidance_scale) * eps_cond - guidance_scale * eps_uncond

        # Noise for this step
        rng, subkey = jax.random.split(rng)
        noise = np.array(jax.random.normal(subkey, shape).astype("float32"))
        if t_val == 0:
            noise = np.zeros_like(noise)

        x = _p_sample_step(x, eps_pred, t_batch, schedule, noise)

        # Save frame at specified timesteps
        if t_val in frame_indices:
            img = ((x[0] + 1.0) / 2.0 * 255).astype(np.uint8)
            frames.append(img)

    return frames


def make_gif(frames_list, labels, output_path, duration=80):
    """Create a GIF grid: single row of classes with text labels."""
    h, w = 28, 28
    n_classes = len(labels)
    n_frames = len(frames_list[0])

    # Layout: single row of images with class labels above each
    padding = 4
    label_h = 16  # height for class name label
    scale = 4     # upscale 28x28 -> 112x112 for visibility

    # Try to get a font, fall back to default
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        from PIL import ImageFont
        font = ImageFont.load_default()

    gif_frames = []
    for f_idx in range(n_frames):
        img_w = w * scale
        img_h = h * scale
        canvas_w = n_classes * (img_w + padding) + padding
        canvas_h = label_h + img_h + padding
        canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

        for class_idx, (frames, label) in enumerate(zip(frames_list, labels)):
            img = frames[f_idx][:, :, 0]  # (28, 28)
            img_big = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
            x_off = class_idx * (img_w + padding) + padding
            y_off = label_h
            canvas[y_off:y_off+img_h, x_off:x_off+img_w] = img_big

        pil_img = Image.fromarray(canvas, mode="L")

        # Draw class labels
        from PIL import ImageDraw
        draw = ImageDraw.Draw(pil_img)
        for class_idx, label in enumerate(labels):
            x_off = class_idx * (w * scale + padding) + padding
            draw.text((x_off + 2, 0), label, fill=0, font=font)

        gif_frames.append(pil_img)

    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"GIF saved: {output_path} ({n_frames} frames, {n_classes} classes)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to EMA weights .h5")
    parser.add_argument("--output", default="artifacts/cfg-run/denoising.gif")
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--num-frames", type=int, default=50)
    args = parser.parse_args()

    # Config — must match the training config exactly
    config = make_config(
        dataset="fashion_mnist", method="class_conditional",
        base_filters=128, num_levels=3,
        channel_multipliers=(1, 2, 2, 2), attention_resolutions=(1, 2),
        num_classes=10,
    )

    # Build model and load EMA weights
    print("Building model...")
    model = build_cond_unet(
        image_size=28, channels=1, base_filters=128, num_levels=3,
        channel_multipliers=(1, 2, 2, 2), attention_resolutions=(1, 2),
        num_classes=10,
    )
    # Build with dummy input
    dummy_x = np.zeros((1, 28, 28, 1), dtype="float32")
    dummy_t = np.zeros((1,), dtype="int32")
    dummy_c = np.zeros((1,), dtype="int32")
    model([dummy_x, dummy_t, dummy_c])

    print(f"Loading EMA weights from {args.checkpoint}...")
    model.load_weights(args.checkpoint)

    # All 10 Fashion-MNIST classes
    selected_classes = list(range(10))

    print(f"Generating denoising frames for {len(selected_classes)} classes...")
    all_frames = []
    for cls_id in selected_classes:
        print(f"  Class {cls_id}: {CLASS_NAMES[cls_id]}")
        frames = denoise_with_frames(
            model, config, cls_id, num_classes=10,
            guidance_scale=args.guidance_scale,
            seed=100 + cls_id,  # unique seed per class
            num_frames=args.num_frames,
        )
        all_frames.append(frames)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    make_gif(all_frames, [CLASS_NAMES[c] for c in selected_classes], args.output)


if __name__ == "__main__":
    main()
