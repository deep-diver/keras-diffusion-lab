"""Generate Keras built-in architecture diagrams for both models.

Uses keras.utils.plot_model() to create visual graphs of:
  1. Unconditional U-Net (build_unet)
  2. Conditional U-Net (build_cond_unet)
"""

import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
from diffusion_harness.base.models import build_unet
from diffusion_harness.methods.class_conditional.models import build_cond_unet

REPORT_DIR_UNCOND = "artifacts/reports/fashion-mnist-diffusion-2026-04-23"
REPORT_DIR_COND = "artifacts/reports/class-conditional-fashionmnist-2026-04-23"


def main():
    os.makedirs(REPORT_DIR_UNCOND, exist_ok=True)
    os.makedirs(REPORT_DIR_COND, exist_ok=True)

    # ── 1. Unconditional U-Net ────────────────────────────────────────────
    print("Building unconditional U-Net...")
    model_uncond = build_unet(
        image_size=28, channels=1, base_filters=128, num_levels=3,
        channel_multipliers=(1, 2, 2), attention_resolutions=(0,),
    )

    # Build with dummy input
    dummy_x = np.zeros((1, 28, 28, 1), dtype="float32")
    dummy_t = np.zeros((1,), dtype="int32")
    model_uncond([dummy_x, dummy_t])

    print(f"  Parameters: {model_uncond.count_params():,}")
    print(f"  Layers: {len(model_uncond.layers)}")

    # Generate diagram
    path_uncond = os.path.join(REPORT_DIR_UNCOND, "architecture_keras.png")
    try:
        keras.utils.plot_model(
            model_uncond,
            to_file=path_uncond,
            show_shapes=True,
            show_layer_names=True,
            dpi=150,
            expand_nested=True,
        )
        print(f"  Saved: {path_uncond}")
    except Exception as e:
        print(f"  plot_model failed: {e}")
        print("  Trying with show_shapes=False...")
        try:
            keras.utils.plot_model(
                model_uncond,
                to_file=path_uncond,
                show_shapes=False,
                show_layer_names=True,
                dpi=150,
            )
            print(f"  Saved (no shapes): {path_uncond}")
        except Exception as e2:
            print(f"  Also failed: {e2}")

    # Also save a text summary
    summary_path = os.path.join(REPORT_DIR_UNCOND, "architecture_summary.txt")
    with open(summary_path, "w") as f:
        model_uncond.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"  Summary: {summary_path}")

    # ── 2. Conditional U-Net ───────────────────────────────────────────────
    print("\nBuilding conditional U-Net...")
    model_cond = build_cond_unet(
        image_size=28, channels=1, base_filters=128, num_levels=3,
        channel_multipliers=(1, 2, 2, 2), attention_resolutions=(1, 2),
        num_classes=10,
    )

    # Build with dummy input
    dummy_c = np.zeros((1,), dtype="int32")
    model_cond([dummy_x, dummy_t, dummy_c])

    print(f"  Parameters: {model_cond.count_params():,}")
    print(f"  Layers: {len(model_cond.layers)}")

    # Generate diagram
    path_cond = os.path.join(REPORT_DIR_COND, "architecture_keras.png")
    try:
        keras.utils.plot_model(
            model_cond,
            to_file=path_cond,
            show_shapes=True,
            show_layer_names=True,
            dpi=150,
            expand_nested=True,
        )
        print(f"  Saved: {path_cond}")
    except Exception as e:
        print(f"  plot_model failed: {e}")
        print("  Trying with show_shapes=False...")
        try:
            keras.utils.plot_model(
                model_cond,
                to_file=path_cond,
                show_shapes=False,
                show_layer_names=True,
                dpi=150,
            )
            print(f"  Saved (no shapes): {path_cond}")
        except Exception as e2:
            print(f"  Also failed: {e2}")

    # Save text summary
    summary_path = os.path.join(REPORT_DIR_COND, "architecture_summary.txt")
    with open(summary_path, "w") as f:
        model_cond.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"  Summary: {summary_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
