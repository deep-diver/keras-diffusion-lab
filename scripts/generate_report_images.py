"""Generate all report images for the class-conditional blog post.

Produces:
  1. loss_curve.png — Full 15K loss progression with moving average
  2. architecture.png — Conditional U-Net architecture diagram
  3. real_vs_generated.png — Real Fashion-MNIST vs CFG-generated
  4. training_evolution.gif — Sample quality progression across steps
  5. milestone_samples.png — Key training checkpoint samples
  6. evolution_strip.png — Single-row evolution strip
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont

REPORT_DIR = "artifacts/reports/class-conditional-fashionmnist-2026-04-23"
SNAP_DIR = "artifacts/cfg-run/snapshots"
LOSS_PATH = "artifacts/cfg-run/loss_history.npy"

# NOTE: SNAP_DIR and LOSS_PATH require downloading training artifacts from GCS first:
#   python remote_train.py --gcs-bucket gs://gcp-ml-172005-ddpm-training/cfg-fashionmnist-5k/run01 --download-only
# Alternatively, download just the needed files:
#   gsutil cp gs://gcp-ml-172005-dddm-training/cfg-fashionmnist-5k/run01/logs/loss_history.npy artifacts/cfg-run/
#   gsutil -m cp gs://gcp-ml-172005-ddpm-training/cfg-fashionmnist-5k/run01/snapshots/*.npy artifacts/cfg-run/snapshots/


def get_font(size=12):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


# ─── 1. Loss Curve ────────────────────────────────────────────────────────────

def generate_loss_curve():
    loss = np.load(LOSS_PATH)
    steps = np.arange(1, len(loss) + 1)

    # 100-step moving average
    window = 100
    if len(loss) >= window:
        moving_avg = np.convolve(loss, np.ones(window) / window, mode='valid')
        avg_steps = steps[window - 1:]
    else:
        moving_avg = loss
        avg_steps = steps

    fig, ax = plt.subplots(figsize=(12, 5))

    # Raw loss (light)
    ax.plot(steps, loss, alpha=0.15, color='#4a90d9', linewidth=0.5, label='Raw loss')

    # Moving average (bold)
    ax.plot(avg_steps, moving_avg, color='#2c5f8a', linewidth=2, label='100-step average')

    # Mark key milestones
    milestones = {
        1000: ('EMA ckpt 1', 0.0371),
        5000: ('Best raw loss', 0.0260),
        6500: ('First timeout', None),
        9000: ('Resume complete', None),
        14800: ('Final EMA ckpt', None),
    }
    for step, (label, _) in milestones.items():
        if step <= len(loss):
            ax.axvline(x=step, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    # Annotate key points
    ax.annotate(f'Step 1000\nloss={loss[999]:.4f}',
                xy=(1000, loss[999]), xytext=(2000, 0.35),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, ha='center')
    ax.annotate(f'Step 5000\nloss={loss[4999]:.4f}',
                xy=(5000, loss[4999]), xytext=(6500, 0.20),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, ha='center')
    ax.annotate(f'Final (step {len(loss)})\nloss={loss[-1]:.4f}',
                xy=(len(loss), loss[-1]), xytext=(13000, 0.15),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, ha='center')

    # Mark resume boundaries (step ~5000, ~6500)
    ax.axvspan(5000, 5200, alpha=0.15, color='red', label='Resume disruptions')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('CFG Training Loss — Fashion-MNIST (15K Steps)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, len(loss) + 100)
    ax.set_ylim(0, min(0.5, np.percentile(loss, 99.5) * 1.2))
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "loss_curve.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {path}")
    return path


# ─── 2. Architecture Diagram ──────────────────────────────────────────────────

def generate_architecture():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color scheme
    c_input = '#E8F4FD'
    c_emb = '#FFF3CD'
    c_encoder = '#D4EDDA'
    c_bottleneck = '#F8D7DA'
    c_decoder = '#CCE5FF'
    c_output = '#E8F4FD'
    c_cfg = '#FFE0CC'

    def draw_box(x, y, w, h, label, color, fontsize=9, bold=False):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='#333', linewidth=1.2)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, family='monospace')

    def arrow(x1, y1, x2, y2, text=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
        if text:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.1, my, text, fontsize=7, color='#666', style='italic')

    # Title
    ax.text(7, 9.7, 'Conditional U-Net (build_cond_unet)', ha='center', fontsize=14, fontweight='bold')

    # ── Inputs (top)
    draw_box(0.5, 8.8, 3.0, 0.6, 'x_t (28x28x1)\nNoisy Image', c_input, bold=True)
    draw_box(4.5, 8.8, 2.5, 0.6, 't (scalar)\nTimestep', c_input, bold=True)
    draw_box(8.0, 8.8, 2.5, 0.6, 'c (int)\nClass ID', c_input, bold=True)
    draw_box(11.0, 8.8, 2.0, 0.6, 'Null (int)\nDropout 10%', c_cfg, fontsize=8)

    # ── Embeddings
    arrow(5.75, 8.8, 5.75, 8.3)
    draw_box(4.0, 7.6, 3.5, 0.6, 'Sinusoidal Time Embed\nDense(512) → SiLU → Dense(512)', c_emb, fontsize=8)

    arrow(9.25, 8.8, 9.25, 8.3)
    draw_box(8.0, 7.6, 2.5, 0.6, 'ClassEmbedding\nEmbed(11, 512)', c_emb, fontsize=8)

    # ── Additive fusion
    arrow(5.75, 7.6, 6.0, 7.1)
    arrow(9.25, 7.6, 7.5, 7.1)
    draw_box(5.0, 6.5, 3.5, 0.5, 't_emb = t_emb + c_emb\n(Additive Fusion)', '#FFE0CC', fontsize=8, bold=True)

    # ── Encoder
    arrow(7.0, 8.8, 7.0, 6.0)  # x_t down
    arrow(6.75, 6.5, 6.75, 6.0)  # combined emb down

    y_enc = 5.5
    draw_box(0.5, y_enc, 5.5, 0.5, 'Level 0: ResBlock×2 (28x28, 128ch) + Self-Attention', c_encoder, fontsize=8)
    arrow(3.25, y_enc, 3.25, y_enc - 0.15, 'Conv2D stride=2')
    draw_box(0.5, y_enc - 0.6, 5.5, 0.5, 'Level 1: ResBlock×2 (14x14, 256ch)', c_encoder, fontsize=8)
    arrow(3.25, y_enc - 0.6, 3.25, y_enc - 0.75, 'Conv2D stride=2')
    draw_box(0.5, y_enc - 1.2, 5.5, 0.5, 'Level 2: ResBlock×2 (7x7, 256ch)', c_encoder, fontsize=8)

    # Skip connections
    ax.annotate('', xy=(6.8, y_enc), xytext=(6.0, y_enc),
                arrowprops=dict(arrowstyle='->', color='green', lw=1, ls='--'))
    ax.annotate('', xy=(6.8, y_enc - 0.6), xytext=(6.0, y_enc - 0.6),
                arrowprops=dict(arrowstyle='->', color='green', lw=1, ls='--'))
    ax.annotate('', xy=(6.8, y_enc - 1.2), xytext=(6.0, y_enc - 1.2),
                arrowprops=dict(arrowstyle='->', color='green', lw=1, ls='--'))
    ax.text(7.0, y_enc + 0.15, 'skip', fontsize=7, color='green')
    ax.text(7.0, y_enc - 0.45, 'skip', fontsize=7, color='green')
    ax.text(7.0, y_enc - 1.05, 'skip', fontsize=7, color='green')

    # FiLM arrows from embedding
    for y in [y_enc + 0.25, y_enc - 0.35, y_enc - 0.95]:
        ax.annotate('', xy=(3.5, y), xytext=(5.0, 6.5),
                    arrowprops=dict(arrowstyle='->', color='#B8860B', lw=0.8, ls=':'))
    ax.text(4.2, 6.2, 'FiLM conditioning', fontsize=7, color='#B8860B', style='italic')

    # ── Bottleneck
    y_bn = y_enc - 1.9
    arrow(3.25, y_enc - 1.2, 3.25, y_bn + 0.4)
    draw_box(0.5, y_bn, 5.5, 0.45, 'Bottleneck: ResBlock×2 (7x7, 256ch)', c_bottleneck, fontsize=8)

    # ── Decoder
    y_dec = y_bn - 0.6
    arrow(3.25, y_bn, 3.25, y_dec + 0.45)
    draw_box(0.5, y_dec, 5.5, 0.45, 'Level 2: Upsample + ResBlock×2 (7x7 → 14x14, 256ch)', c_decoder, fontsize=8)
    arrow(3.25, y_dec, 3.25, y_dec - 0.55)
    draw_box(0.5, y_dec - 0.6, 5.5, 0.45, 'Level 1: Upsample + ResBlock×2 (14x14 → 28x28, 256→128ch)', c_decoder, fontsize=8)
    arrow(3.25, y_dec - 0.6, 3.25, y_dec - 1.15)
    draw_box(0.5, y_dec - 1.2, 5.5, 0.45, 'Level 0: Upsample + ResBlock×2 (28x28, 128ch)', c_decoder, fontsize=8)

    # ── Output
    arrow(3.25, y_dec - 1.2, 3.25, y_dec - 1.85)
    draw_box(1.5, y_dec - 2.2, 3.5, 0.4, 'Conv2D → eps_pred (28x28x1)', c_output, bold=True)

    # ── CFG box (sampling time)
    y_cfg = 0.3
    draw_box(8.5, y_cfg, 4.5, 3.0,
             'CFG Sampling (inference only)\n\n'
             'eps_cond = model([x_t, t, class])\n'
             'eps_uncond = model([x_t, t, null])\n'
             'eps = (1+w)*cond - w*uncond\n\n'
             'w=3.0 (guidance scale)\n'
             'null = num_classes (10)',
             c_cfg, fontsize=8, bold=False)

    # Legend
    legend_items = [
        mpatches.Patch(color=c_input, label='Input'),
        mpatches.Patch(color=c_emb, label='Embedding'),
        mpatches.Patch(color=c_encoder, label='Encoder'),
        mpatches.Patch(color=c_bottleneck, label='Bottleneck'),
        mpatches.Patch(color=c_decoder, label='Decoder'),
        mpatches.Patch(color=c_cfg, label='CFG (sampling)'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "architecture.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {path}")
    return path


# ─── 3. Real vs Generated ─────────────────────────────────────────────────────

def generate_real_vs_generated():
    import keras
    # Load real Fashion-MNIST
    (x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
    x_real = x_train.astype('float32') / 255.0

    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Load generated samples at step 14900
    gen_path = os.path.join(SNAP_DIR, 'samples_step014900.npy')
    if not os.path.exists(gen_path):
        gen_path = os.path.join(SNAP_DIR, 'samples_step014800.npy')
    samples = np.load(gen_path)  # (8, 28, 28, 1) in [-1, 1]
    samples = ((samples + 1.0) / 2.0 * 255).astype(np.uint8)  # [0, 255]

    n_classes = 10
    n_per_class = 4

    fig, axes = plt.subplots(2 * n_classes, n_per_class, figsize=(10, 20))

    for cls in range(n_classes):
        # Real samples for this class
        cls_indices = np.where(y_train == cls)[0]
        chosen_real = cls_indices[:n_per_class]
        for i, idx in enumerate(chosen_real):
            ax = axes[cls * 2, i]
            ax.imshow(x_real[idx], cmap='gray', vmin=0, vmax=255)
            ax.axis('off')
            if i == 0:
                ax.set_ylabel(f'{class_names[cls]}\n(Real)', fontsize=9, fontweight='bold')

        # Generated samples — just use the 8 samples we have
        # We don't have per-class samples from the snapshot, so show all 8
        for i in range(n_per_class):
            ax = axes[cls * 2 + 1, i]
            if i < len(samples):
                ax.imshow(samples[i, :, :, 0], cmap='gray', vmin=0, vmax=255)
            ax.axis('off')
            if i == 0:
                ax.set_ylabel(f'(Generated)', fontsize=9, color='blue')

    fig.suptitle('Real vs Generated Fashion-MNIST (CFG, Step 14900)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(REPORT_DIR, "real_vs_generated.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {path}")
    return path


# ─── 4. Training Evolution GIF ────────────────────────────────────────────────

def generate_training_evolution():
    """Create a GIF showing sample quality progression at key checkpoints."""
    # Select key steps
    key_steps = [500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000,
                 9000, 10000, 11000, 12000, 13000, 14000, 14800, 14900]

    font = get_font(11)
    small_font = get_font(9)

    frames = []
    for step in key_steps:
        fname = f'samples_step{step:06d}.npy'
        fpath = os.path.join(SNAP_DIR, fname)
        if not os.path.exists(fpath):
            continue

        samples = np.load(fpath)  # (8, 28, 28, 1) in [-1, 1]
        samples = np.clip((samples + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)

        # Load loss at this step
        loss_data = np.load(LOSS_PATH)
        step_loss = loss_data[min(step - 1, len(loss_data) - 1)]

        # Create 2x4 grid
        n = len(samples)
        cols = 4
        rows = (n + cols - 1) // cols
        pad = 2
        img_h, img_w = 28, 28
        scale = 4

        header_h = 24
        grid_h = rows * (img_h * scale + pad) + pad
        grid_w = cols * (img_w * scale + pad) + pad
        canvas_h = header_h + grid_h
        canvas_w = grid_w

        canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

        for i in range(n):
            r = i // cols
            c = i % cols
            img = samples[i, :, :, 0]
            img_big = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
            x_off = c * (img_w * scale + pad) + pad
            y_off = header_h + r * (img_h * scale + pad) + pad
            canvas[y_off:y_off + img_h * scale, x_off:x_off + img_w * scale] = img_big

        pil_img = Image.fromarray(canvas, mode='L')
        draw = ImageDraw.Draw(pil_img)
        draw.text((4, 2), f"Step {step:,}  |  Loss: {step_loss:.4f}", fill=0, font=font)

        frames.append(pil_img)

    if not frames:
        print("No frames generated for training evolution GIF")
        return None

    path = os.path.join(REPORT_DIR, "training_evolution.gif")
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=600, loop=0,
    )
    print(f"Generated: {path} ({len(frames)} frames)")
    return path


# ─── 5. Milestone Samples ─────────────────────────────────────────────────────

def generate_milestone_samples():
    """Show samples at key training milestones in a single image."""
    milestones = [500, 1000, 2000, 5000, 9000, 14900]
    loss_data = np.load(LOSS_PATH)

    fig, axes = plt.subplots(len(milestones), 8, figsize=(16, 10))

    for row, step in enumerate(milestones):
        fname = f'samples_step{step:06d}.npy'
        fpath = os.path.join(SNAP_DIR, fname)
        if not os.path.exists(fpath):
            for ax in axes[row]:
                ax.axis('off')
            continue

        samples = np.load(fpath)
        samples = np.clip((samples + 1.0) / 2.0, 0, 1)
        step_loss = loss_data[min(step - 1, len(loss_data) - 1)]

        for i in range(min(8, len(samples))):
            axes[row, i].imshow(samples[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
            axes[row, i].axis('off')

        axes[row, 0].set_ylabel(f'Step {step:,}\nloss={step_loss:.4f}',
                                 fontsize=9, fontweight='bold')

    fig.suptitle('CFG Training Milestones — Fashion-MNIST', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(REPORT_DIR, "milestone_samples.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {path}")
    return path


# ─── 6. Evolution Strip ───────────────────────────────────────────────────────

def generate_evolution_strip():
    """Single-row strip showing sample quality at key steps."""
    steps = [500, 1000, 2000, 3000, 4000, 5000, 7000, 9000, 11000, 13000, 14900]
    loss_data = np.load(LOSS_PATH)

    # Use first sample from each step
    images = []
    labels = []
    for step in steps:
        fname = f'samples_step{step:06d}.npy'
        fpath = os.path.join(SNAP_DIR, fname)
        if not os.path.exists(fpath):
            continue
        samples = np.load(fpath)
        img = np.clip((samples[0, :, :, 0] + 1.0) / 2.0, 0, 1)
        images.append(img)
        step_loss = loss_data[min(step - 1, len(loss_data) - 1)]
        labels.append(f'{step//1000}K\n{step_loss:.3f}')

    if not images:
        print("No images for evolution strip")
        return None

    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 2.5, 3))
    for i, (img, label) in enumerate(zip(images, labels)):
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(label, fontsize=8)
        axes[i].axis('off')

    fig.suptitle('Training Evolution (first sample)', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(REPORT_DIR, "evolution_strip.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated: {path}")
    return path


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("Generating report images...")
    generate_loss_curve()
    generate_architecture()
    generate_real_vs_generated()
    generate_training_evolution()
    generate_milestone_samples()
    generate_evolution_strip()
    print("\nAll report images generated!")
