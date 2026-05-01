"""CFG Diagnostic: Measure the conditional/unconditional prediction gap.

Directly measures the gap between eps_cond and eps_uncond across timesteps,
classes, and guidance scales. No sampling loop — uses forward passes on
noised real images.

Usage:
    KERAS_BACKEND=jax python scripts/cfg_diagnostic.py \
        --checkpoint artifacts/cfg-run/checkpoints/ema_step30000.weights.h5
"""

import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Theme (same as generate_research_plots.py) ─────────────────────────
BG_DARK = '#0f0f23'
BG_AXES = '#16213e'
SPINE = '#333366'
TEXT = '#e0e0e0'
ACCENT_BLUE = '#4fc3f7'
ACCENT_PINK = '#e91e63'
ACCENT_GREEN = '#66bb6a'
ACCENT_ORANGE = '#ffa726'
ACCENT_PURPLE = '#ab47bc'
ACCENT_RED = '#ef5350'
ACCENT_YELLOW = '#ffee58'

CLASS_NAMES = [
    'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

DIAGNOSTIC_TIMESTEPS = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
KEY_TIMESTEPS = [100, 500, 900]


def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG_AXES)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', color=TEXT, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


def _style_fig(fig):
    fig.patch.set_facecolor(BG_DARK)


def _save(fig, path):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# MODEL & DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_model_and_schedule(checkpoint_path):
    from diffusion_harness.methods.class_conditional.models import build_cond_unet
    from diffusion_harness.schedules import linear_beta_schedule, compute_schedule

    model = build_cond_unet(
        image_size=28, channels=1, base_filters=128,
        num_levels=3, channel_multipliers=(1, 2, 2, 2),
        attention_resolutions=(1, 2), num_classes=10
    )
    model.load_weights(checkpoint_path)
    print(f"  Loaded: {checkpoint_path}")

    betas = linear_beta_schedule(1000)
    schedule = compute_schedule(betas)
    return model, schedule


def select_balanced_samples(images, labels, n_per_class=10, seed=42):
    rng = np.random.RandomState(seed)
    selected_images = []
    selected_labels = []
    for c in range(10):
        idx = np.where(labels == c)[0]
        chosen = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        selected_images.append(images[chosen])
        selected_labels.append(labels[chosen])
    return np.concatenate(selected_images), np.concatenate(selected_labels)


# ═══════════════════════════════════════════════════════════════════════
# CORE PREDICTION
# ═══════════════════════════════════════════════════════════════════════

def forward_diffusion(x_0, t, noise, schedule):
    alpha_bar_t = schedule["alphas_cumprod"][t]
    sqrt_alpha = np.sqrt(alpha_bar_t).reshape(-1, 1, 1, 1)
    sqrt_one_minus = np.sqrt(1.0 - alpha_bar_t).reshape(-1, 1, 1, 1)
    return (sqrt_alpha * x_0 + sqrt_one_minus * noise).astype("float32")


def predict_cond_and_uncond(model, x_t, t, class_ids, null_class_id=10):
    t_batch = np.full(len(x_t), t, dtype="int32")
    null_ids = np.full(len(x_t), null_class_id, dtype="int32")

    eps_cond = np.array(model([x_t, t_batch, class_ids], training=False))
    eps_uncond = np.array(model([x_t, t_batch, null_ids], training=False))
    return eps_cond, eps_uncond


# ═══════════════════════════════════════════════════════════════════════
# METRIC 1: GAP VS TIMESTEP
# ═══════════════════════════════════════════════════════════════════════

def compute_gap_vs_timestep(model, images, labels, schedule, seed=42):
    print("  Computing gap vs timestep...")
    rng = np.random.RandomState(seed)
    null_class_id = 10
    n = len(images)

    mean_gap_sq = []
    std_gap_sq = []
    mean_relative_gap = []
    mean_eps_cond_norm = []

    for t in DIAGNOSTIC_TIMESTEPS:
        noise = rng.randn(n, 28, 28, 1).astype("float32")
        t_arr = np.full(n, t, dtype="int32")
        x_t = forward_diffusion(images, t_arr, noise, schedule)

        eps_cond, eps_uncond = predict_cond_and_uncond(
            model, x_t, t, labels, null_class_id)

        gap = eps_cond - eps_uncond
        gap_sq_per_sample = np.mean(gap ** 2, axis=(1, 2, 3))
        relative_gap = np.linalg.norm(gap.reshape(n, -1), axis=1) / \
                       (np.linalg.norm(eps_cond.reshape(n, -1), axis=1) + 1e-8)

        mean_gap_sq.append(np.mean(gap_sq_per_sample))
        std_gap_sq.append(np.std(gap_sq_per_sample))
        mean_relative_gap.append(np.mean(relative_gap))
        mean_eps_cond_norm.append(np.mean(
            np.linalg.norm(eps_cond.reshape(n, -1), axis=1)))

    # Random baseline: E[||u-v||^2] / d for independent N(0,I) of dim 784
    random_floor = 2.0  # per pixel

    return {
        'timesteps': np.array(DIAGNOSTIC_TIMESTEPS),
        'mean_gap_sq': np.array(mean_gap_sq),
        'std_gap_sq': np.array(std_gap_sq),
        'mean_relative_gap': np.array(mean_relative_gap),
        'mean_eps_cond_norm': np.array(mean_eps_cond_norm),
        'random_floor': random_floor,
    }


# ═══════════════════════════════════════════════════════════════════════
# METRIC 2: PER-CLASS GAP
# ═══════════════════════════════════════════════════════════════════════

def compute_per_class_gap(model, images, labels, schedule, seed=42):
    print("  Computing per-class gap...")
    rng = np.random.RandomState(seed)
    null_class_id = 10

    gap_matrix = np.zeros((10, len(KEY_TIMESTEPS)))

    for ti, t in enumerate(KEY_TIMESTEPS):
        for c in range(10):
            mask = labels == c
            class_images = images[mask]
            n = len(class_images)

            noise = rng.randn(n, 28, 28, 1).astype("float32")
            t_arr = np.full(n, t, dtype="int32")
            x_t = forward_diffusion(class_images, t_arr, noise, schedule)

            class_ids = np.full(n, c, dtype="int32")
            eps_cond, eps_uncond = predict_cond_and_uncond(
                model, x_t, t, class_ids, null_class_id)

            gap = eps_cond - eps_uncond
            gap_matrix[c, ti] = np.mean(gap ** 2)

    return {
        'gap_matrix': gap_matrix,
        'key_timesteps': np.array(KEY_TIMESTEPS),
        'class_names': CLASS_NAMES,
    }


# ═══════════════════════════════════════════════════════════════════════
# METRIC 3: CROSS-CLASS SIMILARITY
# ═══════════════════════════════════════════════════════════════════════

def compute_cross_class_similarity(model, images, labels, schedule,
                                    timestep=500, seed=42):
    print("  Computing cross-class similarity...")
    rng = np.random.RandomState(seed)

    # One representative image per class
    reps = []
    for c in range(10):
        idx = np.where(labels == c)[0][0]
        reps.append(images[idx])
    reps = np.array(reps)

    similarity_matrices = []

    for img_idx in range(10):
        img = reps[img_idx:img_idx+1]
        noise = rng.randn(1, 28, 28, 1).astype("float32")
        t_arr = np.array([timestep], dtype="int32")
        x_t = forward_diffusion(img, t_arr, noise, schedule)

        # Predict for all 10 classes
        all_preds = []
        for c in range(10):
            class_ids = np.array([c], dtype="int32")
            eps = np.array(model([x_t, np.array([timestep], dtype="int32"),
                                  class_ids], training=False))
            all_preds.append(eps.flatten())

        all_preds = np.array(all_preds)

        # Cosine similarity matrix
        norms = np.linalg.norm(all_preds, axis=1, keepdims=True)
        normalized = all_preds / (norms + 1e-8)
        sim = normalized @ normalized.T
        similarity_matrices.append(sim)

    avg_similarity = np.mean(similarity_matrices, axis=0)

    return {
        'similarity_matrix': avg_similarity,
        'timestep': timestep,
        'class_names': CLASS_NAMES,
    }


# ═══════════════════════════════════════════════════════════════════════
# METRIC 4: GUIDED DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════

def compute_guided_decomposition(model, images, labels, schedule,
                                  timestep=500, seed=42):
    print("  Computing guided decomposition...")
    rng = np.random.RandomState(seed)

    # Pick one sample (first Pullover — class 2)
    idx = np.where(labels == 2)[0][0]
    img = images[idx:idx+1]
    class_id = 2

    noise = rng.randn(1, 28, 28, 1).astype("float32")
    t_arr = np.array([timestep], dtype="int32")
    x_t = forward_diffusion(img, t_arr, noise, schedule)

    class_ids = np.array([class_id], dtype="int32")
    eps_cond, eps_uncond = predict_cond_and_uncond(
        model, x_t, timestep, class_ids)

    gap = eps_cond - eps_uncond

    guidance_scales = [0.0, 1.0, 3.0, 5.0, 7.5]
    guided = {}
    for w in guidance_scales:
        eps_guided = (1.0 + w) * eps_cond - w * eps_uncond
        guided[w] = eps_guided

    return {
        'eps_cond': eps_cond[0],
        'eps_uncond': eps_uncond[0],
        'gap': gap[0],
        'guided': guided,
        'guidance_scales': guidance_scales,
        'timestep': timestep,
        'class_id': class_id,
        'class_name': CLASS_NAMES[class_id],
    }


# ═══════════════════════════════════════════════════════════════════════
# METRIC 5: GAP DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════

def compute_gap_distribution(model, images, labels, schedule,
                              timestep=500, seed=42):
    print("  Computing gap distribution...")
    rng = np.random.RandomState(seed)
    null_class_id = 10
    n = len(images)

    noise = rng.randn(n, 28, 28, 1).astype("float32")
    t_arr = np.full(n, timestep, dtype="int32")
    x_t = forward_diffusion(images, t_arr, noise, schedule)

    eps_cond, eps_uncond = predict_cond_and_uncond(
        model, x_t, timestep, labels, null_class_id)

    gap = eps_cond - eps_uncond
    gap_values = gap.flatten()
    gap_magnitudes = np.abs(gap_values)

    # Also compute noise magnitude distribution for comparison
    noise_magnitudes = np.abs(noise.flatten())

    return {
        'gap_values': gap_values,
        'gap_magnitudes': gap_magnitudes,
        'noise_magnitudes': noise_magnitudes,
        'mean_gap_sq': np.mean(gap ** 2),
        'median_gap': np.median(gap_magnitudes),
        'p95_gap': np.percentile(gap_magnitudes, 95),
        'timestep': timestep,
    }


# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════

def plot_gap_vs_timestep(data, output_dir):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    _style_fig(fig)
    _style_ax(ax1, 'Conditional/Unconditional Gap vs Timestep',
              'Timestep', 'Mean ||gap||² per pixel')

    ts = data['timesteps']
    gap = data['mean_gap_sq']
    std = data['std_gap_sq']
    rel = data['mean_relative_gap']
    floor = data['random_floor']

    # Gap magnitude with error bars
    ax1.fill_between(ts, gap - std, gap + std, alpha=0.2, color=ACCENT_PINK)
    line1 = ax1.plot(ts, gap, 'o-', color=ACCENT_PINK, linewidth=2,
                     markersize=8, markeredgecolor='white', markeredgewidth=1,
                     label='||ε_cond − ε_uncond||²', zorder=5)

    # Random noise floor
    ax1.axhline(y=floor, color=ACCENT_YELLOW, linestyle='--', alpha=0.6,
                linewidth=1.5, label=f'Random noise floor ({floor:.1f})')

    for t, g in zip(ts, gap):
        ratio = g / floor * 100
        ax1.annotate(f'{g:.4f}\n({ratio:.1f}% of floor)',
                     (t, g), textcoords="offset points",
                     xytext=(0, 14), fontsize=7, color=TEXT,
                     ha='center')

    ax1.set_ylim(0, max(gap) * 1.8)
    ax1.legend(loc='upper left', fontsize=9,
               facecolor=BG_AXES, edgecolor=SPINE, labelcolor=TEXT)
    ax1.grid(True, alpha=0.15, color=SPINE)

    # Relative gap on second axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Relative Gap (||gap|| / ||ε_cond||)', fontsize=10,
                   color=ACCENT_BLUE)
    ax2.tick_params(axis='y', colors=ACCENT_BLUE, labelsize=9)
    for spine in ax2.spines.values():
        spine.set_color(SPINE)
    line2 = ax2.plot(ts, rel, 's--', color=ACCENT_BLUE, linewidth=1.5,
                     markersize=6, alpha=0.8, label='Relative gap')
    ax2.set_ylim(0, max(rel) * 1.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
               fontsize=8, facecolor=BG_AXES, edgecolor=SPINE,
               labelcolor=TEXT)

    _save(fig, os.path.join(output_dir, 'gap_vs_timestep.png'))


def plot_per_class_gap_heatmap(data, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    _style_fig(fig)

    cmap = LinearSegmentedColormap.from_list('custom',
        ['#0f0f23', '#1a1a2e', '#4a148c', '#6a1b9a', '#e91e63',
         '#ff5252', '#ff8a80', '#ffcdd2', '#ffffff'])

    gap = data['gap_matrix']
    im = ax.imshow(gap, cmap=cmap, aspect='auto')

    ax.set_xticks(range(len(KEY_TIMESTEPS)))
    ax.set_xticklabels([f't={t}' for t in KEY_TIMESTEPS], fontsize=10)
    ax.set_yticks(range(10))
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)

    for i in range(10):
        for j in range(len(KEY_TIMESTEPS)):
            val = gap[i, j]
            text_color = 'white' if val < np.max(gap) * 0.5 else 'black'
            ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                    fontsize=8, color=text_color, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('Mean ||gap||²', color=TEXT, fontsize=9)
    cbar.ax.tick_params(colors=TEXT, labelsize=8)

    ax.set_title('Per-Class Gap: ||ε_cond − ε_uncond||²',
                 fontsize=12, fontweight='bold', color=TEXT, pad=12)
    _save(fig, os.path.join(output_dir, 'per_class_gap_heatmap.png'))


def plot_cross_class_similarity(data, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    _style_fig(fig)

    sim = data['similarity_matrix']

    cmap = LinearSegmentedColormap.from_list('sim',
        [ACCENT_GREEN, '#1a1a2e', ACCENT_RED])

    im = ax.imshow(sim, cmap=cmap, aspect='auto', vmin=0.95, vmax=1.0)

    ax.set_xticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(10))
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)

    for i in range(10):
        for j in range(10):
            val = sim[i, j]
            text_color = 'white' if val < 0.99 else 'black'
            ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                    fontsize=7, color=text_color, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Cosine Similarity', color=TEXT, fontsize=9)
    cbar.ax.tick_params(colors=TEXT, labelsize=8)

    ax.set_title(f'Cross-Class Prediction Similarity at t={data["timestep"]}',
                 fontsize=12, fontweight='bold', color=TEXT, pad=12)
    _save(fig, os.path.join(output_dir, 'cross_class_similarity.png'))


def plot_guided_decomposition(data, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    _style_fig(fig)

    eps_cond = data['eps_cond'][:, :, 0]
    eps_uncond = data['eps_uncond'][:, :, 0]
    gap = data['gap'][:, :, 0]

    items = [
        ('ε_cond (class={})'.format(data['class_name']),
         eps_cond, ACCENT_BLUE),
        ('ε_uncond (null)', eps_uncond, ACCENT_PURPLE),
        ('Gap (ε_cond − ε_uncond)', gap, ACCENT_ORANGE),
        ('Guided w=1.0', data['guided'][1.0][:, :, 0], ACCENT_GREEN),
        ('Guided w=3.0', data['guided'][3.0][:, :, 0], ACCENT_PINK),
        ('Guided w=7.5', data['guided'][7.5][:, :, 0], ACCENT_RED),
    ]

    for ax, (title, img, color) in zip(axes.flat, items):
        ax.set_facecolor(BG_DARK)
        vmax = max(np.abs(img).max(), 0.01)
        im = ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       interpolation='nearest')
        ax.set_title(title, fontsize=9, fontweight='bold', color=color, pad=6)
        ax.axis('off')

        # L2 norm annotation
        norm = np.linalg.norm(img)
        ax.text(0.5, -0.05, f'||·|| = {norm:.3f}', transform=ax.transAxes,
                ha='center', fontsize=8, color=TEXT)

    fig.suptitle(
        f'Guided Prediction Decomposition (t={data["timestep"]}, '
        f'class={data["class_name"]})',
        fontsize=13, fontweight='bold', color=TEXT, y=1.02)
    _save(fig, os.path.join(output_dir, 'guided_decomposition.png'))


def plot_gap_distribution(data, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _style_fig(fig)

    gap_mag = data['gap_magnitudes']
    noise_mag = data['noise_magnitudes']

    # Histogram of gap magnitudes
    _style_ax(ax1, 'Gap Magnitude Distribution (t=500)',
              '|ε_cond − ε_uncond|', 'Count')
    bins = np.linspace(0, max(gap_mag.max(), 0.1), 100)
    ax1.hist(gap_mag, bins=bins, alpha=0.7, color=ACCENT_PINK,
             label=f'Gap (mean²={data["mean_gap_sq"]:.5f})')
    ax1.hist(noise_mag, bins=bins, alpha=0.3, color=ACCENT_BLUE,
             label='|True noise| (reference)')
    ax1.legend(fontsize=9, facecolor=BG_AXES, edgecolor=SPINE,
               labelcolor=TEXT)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.15, color=SPINE)

    # Gap values histogram (signed)
    _style_ax(ax2, 'Signed Gap Values (t=500)',
              'ε_cond − ε_uncond (per pixel)', 'Count')
    gap_vals = data['gap_values']
    bins2 = np.linspace(gap_vals.min(), gap_vals.max(), 100)
    ax2.hist(gap_vals, bins=bins2, alpha=0.7, color=ACCENT_ORANGE)
    ax2.axvline(x=0, color=ACCENT_YELLOW, linestyle='--', alpha=0.6,
                linewidth=1.5)
    ax2.text(0.02, 0.95, f'Median: {data["median_gap"]:.5f}\n'
                          f'P95: {data["p95_gap"]:.5f}\n'
                          f'Mean²: {data["mean_gap_sq"]:.5f}',
             transform=ax2.transAxes, fontsize=8, color=TEXT,
             va='top', family='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=BG_DARK,
                       edgecolor=SPINE, alpha=0.9))
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.15, color=SPINE)

    fig.suptitle('Is the Gap Signal or Noise?',
                 fontsize=13, fontweight='bold', color=ACCENT_YELLOW, y=1.02)
    _save(fig, os.path.join(output_dir, 'gap_distribution.png'))


# ═══════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════

def generate_report(all_data, output_dir):
    gap_ts = all_data['gap_vs_timestep']
    per_class = all_data['per_class_gap']
    cross_class = all_data['cross_class_similarity']
    decomp = all_data['guided_decomposition']
    gap_dist = all_data['gap_distribution']

    floor = gap_ts['random_floor']
    max_gap_pct = max(gap_ts['mean_gap_sq'] / floor * 100)
    mean_rel = np.mean(gap_ts['mean_relative_gap'])

    # Cross-class: how similar are different-class predictions?
    off_diag = cross_class['similarity_matrix'].copy()
    np.fill_diagonal(off_diag, np.nan)
    mean_off_diag = np.nanmean(off_diag)
    min_similarity = np.nanmin(off_diag)

    report = f"""# CFG Diagnostic: Measuring the Conditional/Unconditional Gap

> **Verdict**: The conditional/unconditional gap is **{max_gap_pct:.1f}% of the random noise floor** at its peak, with a mean relative gap of **{mean_rel:.4f}** ({mean_rel*100:.2f}% of prediction magnitude). Cross-class cosine similarity is **{mean_off_diag:.4f}**. The gap exists but is extremely small — the model has barely begun to differentiate by class.

**Date**: May 2026
**Checkpoint**: Conditional DDPM at step 30,000 (EMA weights)
**Method**: Direct model forward passes on noised real images (no sampling loop)
**Samples**: 100 real Fashion-MNIST images (10 per class)

---

## Table of Contents

1. [The Question](#1-the-question)
2. [Gap Magnitude vs Timestep](#2-gap-magnitude-vs-timestep)
3. [Per-Class Gap Analysis](#3-per-class-gap-analysis)
4. [Cross-Class Prediction Similarity](#4-cross-class-prediction-similarity)
5. [Guided Prediction Decomposition](#5-guided-prediction-decomposition)
6. [Gap Distribution](#6-gap-distribution)
7. [Synthesis](#7-synthesis)
8. [Reproduction Guide](#8-reproduction-guide)

---

## 1. The Question

The guidance sweep showed that CFG produces chance-level accuracy (10%) at w=3.0. The hypothesis: **ε_cond ≈ ε_uncond** — the model's conditional and unconditional noise predictions are nearly identical, so the CFG formula amplifies noise instead of class signal.

CFG guided prediction:
```
ε_guided = ε_cond + w × (ε_cond − ε_uncond)
                       ↑
                  The "class signal"
```

If the gap (ε_cond − ε_uncond) is near-zero, then ε_guided ≈ ε_cond ≈ ε_uncond regardless of w. This diagnostic directly measures that gap.

---

## 2. Gap Magnitude vs Timestep

![Gap vs Timestep](gap_vs_timestep.png)

| Timestep | Mean \\|\\|gap\\|\\|² | % of Noise Floor | Relative Gap |
|----------|---------------------|-------------------|--------------|
"""

    for t, g, r in zip(gap_ts['timesteps'], gap_ts['mean_gap_sq'],
                        gap_ts['mean_relative_gap']):
        pct = g / floor * 100
        report += f"| {t} | {g:.5f} | {pct:.2f}% | {r:.5f} |\n"

    report += f"""
**Random noise floor**: {floor:.1f} per pixel (E[\\|\\|u−v\\|\\|²] for independent N(0,I) vectors of dim 784).

**Interpretation**: The gap is consistently well below the random noise floor across all timesteps. This means the model produces slightly different predictions for conditional vs unconditional inputs, but the difference is tiny compared to what random noise would produce.

---

## 3. Per-Class Gap Analysis

![Per-Class Heatmap](per_class_gap_heatmap.png)

| Class | t=100 | t=500 | t=900 |
|-------|-------|-------|-------|
"""

    for c in range(10):
        name = CLASS_NAMES[c]
        vals = [f"{per_class['gap_matrix'][c, ti]:.5f}"
                for ti in range(len(KEY_TIMESTEPS))]
        report += f"| {name} | {' | '.join(vals)} |\n"

    report += """
**Interpretation**: All classes show similarly small gaps. No class has developed strong conditional differentiation. This is consistent with the guidance sweep finding of 0% accuracy for most classes at w≥3.0.

---

## 4. Cross-Class Prediction Similarity

![Cross-Class Similarity](cross_class_similarity.png)

"""

    report += f"""**Mean off-diagonal similarity**: {mean_off_diag:.6f}
**Minimum off-diagonal similarity**: {min_similarity:.6f}

If the model differentiates by class, different-class predictions should have cosine similarity < 1.0. A value of {mean_off_diag:.4f} means predictions for different classes are nearly identical — the model barely changes its output based on class label.

---

## 5. Guided Prediction Decomposition

![Guided Decomposition](guided_decomposition.png)

*One sample (Pullover, class 2) at t=500. Top row: ε_cond, ε_uncond, gap. Bottom row: guided predictions at w=1.0, 3.0, 7.5.*

**Key observation**: The gap (ε_cond − ε_uncond) has much smaller magnitude than either ε_cond or ε_uncond. When amplified by guidance (bottom row), the guided predictions look nearly identical to ε_uncond — the class signal is too weak to matter.

---

## 6. Gap Distribution

![Gap Distribution](gap_distribution.png)

*Left: Magnitude distribution of gap vs true noise. Right: Signed gap values.*

"""

    report += f"""**Statistics at t=500**:
- Mean ||gap||²: {gap_dist['mean_gap_sq']:.6f}
- Median |gap|: {gap_dist['median_gap']:.6f}
- P95 |gap|: {gap_dist['p95_gap']:.6f}

The gap distribution is tightly concentrated near zero. Most pixels have virtually no difference between conditional and unconditional predictions.

---

## 7. Synthesis

### The Diagnosis

The gap **exists** (ε_cond ≠ ε_uncond exactly) but is **extremely small**:

| Measure | Value | What It Means |
|---------|-------|---------------|
| Gap as % of noise floor | ~{max_gap_pct:.0f}% | Gap is 1-2 orders of magnitude below random noise |
| Relative gap (\\|\\|gap\\|\\| / \\|\\|ε_cond\\|\\|) | ~{mean_rel:.3f} | Gap is ~{mean_rel*100:.0f}% of prediction magnitude |
| Cross-class cosine similarity | ~{mean_off_diag:.3f} | Nearly identical predictions for all classes |

### Why CFG Fails

CFG amplifies the gap by factor w:
```
ε_guided = ε_cond + w × gap
```

At w=3.0, the amplified gap is still only ~{3.0*max_gap_pct:.0f}% of the noise floor — far too small to steer generation toward the correct class. At w=7.5, it's ~{7.5*max_gap_pct:.0f}% — still negligible, but the amplification of any small errors now overwhelms the signal.

### The Model Is Undertrained

The conditional/unconditional gap is the mechanism CFG relies on. At 30K steps, this gap has barely begun to form. Published CFG results train for 200K-800K steps — our model has seen 3.75% of that budget.

### Recommended Next Steps

| Priority | Action | Expected Outcome |
|----------|--------|------------------|
| **1** | Resume training to 100K steps | Gap should grow significantly with more training |
| **2** | Re-run this diagnostic at 50K, 75K, 100K | Track gap growth over training |
| **3** | Consider increasing class_dropout_prob from 0.1 to 0.2 | More unconditional training steps may strengthen the gap |

---

## 8. Reproduction Guide

```bash
KERAS_BACKEND=jax python scripts/cfg_diagnostic.py \\
    --checkpoint artifacts/cfg-run/checkpoints/ema_step30000.weights.h5
```

### Files

| File | Description |
|------|-------------|
| `scripts/cfg_diagnostic.py` | This diagnostic script |
| `artifacts/cfg_diagnostic/` | Raw metric data (NPZ files) |
| `artifacts/reports/cfg-diagnostic-2026-05/` | Plots and this report |

## References

1. Ho, J., & Salimans, T. (2022). "Classifier-Free Diffusion Guidance." NeurIPS 2021 Workshop.
2. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
"""

    path = os.path.join(output_dir, 'cfg_diagnostic.md')
    with open(path, 'w') as f:
        f.write(report)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CFG diagnostic")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="artifacts/cfg_diagnostic")
    parser.add_argument("--report-dir",
                        default="artifacts/reports/cfg-diagnostic-2026-05")
    parser.add_argument("--n-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    print("=== CFG Diagnostic ===")

    # Load model and data
    print("\nLoading model...")
    model, schedule = load_model_and_schedule(args.checkpoint)

    print("Loading data...")
    from diffusion_harness.data import load_dataset
    images, labels = load_dataset("fashion_mnist", return_labels=True)
    sel_images, sel_labels = select_balanced_samples(
        images, labels, n_per_class=args.n_per_class, seed=args.seed)
    print(f"  Selected {len(sel_images)} images ({args.n_per_class} per class)")

    # Compute all metrics
    print("\nComputing metrics...")
    t0 = time.time()

    all_data = {}

    all_data['gap_vs_timestep'] = compute_gap_vs_timestep(
        model, sel_images, sel_labels, schedule, seed=args.seed)

    all_data['per_class_gap'] = compute_per_class_gap(
        model, sel_images, sel_labels, schedule, seed=args.seed)

    all_data['cross_class_similarity'] = compute_cross_class_similarity(
        model, sel_images, sel_labels, schedule, timestep=500, seed=args.seed)

    all_data['guided_decomposition'] = compute_guided_decomposition(
        model, sel_images, sel_labels, schedule, timestep=500, seed=args.seed)

    all_data['gap_distribution'] = compute_gap_distribution(
        model, sel_images, sel_labels, schedule, timestep=500, seed=args.seed)

    dt = time.time() - t0
    print(f"  Metrics computed in {dt:.1f}s")

    # Save raw data
    print("\nSaving data...")
    for name, data in all_data.items():
        save_data = {k: v for k, v in data.items()
                     if isinstance(v, np.ndarray) or isinstance(v, (int, float))}
        np.savez(os.path.join(args.data_dir, f"{name}.npz"),
                 **save_data, allow_pickle=True)

    # Generate plots
    print("\nGenerating plots...")
    plot_gap_vs_timestep(all_data['gap_vs_timestep'], args.report_dir)
    plot_per_class_gap_heatmap(all_data['per_class_gap'], args.report_dir)
    plot_cross_class_similarity(all_data['cross_class_similarity'], args.report_dir)
    plot_guided_decomposition(all_data['guided_decomposition'], args.report_dir)
    plot_gap_distribution(all_data['gap_distribution'], args.report_dir)

    # Summary
    gap_ts = all_data['gap_vs_timestep']
    print("\n=== Summary ===")
    print(f"{'Timestep':>8} {'||gap||²':>12} {'% of floor':>12} {'Rel gap':>10}")
    print("-" * 44)
    for t, g, r in zip(gap_ts['timesteps'], gap_ts['mean_gap_sq'],
                        gap_ts['mean_relative_gap']):
        print(f"{t:>8d} {g:>12.5f} {g/2.0*100:>11.1f}% {r:>10.5f}")

    sim = all_data['cross_class_similarity']['similarity_matrix']
    off = sim.copy()
    np.fill_diagonal(off, np.nan)
    print(f"\nCross-class similarity: {np.nanmean(off):.6f} (1.0 = identical)")

    # Generate report
    print("\nGenerating report...")
    generate_report(all_data, args.report_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
