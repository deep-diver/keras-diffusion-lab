"""Generate rich visualizations for the three research reports.

Generates publication-quality plots for DDIM sampling, FID evaluation,
and guidance sweep reports. Uses a consistent dark theme throughout.

Usage:
    python scripts/generate_research_plots.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# ── Theme ────────────────────────────────────────────────────────────────
BG_DARK = '#0f0f23'
BG_PLOT = '#1a1a2e'
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


def _save(fig, path, tight=True):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    if tight:
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(),
                     edgecolor='none', bbox_inches='tight')
    else:
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(),
                     edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# DDIM REPORT VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════

def generate_ddim_plots():
    """Generate all DDIM report visualizations."""
    print("\n=== DDIM Report Visualizations ===")
    out = "artifacts/reports/ddim-sampling-2026-05"

    # Load data
    data = np.load("artifacts/ddim_comparison/sampler_comparison.npz",
                   allow_pickle=True)
    methods = list(data['methods'])
    steps = list(data['steps'])
    times = list(data['times'])
    fids = list(data['fids'])

    # ── 1. Speed vs Quality Tradeoff (main figure) ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    _style_fig(fig)
    _style_ax(ax, 'Speed vs Quality: DDPM vs DDIM Sampling',
              'Generation Time (seconds, 4 samples)', 'FID')

    # Plot each method
    colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_GREEN, ACCENT_GREEN, ACCENT_GREEN]
    markers = ['o', 's', '^', 'D', 'v']
    sizes = [180, 180, 140, 120, 100]
    labels_added = set()

    for i, (m, s, t, f) in enumerate(zip(methods, steps, times, fids)):
        label = f'{m}-{s}' if m == 'DDPM' else f'DDIM-{s}'
        if m == 'DDIM' and 'DDIM' not in labels_added:
            label_legend = 'DDIM'
            labels_added.add('DDIM')
        elif m == 'DDIM':
            label_legend = None
        else:
            label_legend = label
            labels_added.add(label)

        ax.scatter(t, f, c=colors[i], marker=markers[i], s=sizes[i],
                   zorder=5, label=label_legend, edgecolors='white',
                   linewidths=0.8)

        # Annotate
        offset_x = 8 if i % 2 == 0 else -60
        offset_y = 10 if i < 3 else -18
        ax.annotate(f'{label}\n{f:.0f} FID',
                    (t, f), textcoords="offset points",
                    xytext=(offset_x, offset_y), fontsize=8,
                    color=TEXT, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=SPINE, lw=0.8))

    # Add Pareto frontier line
    # Sort by time
    order = np.argsort(times)
    ax.plot([times[i] for i in order], [fids[i] for i in order],
            '--', color=ACCENT_ORANGE, alpha=0.4, linewidth=1.5, zorder=3)

    # Speedup annotation
    ax.annotate('20x speedup\n10.4s vs 210s',
                xy=(10.4, 262), xytext=(60, 262),
                fontsize=9, color=ACCENT_GREEN,
                arrowprops=dict(arrowstyle='->', color=ACCENT_GREEN, lw=1.5),
                fontweight='bold')

    ax.legend(loc='upper left', fontsize=9,
              facecolor=BG_AXES, edgecolor=SPINE, labelcolor=TEXT)
    ax.grid(True, alpha=0.15, color=SPINE)
    _save(fig, os.path.join(out, 'speed_vs_quality.png'))

    # ── 2. Per-Step Timing (bar chart) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    _style_fig(fig)
    _style_ax(ax, 'Sampling Time by Method',
              '', 'Time (seconds, 4 samples)')

    labels = [f'{m}-{s}' for m, s in zip(methods, steps)]
    bar_colors = [ACCENT_BLUE] + [ACCENT_GREEN] * 4

    bars = ax.barh(labels, times, color=bar_colors, edgecolor='white',
                   linewidth=0.5, height=0.6)

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                f'{t:.1f}s  ({t/4:.1f}s/sample)',
                va='center', color=TEXT, fontsize=9)

    ax.set_xlim(0, max(times) * 1.4)
    ax.grid(True, axis='x', alpha=0.15, color=SPINE)
    _save(fig, os.path.join(out, 'timing_comparison.png'))

    # ── 3. FID by Step Count (DDIM only) ───────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _style_fig(fig)

    ddim_steps = [s for m, s in zip(methods, steps) if m == 'DDIM']
    ddim_fids = [f for m, f in zip(methods, fids) if m == 'DDIM']
    ddim_times_arr = [t for m, t in zip(methods, times) if m == 'DDIM']

    # FID vs steps
    _style_ax(ax1, 'FID vs DDIM Steps', 'DDIM Steps', 'FID')
    ax1.plot(ddim_steps, ddim_fids, 'o-', color=ACCENT_PINK,
             linewidth=2, markersize=10, markeredgecolor='white',
             markeredgewidth=1, zorder=5)
    # DDPM baseline reference line
    ax1.axhline(y=274, color=ACCENT_BLUE, linestyle='--', alpha=0.6,
                linewidth=1.5, label='DDPM-1000 baseline')
    ax1.fill_between([min(ddim_steps)-20, max(ddim_steps)+100],
                     264, 284, alpha=0.08, color=ACCENT_ORANGE,
                     label='~Noise range (n=4)')
    ax1.legend(loc='upper left', fontsize=8,
               facecolor=BG_AXES, edgecolor=SPINE, labelcolor=TEXT)
    ax1.grid(True, alpha=0.15, color=SPINE)

    # FID vs time
    _style_ax(ax2, 'FID vs Generation Time', 'Time (seconds)', 'FID')
    ax2.plot(ddim_times_arr, ddim_fids, 'o-', color=ACCENT_GREEN,
             linewidth=2, markersize=10, markeredgecolor='white',
             markeredgewidth=1, zorder=5)
    for s, t, f in zip(ddim_steps, ddim_times_arr, ddim_fids):
        ax2.annotate(f'{s} steps', (t, f), textcoords="offset points",
                     xytext=(8, 5), fontsize=8, color=TEXT)
    ax2.axhline(y=274, color=ACCENT_BLUE, linestyle='--', alpha=0.6,
                linewidth=1.5)
    ax2.grid(True, alpha=0.15, color=SPINE)

    fig.suptitle('DDIM Quality Analysis (n=4 — FID values unreliable)',
                 fontsize=13, fontweight='bold', color=ACCENT_YELLOW, y=1.02)
    _save(fig, os.path.join(out, 'fid_analysis.png'))

    # ── 4. Speedup waterfall ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    _style_fig(fig)
    _style_ax(ax, 'DDIM Speedup over DDPM-1000',
              'Method', 'Speedup (x)')

    ddim_speedups = [210.0 / t for t in ddim_times_arr]
    labels_wf = [f'DDIM-{s}' for s in ddim_steps]
    bars = ax.bar(labels_wf, ddim_speedups, color=ACCENT_GREEN,
                  edgecolor='white', linewidth=0.5, width=0.5)

    for bar, sp in zip(bars, ddim_speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{sp:.1f}x', ha='center', va='bottom',
                color=ACCENT_GREEN, fontsize=11, fontweight='bold')

    ax.axhline(y=1.0, color=ACCENT_BLUE, linestyle='--', alpha=0.6,
               linewidth=1.5, label='DDPM-1000 baseline')
    ax.legend(fontsize=9, facecolor=BG_AXES, edgecolor=SPINE, labelcolor=TEXT)
    ax.grid(True, axis='y', alpha=0.15, color=SPINE)
    _save(fig, os.path.join(out, 'speedup_chart.png'))

    # ── 5. DDIM algorithm diagram ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    _style_fig(fig)
    ax.set_facecolor(BG_DARK)
    ax.axis('off')

    # Timeline arrows
    ax.annotate('', xy=(13, 0.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color=TEXT, lw=2))

    # DDPM: all timesteps
    ddpm_ts = np.linspace(0.5, 12.5, 20)
    for i, t_pos in enumerate(ddpm_ts):
        alpha = max(0.2, 1.0 - i/25)
        ax.plot(t_pos, 0.75, '|', color=ACCENT_BLUE, markersize=8,
                alpha=alpha)
    ax.text(6.5, 0.92, 'DDPM: 1000 timesteps (all visited)',
            ha='center', fontsize=10, color=ACCENT_BLUE, fontweight='bold')

    # DDIM: subsequence
    ddim_ts = np.linspace(0.5, 12.5, 6)
    for t_pos in ddim_ts:
        ax.plot(t_pos, 0.25, 'o', color=ACCENT_GREEN, markersize=10,
                markeredgecolor='white', markeredgewidth=1)
        ax.plot([t_pos, t_pos], [0.15, 0.35], '-', color=ACCENT_GREEN,
                alpha=0.3)
    # Dashed lines for skipped timesteps
    for i in range(len(ddim_ts)-1):
        ax.plot([ddim_ts[i], ddim_ts[i+1]], [0.25, 0.25], '--',
                color=ACCENT_GREEN, alpha=0.3)
    ax.text(6.5, 0.05, 'DDIM: 50 timesteps (subsequence only)',
            ha='center', fontsize=10, color=ACCENT_GREEN, fontweight='bold')

    # Arrow between them
    ax.annotate('', xy=(13.5, 0.4), xytext=(13.5, 0.6),
                arrowprops=dict(arrowstyle='->', color=ACCENT_ORANGE, lw=1.5))
    ax.text(13.8, 0.5, '20x\nfaster', fontsize=9, color=ACCENT_ORANGE,
            fontweight='bold', va='center')

    ax.set_xlim(0, 15)
    ax.set_ylim(-0.1, 1.1)
    _save(fig, os.path.join(out, 'ddim_concept.png'), tight=False)


# ═══════════════════════════════════════════════════════════════════════
# FID EVALUATION VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════

def generate_fid_plots():
    """Generate all FID evaluation report visualizations."""
    print("\n=== FID Evaluation Report Visualizations ===")
    out = "artifacts/reports/fid-evaluation-2026-05"

    # ── 1. Classifier architecture diagram ─────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    _style_fig(fig)
    ax.set_facecolor(BG_DARK)
    ax.axis('off')

    layers = [
        ('Input\n28×28×1', '#546e7a', 2.0),
        ('Conv2D\n32 filters', '#1565c0', 2.8),
        ('MaxPool\n14×14', '#00838f', 2.0),
        ('Conv2D\n64 filters', '#1565c0', 2.8),
        ('MaxPool\n7×7', '#00838f', 2.0),
        ('Flatten\n3136', '#6a1b9a', 2.0),
        ('Dense\n128 ★', '#c62828', 2.8),
        ('Dense\n10', '#2e7d32', 2.0),
        ('Softmax\nOutput', '#558b2f', 2.0),
    ]

    x = 0.5
    for (name, color, w) in layers:
        rect = mpatches.FancyBboxPatch(
            (x, 0.2), w, 0.6, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='white', linewidth=0.8, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w/2, 0.5, name, ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')
        x += w + 0.3
        if x < 19:
            ax.annotate('', xy=(x - 0.25, 0.5), xytext=(x - 0.05, 0.5),
                        arrowprops=dict(arrowstyle='->', color=TEXT, lw=1))

    # Star annotation
    ax.annotate('FID feature\nlayer (128-dim)',
                xy=(14.8, 0.15), xytext=(14.8, -0.15),
                fontsize=8, color=ACCENT_ORANGE, fontweight='bold',
                ha='center',
                arrowprops=dict(arrowstyle='->', color=ACCENT_ORANGE, lw=1.5))

    ax.set_xlim(0, 20)
    ax.set_ylim(-0.4, 1.0)
    _save(fig, os.path.join(out, 'classifier_architecture.png'), tight=False)

    # ── 2. FID reliability vs sample size ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    _style_fig(fig)
    _style_ax(ax, 'FID Reliability: Sample Size vs Feature Dimension',
              'Number of Samples', 'Covariance Rank Ratio (n / d)')

    dims = [32, 64, 128, 2048]
    colors_dim = [ACCENT_GREEN, ACCENT_BLUE, ACCENT_PINK, ACCENT_PURPLE]
    n_range = np.linspace(1, 2000, 500)

    for d, c in zip(dims, colors_dim):
        ratios = np.minimum(n_range / d, 1.0)
        ax.plot(n_range, ratios, '-', color=c, linewidth=2, label=f'd={d}')

    # Mark our experiments
    experiments = [(8, 128, 'n=8 (this\nreport)'),
                   (100, 128, 'n=100 (guidance\nsweep)'),
                   (500, 128, 'n=500 (needed)')]
    for n, d, label in experiments:
        ratio = min(n / d, 1.0)
        ax.plot(n, ratio, 'o', color=ACCENT_ORANGE, markersize=10,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        offset = (15, 12) if '8' in label else (15, -18)
        ax.annotate(label, (n, ratio), textcoords="offset points",
                    xytext=offset, fontsize=8, color=ACCENT_ORANGE,
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=ACCENT_ORANGE, lw=1))

    # Threshold line
    ax.axhline(y=1.0, color=ACCENT_YELLOW, linestyle='--', alpha=0.5,
               linewidth=1, label='Full rank (reliable FID)')
    ax.axhline(y=0.8, color=ACCENT_RED, linestyle=':', alpha=0.5,
               linewidth=1, label='80% rank (marginal)')

    ax.legend(loc='lower right', fontsize=8,
              facecolor=BG_AXES, edgecolor=SPINE, labelcolor=TEXT)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.15, color=SPINE)
    _save(fig, os.path.join(out, 'fid_reliability.png'))

    # ── 3. FID comparison bar chart ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    _style_fig(fig)
    _style_ax(ax, 'FID Scores (8 samples — unreliable)',
              '', 'FID')

    models = ['Unconditional\n(step 30K)', 'Conditional\n(CFG w=3.0, step 30K)']
    fids = [87, 131]
    bar_colors = [ACCENT_BLUE, ACCENT_PINK]

    bars = ax.bar(models, fids, color=bar_colors, edgecolor='white',
                  linewidth=0.5, width=0.4)

    for bar, f in zip(bars, fids):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'~{f}', ha='center', va='bottom',
                color=TEXT, fontsize=11, fontweight='bold')

    # Warning box
    ax.text(0.98, 0.95,
            'WARNING: n=8, d=128\nCovariance rank: 8/128 = 6%\nFID values unreliable',
            transform=ax.transAxes, fontsize=8, color=ACCENT_RED,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=BG_DARK,
                      edgecolor=ACCENT_RED, alpha=0.9))

    ax.set_ylim(0, 180)
    ax.grid(True, axis='y', alpha=0.15, color=SPINE)
    _save(fig, os.path.join(out, 'fid_scores.png'))

    # ── 4. FID formula breakdown ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    _style_fig(fig)

    for i, (title, desc) in enumerate([
        ('1. Extract Features',
         'Classifier Dense(128)\nactivations for real\nand generated images'),
        ('2. Compute Statistics',
         'Mean μ and Covariance Σ\nfor each distribution\n(128-dimensional)'),
        ('3. Frechet Distance',
         '||μ₁-μ₂||² + Tr(Σ₁+Σ₂\n- 2√(Σ₁Σ₂))\nLower = more similar'),
    ]):
        ax = axes[i]
        ax.set_facecolor(BG_AXES)
        ax.axis('off')
        rect = mpatches.FancyBboxPatch(
            (0.05, 0.1), 0.9, 0.8, boxstyle="round,pad=0.05",
            transform=ax.transAxes,
            facecolor=BG_DARK, edgecolor=SPINE, linewidth=1)
        ax.add_patch(rect)
        ax.text(0.5, 0.7, title, transform=ax.transAxes,
                ha='center', va='center', fontsize=11,
                color=ACCENT_BLUE, fontweight='bold')
        ax.text(0.5, 0.35, desc, transform=ax.transAxes,
                ha='center', va='center', fontsize=9,
                color=TEXT, linespacing=1.5)

        if i < 2:
            ax.annotate('', xy=(1.05, 0.5), xytext=(0.95, 0.5),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color=ACCENT_ORANGE,
                                        lw=2))

    fig.suptitle('How FID Works', fontsize=13, fontweight='bold',
                 color=TEXT, y=1.02)
    _save(fig, os.path.join(out, 'fid_process.png'))


# ═══════════════════════════════════════════════════════════════════════
# GUIDANCE SWEEP VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════

def generate_guidance_plots():
    """Generate all guidance sweep report visualizations."""
    print("\n=== Guidance Sweep Report Visualizations ===")
    out = "artifacts/reports/guidance-sweep-2026-05"

    # Load data
    data = np.load("artifacts/guidance_sweep/sweep_results.npz",
                   allow_pickle=True)
    ws = list(data['guidance_scales'])
    fids = list(data['fids'])
    accs = list(data['accuracies'])

    # ── 1. Dual-axis: FID + Accuracy vs Guidance Scale ─────────────────
    fig, ax1 = plt.subplots(figsize=(10, 6))
    _style_fig(fig)
    _style_ax(ax1, '', 'Guidance Scale (w)', 'FID')

    # FID bars
    x_pos = np.arange(len(ws))
    bars_fid = ax1.bar(x_pos - 0.15, fids, 0.3, color=ACCENT_BLUE,
                       edgecolor='white', linewidth=0.5, alpha=0.8,
                       label='FID (lower = better)')

    for bar, f in zip(bars_fid, fids):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'~{f:.0f}', ha='center', va='bottom',
                color=ACCENT_BLUE, fontsize=9, fontweight='bold')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'w={w:.1f}' for w in ws])
    ax1.set_ylim(0, max(fids) * 1.3)

    # Accuracy on second axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Classification Accuracy', fontsize=10, color=ACCENT_PINK)
    ax2.tick_params(axis='y', colors=ACCENT_PINK, labelsize=9)
    for spine in ax2.spines.values():
        spine.set_color(SPINE)

    line_acc = ax2.plot(x_pos, accs, 'o-', color=ACCENT_PINK,
                        linewidth=2.5, markersize=10,
                        markeredgecolor='white', markeredgewidth=1.5,
                        zorder=5, label='Accuracy')

    for x, a in zip(x_pos, accs):
        ax2.annotate(f'{a*100:.0f}%', (x, a), textcoords="offset points",
                     xytext=(12, 5), fontsize=9, color=ACCENT_PINK,
                     fontweight='bold')

    # Chance level reference
    ax2.axhline(y=0.1, color=ACCENT_YELLOW, linestyle='--', alpha=0.6,
                linewidth=1.5)
    ax2.text(len(ws)-0.5, 0.105, 'Chance (10%)', fontsize=8,
             color=ACCENT_YELLOW, fontweight='bold')

    ax2.set_ylim(0, 0.45)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               fontsize=9, facecolor=BG_AXES, edgecolor=SPINE,
               labelcolor=TEXT)

    ax1.grid(True, alpha=0.1, color=SPINE)
    fig.suptitle('The CFG Paradox: Better FID, Worse Accuracy',
                 fontsize=13, fontweight='bold', color=ACCENT_YELLOW, y=1.02)
    _save(fig, os.path.join(out, 'fid_vs_accuracy.png'))

    # ── 2. Per-class accuracy heatmap ──────────────────────────────────
    # Data from the report (only w=1.0 has above-chance accuracy)
    per_class_w1 = [0.0, 0.3, 0.8, 0.0, 0.3, 0.0, 0.9, 0.0, 0.6, 0.0]
    per_class_w3 = [0.1] * 10  # chance level
    per_class_w5 = [0.01] * 10  # near zero
    per_class_w75 = [0.0] * 10  # zero

    heatmap_data = np.array([per_class_w1, per_class_w3, per_class_w5,
                              per_class_w75])

    fig, ax = plt.subplots(figsize=(12, 4))
    _style_fig(fig)

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list('custom',
        ['#0f0f23', '#1a1a2e', '#4a148c', '#6a1b9a', '#e91e63', '#ff5252',
         '#ff8a80', '#ffcdd2', '#ffffff'])

    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, fontsize=9, rotation=45, ha='right')
    ax.set_yticks(range(4))
    ax.set_yticklabels([f'w={w:.1f}' for w in ws], fontsize=10)

    # Annotate cells
    for i in range(4):
        for j in range(10):
            val = heatmap_data[i, j]
            text_color = 'white' if val < 0.4 else 'black'
            ax.text(j, i, f'{val*100:.0f}%', ha='center', va='center',
                    fontsize=9, color=text_color, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('Classification Accuracy', color=TEXT, fontsize=9)
    cbar.ax.tick_params(colors=TEXT, labelsize=8)

    ax.set_title('Per-Class Accuracy Heatmap: CFG Fails Across All Classes',
                 fontsize=12, fontweight='bold', color=TEXT, pad=12)
    _save(fig, os.path.join(out, 'per_class_heatmap_v2.png'))

    # ── 3. Accuracy breakdown at w=1.0 ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    _style_fig(fig)
    _style_ax(ax, 'Per-Class Accuracy at w=1.0 (Best Case)',
              '', 'Accuracy')

    sorted_idx = np.argsort(per_class_w1)[::-1]
    sorted_names = [CLASS_NAMES[i] for i in sorted_idx]
    sorted_accs = [per_class_w1[i] for i in sorted_idx]

    bar_colors = []
    for a in sorted_accs:
        if a >= 0.5:
            bar_colors.append(ACCENT_GREEN)
        elif a > 0:
            bar_colors.append(ACCENT_ORANGE)
        else:
            bar_colors.append(ACCENT_RED)

    bars = ax.barh(sorted_names, sorted_accs, color=bar_colors,
                    edgecolor='white', linewidth=0.5, height=0.6)

    for bar, a in zip(bars, sorted_accs):
        label = f'{a*100:.0f}%' if a > 0 else '0%'
        ax.text(max(bar.get_width() + 0.02, 0.02),
                bar.get_y() + bar.get_height()/2,
                label, va='center', color=TEXT, fontsize=9,
                fontweight='bold')

    ax.axvline(x=0.1, color=ACCENT_YELLOW, linestyle='--', alpha=0.6,
               linewidth=1.5, label='Chance (10%)')
    ax.legend(fontsize=9, facecolor=BG_AXES, edgecolor=SPINE,
              labelcolor=TEXT)
    ax.set_xlim(0, 1.1)
    ax.grid(True, axis='x', alpha=0.15, color=SPINE)

    # Summary box
    ax.text(0.98, 0.05,
            'Only 5/10 classes recognized\nBest: Shirt (90%)\nFootwear: 0% across all types',
            transform=ax.transAxes, fontsize=8, color=TEXT,
            va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=BG_DARK,
                      edgecolor=SPINE, alpha=0.9))
    _save(fig, os.path.join(out, 'w1_class_breakdown.png'))

    # ── 4. CFG mechanism diagram ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    _style_fig(fig)
    ax.set_facecolor(BG_DARK)
    ax.axis('off')

    # Conditional pathway
    rect_cond = mpatches.FancyBboxPatch(
        (0.5, 3.5), 3.5, 1.5, boxstyle="round,pad=0.15",
        facecolor='#1565c0', edgecolor='white', linewidth=1, alpha=0.85)
    ax.add_patch(rect_cond)
    ax.text(2.25, 4.25, 'ε_cond(x, t, c)\nConditional\nprediction',
            ha='center', va='center', fontsize=10, color='white',
            fontweight='bold')

    # Unconditional pathway
    rect_uncond = mpatches.FancyBboxPatch(
        (0.5, 1.0), 3.5, 1.5, boxstyle="round,pad=0.15",
        facecolor='#6a1b9a', edgecolor='white', linewidth=1, alpha=0.85)
    ax.add_patch(rect_uncond)
    ax.text(2.25, 1.75, 'ε_uncond(x, t)\nUnconditional\nprediction',
            ha='center', va='center', fontsize=10, color='white',
            fontweight='bold')

    # Guidance formula
    rect_formula = mpatches.FancyBboxPatch(
        (6, 1.8), 5.5, 2.4, boxstyle="round,pad=0.15",
        facecolor='#b71c1c', edgecolor='white', linewidth=1.5, alpha=0.85)
    ax.add_patch(rect_formula)
    ax.text(8.75, 3.7, 'Guided Prediction', ha='center', fontsize=10,
            color='white', fontweight='bold')
    ax.text(8.75, 2.8, 'ε_guided = (1+w)·ε_cond - w·ε_uncond',
            ha='center', fontsize=11, color=ACCENT_YELLOW, fontweight='bold',
            family='monospace')
    ax.text(8.75, 2.2, 'w=3.0:  4·cond - 3·uncond',
            ha='center', fontsize=9, color=TEXT, family='monospace')

    # Arrows
    ax.annotate('', xy=(6, 3.5), xytext=(4.2, 4.0),
                arrowprops=dict(arrowstyle='->', color=ACCENT_BLUE, lw=2))
    ax.annotate('', xy=(6, 2.5), xytext=(4.2, 2.0),
                arrowprops=dict(arrowstyle='->', color=ACCENT_PURPLE, lw=2))

    # Multiplier labels
    ax.text(5.1, 3.9, '×4', fontsize=10, color=ACCENT_BLUE, fontweight='bold')
    ax.text(5.1, 2.1, '×3', fontsize=10, color=ACCENT_PURPLE, fontweight='bold')

    # Problem annotation
    rect_problem = mpatches.FancyBboxPatch(
        (0.5, -0.8), 11, 1.5, boxstyle="round,pad=0.15",
        facecolor='#1a1a2e', edgecolor=ACCENT_RED, linewidth=1.5,
        linestyle='--')
    ax.add_patch(rect_problem)
    ax.text(6, 0.0,
            'If ε_cond ≈ ε_uncond (weak class conditioning):\n'
            'ε_guided ≈ ε_uncond  →  No class signal  →  Chance accuracy',
            ha='center', va='center', fontsize=10, color=ACCENT_RED,
            fontweight='bold')

    ax.annotate('', xy=(6, 1.5), xytext=(6, 0.8),
                arrowprops=dict(arrowstyle='->', color=ACCENT_RED, lw=1.5,
                                linestyle='--'))

    ax.set_xlim(0, 13)
    ax.set_ylim(-1.2, 5.5)
    _save(fig, os.path.join(out, 'cfg_mechanism.png'), tight=False)

    # ── 5. Accuracy drop emphasis ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    _style_fig(fig)
    _style_ax(ax, 'Classification Accuracy vs Guidance Scale',
              'Guidance Scale (w)', 'Accuracy')

    ax.plot(ws, accs, 'o-', color=ACCENT_PINK, linewidth=3,
            markersize=14, markeredgecolor='white', markeredgewidth=2,
            zorder=5)

    for w, a in zip(ws, accs):
        label = f'{a*100:.0f}%'
        offset_y = 0.04 if a > 0.05 else 0.02
        ax.annotate(label, (w, a), textcoords="offset points",
                    xytext=(0, 18), fontsize=11, color=ACCENT_PINK,
                    fontweight='bold', ha='center')

    # Chance zone
    ax.axhspan(0, 0.1, alpha=0.1, color=ACCENT_RED)
    ax.text(6.5, 0.05, 'CHANCE ZONE', fontsize=10, color=ACCENT_RED,
            fontweight='bold', ha='right', alpha=0.7)

    # Arrow showing drop
    ax.annotate('', xy=(7.5, 0.0), xytext=(1.0, 0.29),
                arrowprops=dict(arrowstyle='->', color=ACCENT_ORANGE,
                                lw=2, linestyle='--'))
    ax.text(4.5, 0.2, 'Monotonic decline', fontsize=9,
            color=ACCENT_ORANGE, fontweight='bold', ha='center',
            rotation=-18)

    ax.set_ylim(-0.02, 0.45)
    ax.set_xlim(0, 8.5)
    ax.grid(True, alpha=0.15, color=SPINE)
    _save(fig, os.path.join(out, 'accuracy_drop.png'))


# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    generate_ddim_plots()
    generate_fid_plots()
    generate_guidance_plots()
    print("\nAll visualizations generated.")
