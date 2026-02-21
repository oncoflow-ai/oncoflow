"""
visualisation.py – Plotting utilities for OncoFlow ML exploration.

Provides:
  - Multi-slice overlays (MRI + mask)
  - Side-by-side comparison of processing options
  - Benchmark bar/radar charts
  - Longitudinal volume timeline
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns


sns.set_theme(style="darkgrid", palette="muted")
MASK_ALPHA = 0.45
CMAP_MRI = "gray"


# ---------------------------------------------------------------------------
# Slice selection helpers
# ---------------------------------------------------------------------------

def _get_representative_slice(mask: np.ndarray) -> int:
    """Return the axial slice index with the most foreground voxels."""
    counts = (mask > 0).sum(axis=(0, 1))
    idx = int(counts.argmax())
    return max(5, min(idx, mask.shape[2] - 6))  # avoid edge slices


# ---------------------------------------------------------------------------
# Single overlay
# ---------------------------------------------------------------------------

def plot_mask_overlay(
    mri: np.ndarray,
    mask: np.ndarray,
    title: str = "",
    n_slices: int = 5,
    ax_start: Optional[int] = None,
    cmap_mask: str = "Reds",
    figsize: Tuple[int, int] = None,
) -> plt.Figure:
    """
    Plot N axial slices with MRI and mask overlay.

    Args:
        mri:      3D MRI array (H, W, D).
        mask:     3D binary mask (same shape).
        title:    Figure title.
        n_slices: Number of slices to display.
        ax_start: Starting slice index (auto if None).
    """
    d = mri.shape[2]
    center = ax_start if ax_start is not None else _get_representative_slice(mask)
    half = n_slices // 2
    slices = [max(0, min(center + i, d - 1)) for i in range(-half, n_slices - half)]

    figsize = figsize or (n_slices * 3, 4)
    fig, axes = plt.subplots(1, n_slices, figsize=figsize)
    if n_slices == 1:
        axes = [axes]

    for ax, s in zip(axes, slices):
        ax.imshow(mri[:, :, s].T, cmap=CMAP_MRI, origin="lower", aspect="auto")
        m = mask[:, :, s].T
        if m.any():
            ax.imshow(
                np.ma.masked_where(m == 0, m),
                cmap=cmap_mask,
                alpha=MASK_ALPHA,
                origin="lower",
                aspect="auto",
                vmin=0,
                vmax=1,
            )
        ax.set_title(f"Slice {s}", fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Side-by-side preprocessing comparison
# ---------------------------------------------------------------------------

def plot_preprocessing_comparison(
    volumes: Dict[str, np.ndarray],
    mask: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    figsize: Tuple[int, int] = None,
) -> plt.Figure:
    """
    Plot a single axial slice for each preprocessing option side-by-side.

    Args:
        volumes: Dict of option_name → 3D array.
        mask:    Optional ground-truth mask.
        slice_idx: Axial slice to display (auto if None).
    """
    n = len(volumes)
    figsize = figsize or (n * 4, 4.5)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    if slice_idx is None and mask is not None:
        slice_idx = _get_representative_slice(mask)
    elif slice_idx is None:
        first_vol = next(iter(volumes.values()))
        slice_idx = first_vol.shape[2] // 2

    for ax, (name, vol) in zip(axes, volumes.items()):
        s = vol[:, :, slice_idx].T
        ax.imshow(s, cmap=CMAP_MRI, origin="lower", aspect="auto")
        if mask is not None:
            m = mask[:, :, slice_idx].T
            if m.any():
                ax.imshow(
                    np.ma.masked_where(m == 0, m),
                    cmap="Reds",
                    alpha=MASK_ALPHA,
                    origin="lower",
                    aspect="auto",
                )
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle(f"Preprocessing Comparison – Axial slice {slice_idx}", fontsize=13)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-model mask comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    mri: np.ndarray,
    predictions: Dict[str, np.ndarray],
    gt: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    figsize: Tuple[int, int] = None,
) -> plt.Figure:
    """
    Compare model predictions on one axial slice.

    Columns: [MRI only, GT (if provided), model1, model2, ...]
    """
    cols = ["MRI"]
    if gt is not None:
        cols.append("GT")
    cols += list(predictions.keys())
    n = len(cols)

    if slice_idx is None and gt is not None:
        slice_idx = _get_representative_slice(gt)
    elif slice_idx is None:
        slice_idx = mri.shape[2] // 2

    figsize = figsize or (n * 3.5, 4)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    cmap_list = ["Reds", "Blues", "Greens", "Purples", "Oranges"]

    col_idx = 0
    # MRI raw
    axes[col_idx].imshow(mri[:, :, slice_idx].T, cmap=CMAP_MRI, origin="lower", aspect="auto")
    axes[col_idx].set_title("MRI (T1c)", fontsize=10)
    axes[col_idx].axis("off")
    col_idx += 1

    # GT
    if gt is not None:
        axes[col_idx].imshow(mri[:, :, slice_idx].T, cmap=CMAP_MRI, origin="lower", aspect="auto")
        gm = gt[:, :, slice_idx].T
        if gm.any():
            axes[col_idx].imshow(np.ma.masked_where(gm == 0, gm), cmap="Greens",
                                  alpha=MASK_ALPHA, origin="lower", aspect="auto")
        axes[col_idx].set_title("Ground Truth", fontsize=10)
        axes[col_idx].axis("off")
        col_idx += 1

    # Models
    for (model_name, pred), cmap in zip(predictions.items(), cmap_list):
        axes[col_idx].imshow(mri[:, :, slice_idx].T, cmap=CMAP_MRI, origin="lower", aspect="auto")
        pm = pred[:, :, slice_idx].T
        if pm.any():
            axes[col_idx].imshow(np.ma.masked_where(pm == 0, pm), cmap=cmap,
                                  alpha=MASK_ALPHA, origin="lower", aspect="auto")
        axes[col_idx].set_title(model_name, fontsize=10, fontweight="bold")
        axes[col_idx].axis("off")
        col_idx += 1

    fig.suptitle(f"Model Comparison – Axial slice {slice_idx}", fontsize=13)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Benchmark charts
# ---------------------------------------------------------------------------

def plot_benchmark_bar(
    summary_df: pd.DataFrame,
    metric: str = "dice",
    title: str = "",
    figsize: Tuple[int, int] = (9, 5),
) -> plt.Figure:
    """Bar chart comparing one metric across models."""
    df = summary_df.reset_index() if "model" not in summary_df.columns else summary_df
    df = df.dropna(subset=[metric]).sort_values(metric, ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("viridis", n_colors=len(df))
    bars = ax.bar(df["model"], df[metric], color=palette, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylabel(metric.upper().replace("_", " "), fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(title or f"{metric.upper()} by Model", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1 if metric in ("dice", "iou") else None)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_benchmark_radar(
    summary_df: pd.DataFrame,
    metrics: List[str],
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Radar / spider chart comparing multiple metrics across models.
    Each metric is normalised 0–1 (higher = better) before plotting.
    """
    df = summary_df.reset_index() if "model" not in summary_df.columns else summary_df
    df = df.dropna(subset=metrics)
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig

    # Normalise (for visualization); invert time-based metrics
    norm = df[metrics].copy()
    for col in metrics:
        rng = norm[col].max() - norm[col].min()
        if rng > 0:
            norm[col] = (norm[col] - norm[col].min()) / rng
        else:
            norm[col] = 0.5

    # Invert inference_s and hd95_mm (lower is better)
    for col in ["inference_s", "hd95_mm"]:
        if col in norm.columns:
            norm[col] = 1.0 - norm[col]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    colors = sns.color_palette("tab10", n_colors=len(df))

    for (_, row), color in zip(norm.iterrows(), colors):
        model_name = df.loc[row.name, "model"]
        values = [row[m] for m in metrics] + [row[metrics[0]]]
        ax.plot(angles, values, color=color, linewidth=2, label=model_name)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", " ").upper() for m in metrics], size=10)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title("Model Comparison (normalised)", size=14, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Longitudinal volume timeline
# ---------------------------------------------------------------------------

def plot_longitudinal_volume(
    volumes: Dict[str, Dict[str, float]],
    timepoint_labels: List[str],
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot tumor volume over time for multiple models + ground truth.

    Args:
        volumes: Dict of model_name → {timepoint → volume_cm3}
        timepoint_labels: Ordered list of timepoint names.
    """
    fig, ax = plt.subplots(figsize=figsize)
    styles = ["-o", "--s", "-.^", ":D", "-*"]
    colors = sns.color_palette("Set2", n_colors=len(volumes))

    for (model, tp_dict), style, color in zip(volumes.items(), styles, colors):
        vals = [tp_dict.get(tp, None) for tp in timepoint_labels]
        ax.plot(timepoint_labels, vals, style, label=model, color=color,
                linewidth=2, markersize=8)

    ax.set_xlabel("Timepoint", fontsize=12)
    ax.set_ylabel("Tumor Volume (cm³)", fontsize=12)
    ax.set_title("Longitudinal Tumor Volume Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Preprocessing timing table
# ---------------------------------------------------------------------------

def plot_preprocessing_table(results: List[Dict], figsize: Tuple[int, int] = (10, 3)) -> plt.Figure:
    """
    Render a preprocessing comparison table as a matplotlib figure.
    """
    df = pd.DataFrame(results)
    cols = [c for c in ["method", "duration_s", "file_size_mb", "size", "spacing_mm"] if c in df.columns]
    df = df[cols]

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)
    fig.suptitle("Preprocessing Method Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig
