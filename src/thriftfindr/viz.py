"""Static report figures (matplotlib), light print palette.

Categorical colours (blue, orange, teal) were checked for separation on white, including under
colour-vision deficiency (CVD dE >= 9.2, normal-vision dE >= 24). The teal is below 3:1 contrast,
so series also differ by marker shape and exact numbers are given in the report tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

from thriftfindr import paths  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "^")
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42, "lines.linewidth": 1.5,
    })


def save(fig, name: str, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or paths.FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    return out_dir / f"{name}.pdf"


def confusion(ax, cm, labels, title: str, annotate_min: float = 0.05) -> None:
    """Row-normalised confusion matrix (each row = recall over that true class)."""
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap=SEQ_BLUE, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j or norm[i, j] >= annotate_min:
                ax.text(j, i, f"{100 * norm[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, loc="left")


def robustness(results: dict, series: list[tuple[str, str]], task: str, corruptions, title: str | None = None):
    """Small multiples, one panel per corruption: accuracy vs severity (0 = clean) for each model.
    Leave title None when a caption names the figure (report figures)."""
    if len(series) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} series fit the validated palette; got {len(series)}")
    fig, axes = plt.subplots(1, len(corruptions), figsize=(7.0, 1.9), sharey=True)
    for ax, kind in zip(axes, corruptions):
        for (key, label), color, marker in zip(series, SERIES, MARKERS):
            r = results[key]
            y = [r["clean"]["0"][task]] + [r[kind][s][task] for s in sorted(r[kind], key=int)]
            ax.plot(range(len(y)), y, color=color, marker=marker, markersize=4, label=label)
        ax.set_title("JPEG" if kind == "jpeg" else kind, loc="left")
        ax.set_xticks(range(4), ["0", "1", "2", "3"])
        ax.set_xlabel("severity")
    axes[0].set_ylabel(f"{task} accuracy")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(series), bbox_to_anchor=(0.5, 0.86))
    if title:
        fig.suptitle(title, x=0.01, ha="left", y=1.0, fontsize=9)
    return fig


def image_grid(rows: list[tuple[str, list[Path]]], cell: float = 0.9, title: str | None = None):
    """One row per class: a label on the left, then square thumbnails."""
    ncol = max(len(p) for _, p in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(cell * ncol + 1.2, cell * len(rows)),
                             squeeze=False, gridspec_kw={"wspace": 0.04, "hspace": 0.04})
    for r, (label, imgs) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(imgs):
                im = Image.open(imgs[c]).convert("RGB")
                side = min(im.size)
                left, top = (im.width - side) // 2, (im.height - side) // 2
                ax.imshow(im.crop((left, top, left + side, top + side)).resize((224, 224)))
        axes[r][0].text(-0.08, 0.5, label, transform=axes[r][0].transAxes, ha="right", va="center", fontsize=7)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9)
    return fig


def price_scatter(true, pred, title: str):
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    ax.scatter(true, pred, s=4, alpha=0.25, color=SERIES[0], linewidths=0)
    lo, hi = min(np.min(true), np.min(pred)), max(np.max(true), np.max(pred))
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true price (USD)")
    ax.set_ylabel("predicted price (USD)")
    ax.set_title(title, loc="left")
    return fig
