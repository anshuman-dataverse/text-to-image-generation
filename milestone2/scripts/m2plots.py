"""Figures for the Milestone 2 report."""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BLUE, GREEN, ORANGE, RED = "#4C72B0", "#55A868", "#DD8452", "#C44E52"


def _thumb(path, size=256):
    return Image.open(path).convert("RGB").resize((size, size))


def image_table(rows: list[list], row_labels: list[str], col_labels: list[str], out: Path,
                title: str = "", cell: float = 2.3, label_width: int = 30):
    """Grid of image paths (None = empty cell) with prompt labels on the left."""
    nr, nc = len(rows), len(col_labels)
    fig, axes = plt.subplots(nr, nc, figsize=(nc * cell + 1.8, nr * cell + 0.6), squeeze=False)
    for r, row in enumerate(rows):
        for c, p in enumerate(row):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            if p is not None:
                ax.imshow(_thumb(p))
            else:
                ax.axis("off")
            if r == 0:
                ax.set_title(col_labels[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(textwrap.fill(row_labels[r], label_width), fontsize=7, rotation=0,
                              ha="right", va="center")
    fig.tight_layout(rect=(0, 0, 1, 0.97) if title else None)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def captioned_grid(paths: list, captions: list[str], out: Path, title: str = "", cols: int = 4):
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, p, t in zip(axes.flat, paths, captions):
        ax.imshow(_thumb(p))
        ax.set_title(textwrap.fill(t, 36), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.97) if title else None)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def clip_hist(gen: np.ndarray, real: np.ndarray | None, out: Path):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bins = np.linspace(min(gen.min(), real.min() if real is not None else 99) - 1,
                       max(gen.max(), real.max() if real is not None else 0) + 1, 30)
    ax.hist(gen, bins=bins, alpha=0.7, color=BLUE, label=f"generated (mean {gen.mean():.1f})")
    if real is not None:
        ax.hist(real, bins=bins, alpha=0.6, color=GREEN, label=f"real COCO photo (mean {real.mean():.1f})")
    ax.set(xlabel="CLIP score (ViT-L/14)", ylabel="images", title="CLIP score against the caption")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def guidance_curves(summary: list[dict], out: Path):
    w = [s["guidance"] for s in summary]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    panels = [("clip_score", "CLIP score (↑ prompt alignment)", BLUE),
              ("diversity", "seed diversity: 1 − cos(CLIP img)", GREEN),
              ("saturation", "mean HSV saturation", ORANGE)]
    for ax, (k, label, color) in zip(axes, panels):
        m = np.array([s[k] for s in summary])
        e = np.array([s[f"{k}_ci95"] for s in summary])
        ax.errorbar(w, m, yerr=e, marker="o", color=color, capsize=3)
        ax.set(xlabel="guidance scale w", title=label)
        ax.set_xticks(w)
        ax.grid(alpha=0.3)
    fig.suptitle("Classifier-free guidance: DDIM, 30 steps (mean ± 95% CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _text_color(im, v) -> str:
    r, g, b, _ = im.cmap(im.norm(v))
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white"


def pairs_grid(top: list, bottom: list, captions: list[str], row_labels: tuple[str, str], out: Path,
               title: str = ""):
    """Two rows of images (e.g. real vs. generated), one column per caption."""
    n = len(top)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.6, 6.2), squeeze=False)
    for c in range(n):
        for r, p in enumerate((top[c], bottom[c])):
            ax = axes[r, c]
            ax.imshow(_thumb(p))
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=10)
        axes[0, c].set_title(textwrap.fill(captions[c], 30), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95) if title else None)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def scheduler_heatmaps(summary: list[dict], schedulers: list[str], steps: list[int], out: Path):
    keys = [("clip_score", "CLIP score", "viridis"),
            ("pixel_diff_to_50", "mean |Δpixel| vs. own 50-step image (/255)", "magma_r"),
            ("seconds_per_image", "seconds per image", "Blues")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for ax, (k, label, cmap) in zip(axes, keys):
        M = np.full((len(schedulers), len(steps)), np.nan)
        for s in summary:
            if s.get(k) is not None:
                M[schedulers.index(s["scheduler"]), steps.index(s["steps"])] = s[k]
        im = ax.imshow(M, cmap=cmap, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.1f}" if k != "seconds_per_image" else f"{M[i, j]:.2f}",
                            ha="center", va="center", fontsize=8,
                            color=_text_color(im, M[i, j]))
        ax.set_xticks(range(len(steps)), steps)
        ax.set_yticks(range(len(schedulers)), schedulers)
        ax.set(xlabel="sampling steps", title=label)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Scheduler × steps (guidance 7.5)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def scheduler_lines(summary: list[dict], schedulers: list[str], out: Path):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for name in schedulers:
        pts = sorted((s["steps"], s["clip_score"], s["clip_score_ci95"]) for s in summary
                     if s["scheduler"] == name)
        x, y, e = zip(*pts)
        ax.errorbar(x, y, yerr=e, marker="o", capsize=2, label=name)
    ax.set(xlabel="sampling steps", ylabel="CLIP score", title="CLIP score vs. steps per scheduler")
    ax.set_xticks(sorted({s["steps"] for s in summary}))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
