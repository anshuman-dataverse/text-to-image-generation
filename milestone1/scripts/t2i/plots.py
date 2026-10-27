"""Figures for the Milestone 1 report."""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def dataset_examples(records: list[dict], img_dir: Path, out: Path, n: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=min(n, len(records)), replace=False)
    cols = 4
    rows = (len(idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, idx):
        r = records[i]
        ax.imshow(Image.open(img_dir / r["file_name"]))
        ax.set_title("\n".join(textwrap.shorten(c, 46) for c in r["captions"][:3]), fontsize=8)
    fig.suptitle("COCO val2017 after preprocessing (512×512) with 3 of ~5 captions")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def caption_lengths(words: list[int], tokens: list[int], out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].hist(words, bins=range(0, max(words) + 2), color="#4C72B0")
    axes[0].set(xlabel="words per caption", ylabel="captions", title="Caption length (words)")
    axes[1].hist(tokens, bins=range(0, max(tokens) + 2), color="#55A868")
    axes[1].axvline(77, color="red", ls="--", label="CLIP limit (77)")
    axes[1].set(xlabel="CLIP tokens (incl. BOS/EOS)", title="Caption length (CLIP tokens)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def similarity_matrix(sim: np.ndarray, owner: list[int], out: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(sim, cmap="viridis")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    bounds = [i for i in range(1, len(owner)) if owner[i] != owner[i - 1]]
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=0.6)
        ax.axvline(b - 0.5, color="white", lw=0.6)
    ax.set(title="CLIP caption–caption similarity\n(blocks = captions of the same image)",
           xlabel="caption", ylabel="caption")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def retrieval_examples(queries: list[str], top_files: list[list[str]], hits: list[list[bool]],
                       img_dir: Path, out: Path):
    k = len(top_files[0])
    fig, axes = plt.subplots(len(queries), k, figsize=(k * 2.6, len(queries) * 2.9), squeeze=False)
    for row, (q, files, h) in enumerate(zip(queries, top_files, hits)):
        for col, (f, ok) in enumerate(zip(files, h)):
            ax = axes[row, col]
            ax.imshow(Image.open(img_dir / f).resize((256, 256)))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("limegreen" if ok else "lightgray")
                s.set_linewidth(4 if ok else 1)
            if col == 0:
                ax.set_ylabel(textwrap.fill(q, 28), fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(f"CLIP text→image retrieval: top-{k} (green = ground-truth image)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
