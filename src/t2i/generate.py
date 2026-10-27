"""Stable Diffusion pipeline loading, scheduler swapping and text-conditioned sampling."""

from pathlib import Path

import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
)
from PIL import Image

from .embeddings import default_device

SCHEDULERS = {
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "pndm": PNDMScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpmpp": DPMSolverMultistepScheduler,
}


def load_pipeline(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
                  scheduler: str = "ddim", device: str | None = None,
                  safety_checker: bool = False) -> StableDiffusionPipeline:
    """Safety checker is off for benign COCO-prompt experiments (it blacks out
    false positives and skews FID); turn it on for any public-facing demo."""
    device = device or default_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    extra = {} if safety_checker else dict(safety_checker=None, requires_safety_checker=False)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, **extra)
    set_scheduler(pipe, scheduler)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe


def set_scheduler(pipe: StableDiffusionPipeline, name: str) -> None:
    """Swap the noise scheduler, keeping the model's trained beta schedule."""
    pipe.scheduler = SCHEDULERS[name].from_config(pipe.scheduler.config)


@torch.no_grad()
def encode_prompt(pipe: StableDiffusionPipeline, prompts: list[str]) -> torch.Tensor:
    """CLIP per-token embeddings (B, 77, 768): the conditioning SD cross-attends to."""
    tok = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt")
    return pipe.text_encoder(tok.input_ids.to(pipe.device))[0]


@torch.no_grad()
def generate(pipe: StableDiffusionPipeline, prompts: list[str], steps: int = 30,
             guidance_scale: float = 7.5, seed: int = 0, negative_prompt: str | None = None,
             prompt_embeds: torch.Tensor | None = None, height: int = 512,
             width: int = 512) -> list[Image.Image]:
    """Generate one image per prompt with a fixed seed (one generator per image).

    Passing `prompt_embeds` bypasses the pipeline's own tokenizer so external
    text embeddings (e.g. from t2i.embeddings.TextEncoder) drive the U-Net.
    """
    n = len(prompts) if prompt_embeds is None else prompt_embeds.shape[0]
    gens = [torch.Generator(device="cpu").manual_seed(seed + i) for i in range(n)]
    kwargs = dict(num_inference_steps=steps, guidance_scale=guidance_scale,
                  generator=gens, height=height, width=width)
    if prompt_embeds is not None:
        neg = encode_prompt(pipe, [negative_prompt or ""] * n)
        kwargs.update(prompt_embeds=prompt_embeds.to(pipe.device, pipe.unet.dtype),
                      negative_prompt_embeds=neg)
    else:
        kwargs.update(prompt=prompts,
                      negative_prompt=[negative_prompt] * n if negative_prompt else None)
    return pipe(**kwargs).images


def make_grid(images: list[Image.Image], captions: list[str] | None = None,
              cols: int = 5, thumb: int = 256) -> Image.Image:
    """Tile images into a grid, with captions underneath when given."""
    import matplotlib.pyplot as plt
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.8), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, img in enumerate(images):
        ax = axes.flat[i]
        ax.imshow(img.resize((thumb, thumb)))
        if captions:
            ax.set_title(_wrap(captions[i]), fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    grid = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return grid


def _wrap(text: str, width: int = 34) -> str:
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    return "\n".join(lines[:4])


def save_images(images: list[Image.Image], out_dir: Path, prefix: str = "img") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        p = out_dir / f"{prefix}_{i:02d}.png"
        img.save(p)
        paths.append(p)
    return paths
