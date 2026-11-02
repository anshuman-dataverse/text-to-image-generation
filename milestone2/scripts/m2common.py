"""Shared helpers for Milestone 2: prompts, seeded batch generation with resume, CLIP score,
image statistics. Reuses the Milestone 1 `t2i` package (pipeline loading, schedulers, grids)."""

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]          # milestone2/
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "milestone1" / "scripts"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from t2i.embeddings import default_device  # noqa: E402
from t2i.generate import load_pipeline, make_grid, set_scheduler  # noqa: E402,F401

SEED_STRIDE = 10_000  # seed of prompt i, repeat k = i + k * SEED_STRIDE


def load_config(path: str | Path | None = None) -> dict:
    """configs/default.yaml, or the file named by $M2_CONFIG (used for quick local tests)."""
    path = path or os.environ.get("M2_CONFIG") or ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def load_prompts(cfg: dict, n: int) -> list[dict]:
    """First n pairs of a fixed seeded shuffle of the 5,000-pair manifest. Every experiment uses
    a prefix of the same order, so prompt i (and its seed) is shared across experiments."""
    with open(resolve(cfg["prompts"]["manifest"])) as f:
        rows = list(csv.DictReader(f))
    order = np.random.default_rng(cfg["prompts"]["seed"]).permutation(len(rows))
    return [dict(rows[j], idx=i) for i, j in enumerate(order[:n])]


def build_pipeline(cfg: dict, scheduler: str):
    dc = cfg["diffusion"]
    pipe = load_pipeline(dc["model_id"], scheduler)
    if pipe.device.type == "cuda" and not dc.get("attention_slicing", True):
        pipe.disable_attention_slicing()
    return pipe


@torch.no_grad()
def generate_seeded(pipe, prompts: list[str], seeds: list[int], steps: int, guidance: float,
                    height: int = 512, width: int = 512) -> list[Image.Image]:
    """One image per (prompt, seed). guidance <= 1 turns classifier-free guidance off (diffusers),
    i.e. w = 1 is the purely conditional model."""
    gens = [torch.Generator(device="cpu").manual_seed(int(s)) for s in seeds]
    return pipe(prompt=prompts, num_inference_steps=steps, guidance_scale=guidance, generator=gens,
                height=height, width=width).images


def run_jobs(pipe, jobs: list[dict], steps: int, guidance: float, cfg: dict, desc: str) -> dict:
    """Generate every job {'path', 'prompt', 'seed'} whose image is not on disk yet (so a
    disconnected Colab session resumes). Returns timing for the images made in this call."""
    dc = cfg["diffusion"]
    todo = [j for j in jobs if not Path(j["path"]).exists()]
    if todo:  # warm-up so timing excludes CUDA kernel compilation
        generate_seeded(pipe, [todo[0]["prompt"]], [0], 2, guidance, dc["height"], dc["width"])
    bs, made, secs = dc["batch_size"], 0, 0.0
    for i in tqdm(range(0, len(todo), bs), desc=desc, disable=not todo):
        b = todo[i:i + bs]
        if pipe.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        imgs = generate_seeded(pipe, [j["prompt"] for j in b], [j["seed"] for j in b], steps,
                               guidance, dc["height"], dc["width"])
        if pipe.device.type == "cuda":
            torch.cuda.synchronize()
        secs += time.time() - t0
        made += len(b)
        for j, img in zip(b, imgs):
            Path(j["path"]).parent.mkdir(parents=True, exist_ok=True)
            img.save(j["path"])
    return {"generated": made, "skipped": len(jobs) - len(todo),
            "seconds_per_image": round(secs / made, 3) if made else None}


def merge_timing(path: Path, key: str, timing: dict) -> dict:
    """Keep per-config timing across resumed runs: a later run with nothing to generate
    does not overwrite the timing measured when the images were made."""
    data = json.loads(path.read_text()) if path.exists() else {}
    if timing["seconds_per_image"] is not None or key not in data:
        data[key] = timing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return data[key]


class ClipScorer:
    """CLIP score = 100 * max(cos(image, text), 0) (Hessel et al. 2021; torchmetrics convention)."""

    def __init__(self, name: str = "openai/clip-vit-large-patch14", batch_size: int = 32):
        from transformers import CLIPModel, CLIPProcessor
        self.device = default_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(name, torch_dtype=self.dtype).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        self.bs = batch_size

    @staticmethod
    def _out(o) -> torch.Tensor:  # transformers<5 returns a tensor, >=5 an output object
        return o if isinstance(o, torch.Tensor) else o.pooler_output

    @torch.no_grad()
    def images(self, paths: list) -> np.ndarray:
        z = []
        for i in tqdm(range(0, len(paths), self.bs), desc="clip images", leave=False):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + self.bs]]
            px = self.proc(images=ims, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            z.append(F.normalize(self._out(self.model.get_image_features(pixel_values=px)).float(), dim=-1).cpu())
        return torch.cat(z).numpy()

    @torch.no_grad()
    def texts(self, texts: list[str]) -> np.ndarray:
        z = []
        for i in range(0, len(texts), self.bs):
            t = self.proc(text=texts[i:i + self.bs], return_tensors="pt", padding=True,
                          truncation=True).to(self.device)
            z.append(F.normalize(self._out(self.model.get_text_features(**t)).float(), dim=-1).cpu())
        return torch.cat(z).numpy()

    @staticmethod
    def score(img_z: np.ndarray, txt_z: np.ndarray) -> np.ndarray:
        return 100 * np.clip((img_z * txt_z).sum(-1), 0, None)


def image_stats(path) -> dict:
    """Colour saturation (HSV S, 0-1) and fraction of clipped channels: high guidance
    over-saturates and burns out images."""
    im = Image.open(path).convert("RGB")
    rgb = np.asarray(im)
    s = np.asarray(im.convert("HSV"))[..., 1] / 255.0
    return {"saturation": float(s.mean()), "clipped": float(((rgb == 0) | (rgb == 255)).mean())}


def mean_ci(x) -> tuple[float, float]:
    """Mean and 95% normal-approximation half-width."""
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else default_device()
