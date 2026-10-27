"""Milestone 1 deliverable: 5 prompt -> image test runs with Stable Diffusion v1.5.

    python scripts/sample_prompts.py
    python scripts/sample_prompts.py --scheduler dpmpp --steps 25 --guidance 7.5
"""

import argparse
import json
import time

from t2i.config import load_config, resolve
from t2i.generate import generate, load_pipeline, make_grid, save_images

# Mix of COCO-style everyday scenes and harder compositional prompts.
DEFAULT_PROMPTS = [
    "A man riding a bicycle down a city street.",
    "A plate of pasta with tomatoes and basil on a wooden table.",
    "Two dogs playing with a frisbee in a grassy park.",
    "A red double-decker bus parked next to a blue car.",
    "A cat sitting on a windowsill watching the rain at sunset.",
]


def main():
    cfg = load_config()["diffusion"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    ap.add_argument("--scheduler", default=cfg["scheduler"])
    ap.add_argument("--steps", type=int, default=cfg["num_inference_steps"])
    ap.add_argument("--guidance", type=float, default=cfg["guidance_scale"])
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--out", default="outputs/milestone1")
    args = ap.parse_args()

    pipe = load_pipeline(cfg["model_id"], args.scheduler)
    t0 = time.time()
    images = generate(pipe, args.prompts, args.steps, args.guidance, args.seed,
                      height=cfg["height"], width=cfg["width"])
    elapsed = time.time() - t0

    out = resolve(args.out)
    paths = save_images(images, out, prefix="sample")
    make_grid(images, args.prompts).save(out / "grid.png")
    meta = {
        "model_id": cfg["model_id"], "scheduler": args.scheduler, "steps": args.steps,
        "guidance_scale": args.guidance, "seed": args.seed,
        "seconds_total": round(elapsed, 1),
        "samples": [{"prompt": p, "file": f.name} for p, f in zip(args.prompts, paths)],
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(images)} images in {elapsed:.1f}s -> {out}")


if __name__ == "__main__":
    main()
