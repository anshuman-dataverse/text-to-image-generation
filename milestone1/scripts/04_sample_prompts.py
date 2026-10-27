"""Milestone 1 test runs: 5 prompts -> images with Stable Diffusion v1.5, fixed seeds.

Also checks that embeddings from our own CLIP encoder (t2i.embeddings) drive the
U-Net the same way as a plain prompt string.

Writes results/samples/{sample_XX.png, metadata.json}, results/conditioning_check.json
and screenshots/{milestone1_samples, conditioning_check}.png.

    python 04_sample_prompts.py
    python 04_sample_prompts.py --scheduler dpmpp --steps 25 --guidance 5
"""
import argparse
import json
import time

import numpy as np
import torch

from t2i.config import ROOT, load_config
from t2i.embeddings import TextEncoder
from t2i.generate import encode_prompt, generate, load_pipeline, make_grid, save_images

# COCO-style everyday scenes, plus counting / colour binding / mood
DEFAULT_PROMPTS = [
    "A man riding a bicycle down a city street.",
    "A plate of pasta with tomatoes and basil on a wooden table.",
    "Two dogs playing with a frisbee in a grassy park.",
    "A red double-decker bus parked next to a blue car.",
    "A cat sitting on a windowsill watching the rain at sunset.",
]


def main():
    cfg = load_config()
    dc = cfg["diffusion"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    ap.add_argument("--scheduler", default=dc["scheduler"])
    ap.add_argument("--steps", type=int, default=dc["num_inference_steps"])
    ap.add_argument("--guidance", type=float, default=dc["guidance_scale"])
    ap.add_argument("--seed", type=int, default=dc["seed"])
    args = ap.parse_args()

    pipe = load_pipeline(dc["model_id"], args.scheduler)
    gen = dict(steps=args.steps, guidance_scale=args.guidance, seed=args.seed,
               height=dc["height"], width=dc["width"])

    # Warm-up so the timing below excludes CUDA kernel compilation
    generate(pipe, args.prompts[:1], steps=2, guidance_scale=args.guidance)
    t0 = time.time()
    images = generate(pipe, args.prompts, **gen)
    elapsed = time.time() - t0

    out = ROOT / "results" / "samples"
    paths = save_images(images, out, prefix="sample")
    make_grid(images, args.prompts).save(ROOT / "screenshots" / "milestone1_samples.png")
    meta = {
        "model_id": dc["model_id"], "scheduler": args.scheduler, "steps": args.steps,
        "guidance_scale": args.guidance, "seed": args.seed, "size": [dc["height"], dc["width"]],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "seconds_per_image": round(elapsed / len(images), 2),
        "samples": [{"prompt": p, "file": f.name, "seed": args.seed + i}
                    for i, (p, f) in enumerate(zip(args.prompts, paths))],
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    # External-embedding conditioning: same seeds, embeddings from our TextEncoder
    enc = TextEncoder(cfg["text_encoder"]["name"], "clip", cfg["text_encoder"]["max_length"])
    hidden = enc.encode(args.prompts)["hidden"]
    # Compare the tensors first: an encoder mismatch shows up here without any sampling noise
    ref = encode_prompt(pipe, args.prompts).float()
    emb_diff = (hidden.to(ref.device).float() - ref).abs()
    print("embedding max abs diff vs pipeline encoder:", emb_diff.max().item(),
          "| per position (first 20):", [round(v, 3) for v in emb_diff.amax(dim=(0, 2))[:20].tolist()])
    ext = generate(pipe, args.prompts, prompt_embeds=hidden, **gen)
    diffs = [float(np.abs(np.asarray(a, float) - np.asarray(b, float)).mean()) for a, b in zip(images, ext)]
    check = {
        "embedding_shape": list(hidden.shape),
        "embedding_max_abs_diff": round(emb_diff.max().item(), 4),
        "embedding_max_abs_diff_real_tokens_vs_padding": [
            round(emb_diff[:, :8].max().item(), 4), round(emb_diff[:, 30:].max().item(), 4)],
        "mean_abs_pixel_diff_per_prompt": [round(d, 3) for d in diffs],
        "max_mean_abs_pixel_diff": round(max(diffs), 3),
        "note": "0 to a few /255 means our CLIP embeddings condition the U-Net identically to the built-in path",
    }
    (ROOT / "results" / "conditioning_check.json").write_text(json.dumps(check, indent=2))
    print(json.dumps(check, indent=2))
    make_grid(images[:3] + ext[:3], [f"prompt string: {p}" for p in args.prompts[:3]] +
              [f"external embeddings: {p}" for p in args.prompts[:3]], cols=3).save(
        ROOT / "screenshots" / "conditioning_check.png")


if __name__ == "__main__":
    main()
