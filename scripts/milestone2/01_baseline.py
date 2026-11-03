"""Experiment 1: baseline generation over COCO captions with the Milestone 1 settings
(SD v1.5, DDIM, 30 steps, guidance 7.5). Prompt i uses seed i.

Scores every image with CLIP score against its caption and, when the processed COCO images
are available, also scores the real photo as a reference.

Writes outputs/milestone2/results/baseline.csv, outputs/milestone2/results/baseline_summary.json and
outputs/milestone2/figures/{baseline_gallery, baseline_clip_extremes, baseline_clip_hist}.png.
Images go to outputs/milestone2/images/baseline/ (skipped if already there, so a session can resume).

    python scripts/milestone2/01_baseline.py
    python scripts/milestone2/01_baseline.py --n 50 --score-only
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # make the t2i package importable

from t2i import experiments as m2
from t2i import experiment_plots as plots


def main():
    cfg = m2.load_config()
    ec, dc = cfg["baseline"], cfg["diffusion"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=ec["n_prompts"])
    ap.add_argument("--model-id", default=dc["model_id"])
    ap.add_argument("--score-only", action="store_true", help="skip generation, score existing images")
    args = ap.parse_args()
    dc["model_id"] = args.model_id

    out_dir = m2.resolve(cfg["outputs"]) / "baseline"
    prompts = m2.load_prompts(cfg, args.n)
    jobs = [{"path": out_dir / f"{p['idx']:04d}_{p['image_id']}.png", "prompt": p["caption"],
             "seed": p["idx"]} for p in prompts]

    timing_path = m2.resolve(cfg["outputs"]) / "timing.json"
    if not args.score_only:
        pipe = m2.build_pipeline(cfg, ec["scheduler"])
        t = m2.run_jobs(pipe, jobs, ec["steps"], ec["guidance"], cfg, "baseline")
        print("generation:", t)
        m2.merge_timing(timing_path, "baseline", t)
        del pipe
        torch.cuda.empty_cache()
    keep = [k for k, j in enumerate(jobs) if j["path"].exists()]
    prompts, jobs = [prompts[k] for k in keep], [jobs[k] for k in keep]
    timing = json.loads(timing_path.read_text()).get("baseline", {}) if timing_path.exists() else {}

    scorer = m2.ClipScorer(cfg["clip_score"]["model"], cfg["clip_score"]["batch_size"])
    txt = scorer.texts([j["prompt"] for j in jobs])
    gen = scorer.score(scorer.images([j["path"] for j in jobs]), txt)

    coco = m2.resolve(cfg["prompts"]["coco_images"])
    real_paths = [coco / p["file_name"] for p in prompts]
    have_real = all(p.exists() for p in real_paths)
    real = scorer.score(scorer.images(real_paths), txt) if have_real else None

    rows = []
    for k, (p, j) in enumerate(zip(prompts, jobs)):
        rows.append({"idx": p["idx"], "image_id": p["image_id"], "seed": j["seed"],
                     "caption": p["caption"], "file": j["path"].name,
                     "clip_score": round(float(gen[k]), 3),
                     "clip_score_real": round(float(real[k]), 3) if have_real else "",
                     **{key: round(v, 4) for key, v in m2.image_stats(j["path"]).items()}})
    res = m2.OUT / "results"
    m2.write_csv(rows, res / "baseline.csv")

    m, e = m2.mean_ci(gen)
    summary = {
        "model_id": dc["model_id"], "scheduler": ec["scheduler"], "steps": ec["steps"],
        "guidance": ec["guidance"], "images": len(rows), "gpu": m2.gpu_name(),
        "seconds_per_image": timing.get("seconds_per_image"), "batch_size": dc["batch_size"],
        "clip_score": round(m, 2), "clip_score_ci95": round(e, 2),
        "clip_score_min_max": [round(float(gen.min()), 2), round(float(gen.max()), 2)],
        "saturation": round(float(np.mean([r["saturation"] for r in rows])), 4),
    }
    if have_real:
        mr, er = m2.mean_ci(real)
        summary.update(clip_score_real=round(mr, 2), clip_score_real_ci95=round(er, 2),
                       generated_beats_real=round(float((gen > real).mean()), 3))
    (res / "baseline_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    shots = m2.OUT / "figures"
    shots.mkdir(exist_ok=True)
    pick = list(range(min(8, len(jobs))))
    if have_real:
        plots.pairs_grid([real_paths[i] for i in pick[:6]], [jobs[i]["path"] for i in pick[:6]],
                         [prompts[i]["caption"] for i in pick[:6]], ("real COCO photo", "SD v1.5 from caption"),
                         shots / "baseline_gallery.png", "Baseline: real image vs. image generated from its caption")
    else:
        plots.captioned_grid([jobs[i]["path"] for i in pick], [prompts[i]["caption"] for i in pick],
                             shots / "baseline_gallery.png", "Baseline samples (DDIM, 30 steps, w = 7.5)")
    order = np.argsort(gen)
    ext = list(order[:4]) + list(order[-4:])
    plots.captioned_grid([jobs[i]["path"] for i in ext],
                         [f"CLIP {gen[i]:.1f}: {prompts[i]['caption']}" for i in ext],
                         shots / "baseline_clip_extremes.png",
                         "Lowest (top row) and highest (bottom row) CLIP score")
    plots.clip_hist(gen, real, shots / "baseline_clip_hist.png")


if __name__ == "__main__":
    main()
