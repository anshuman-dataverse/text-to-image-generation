"""Experiment 2: classifier-free guidance scale w ∈ {1, 3, 5, 7.5, 10, 15} (DDIM, 30 steps).

The same prompts and seeds are used at every w, so differences come from guidance alone.
Each prompt is sampled with 2 seeds so we can measure diversity as well as alignment:
  - CLIP score (prompt alignment), averaged over seeds, then mean ± 95% CI over prompts
  - seed diversity = 1 − cos(CLIP image embeddings of the two seeds) for the same prompt
  - colour saturation and clipped-pixel fraction (over-saturation at high w)
w = 1 means no classifier-free guidance (the conditional model alone, one U-Net pass per step).

Writes outputs/milestone2/results/guidance_sweep.csv, outputs/milestone2/results/guidance_sweep_summary.json and
outputs/milestone2/figures/{guidance_curves, guidance_grid}.png. Images go to outputs/milestone2/images/guidance/w<w>/.

    python scripts/milestone2/02_guidance_sweep.py
    python scripts/milestone2/02_guidance_sweep.py --n 8 --scales 1 7.5 15
"""
import argparse
import itertools
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
    ec, dc = cfg["guidance_sweep"], cfg["diffusion"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=ec["n_prompts"])
    ap.add_argument("--seeds", type=int, default=ec["seeds_per_prompt"])
    ap.add_argument("--scales", type=float, nargs="+", default=ec["scales"])
    ap.add_argument("--model-id", default=dc["model_id"])
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    dc["model_id"] = args.model_id

    out_dir = m2.resolve(cfg["outputs"]) / "guidance"
    prompts = m2.load_prompts(cfg, args.n)

    def jobs_for(w):
        return [{"path": out_dir / f"w{w:g}" / f"{p['idx']:04d}_s{k}.png", "prompt": p["caption"],
                 "seed": p["idx"] + k * m2.SEED_STRIDE, "idx": p["idx"], "k": k, "w": w}
                for p in prompts for k in range(args.seeds)]

    timing_path = m2.resolve(cfg["outputs"]) / "timing.json"
    if not args.score_only:
        pipe = m2.build_pipeline(cfg, ec["scheduler"])
        for w in args.scales:
            t = m2.run_jobs(pipe, jobs_for(w), ec["steps"], w, cfg, f"w={w:g}")
            print(f"w={w:g}:", t)
            m2.merge_timing(timing_path, f"guidance_w{w:g}", t)
        del pipe
        torch.cuda.empty_cache()
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}

    scorer = m2.ClipScorer(cfg["clip_score"]["model"], cfg["clip_score"]["batch_size"])
    txt = dict(zip([p["idx"] for p in prompts], scorer.texts([p["caption"] for p in prompts])))
    rows, summary = [], []
    for w in args.scales:
        jobs = [j for j in jobs_for(w) if j["path"].exists()]
        if not jobs:
            continue
        img = scorer.images([j["path"] for j in jobs])
        clip = scorer.score(img, np.stack([txt[j["idx"]] for j in jobs]))
        by_prompt: dict[int, list[int]] = {}
        for n, (j, c) in enumerate(zip(jobs, clip)):
            st = m2.image_stats(j["path"])
            rows.append({"guidance": w, "idx": j["idx"], "seed": j["seed"], "file": f"w{w:g}/{j['path'].name}",
                         "clip_score": round(float(c), 3), **{k: round(v, 4) for k, v in st.items()}})
            rows[-1]["_row"] = n
            by_prompt.setdefault(j["idx"], []).append(len(rows) - 1)
        per_prompt_clip, diversity, sat, clipped = [], [], [], []
        for idx, rr in by_prompt.items():
            per_prompt_clip.append(np.mean([rows[r]["clip_score"] for r in rr]))
            sat.append(np.mean([rows[r]["saturation"] for r in rr]))
            clipped.append(np.mean([rows[r]["clipped"] for r in rr]))
            z = img[[rows[r]["_row"] for r in rr]]
            if len(z) > 1:
                diversity.append(np.mean([1 - z[a] @ z[b] for a, b in itertools.combinations(range(len(z)), 2)]))
        s = {"guidance": w, "images": len(jobs), "prompts": len(by_prompt),
             "seconds_per_image": timing.get(f"guidance_w{w:g}", {}).get("seconds_per_image")}
        for key, vals in [("clip_score", per_prompt_clip), ("diversity", diversity),
                          ("saturation", sat), ("clipped", clipped)]:
            if vals:
                m, e = m2.mean_ci(vals)
                s[key], s[f"{key}_ci95"] = round(m, 4), round(e, 4)
        s["_per_prompt_clip"] = dict(zip(by_prompt, per_prompt_clip))
        summary.append(s)

    # Paired difference in CLIP score against the default w = 7.5 (same prompts and seeds)
    per_prompt = {s["guidance"]: s.pop("_per_prompt_clip") for s in summary}
    ref = per_prompt.get(7.5)
    for s in summary:
        if ref is not None and s["guidance"] != 7.5:
            pp = per_prompt[s["guidance"]]
            m, e = m2.mean_ci([pp[i] - ref[i] for i in sorted(set(pp) & set(ref))])
            s["clip_vs_w7.5"], s["clip_vs_w7.5_ci95"] = round(m, 3), round(e, 3)
    for r in rows:
        r.pop("_row")

    res = m2.OUT / "results"
    m2.write_csv(rows, res / "guidance_sweep.csv")
    out = {"model_id": dc["model_id"], "scheduler": ec["scheduler"], "steps": ec["steps"],
           "seeds_per_prompt": args.seeds, "gpu": m2.gpu_name(), "batch_size": dc["batch_size"],
           "per_scale": summary}
    (res / "guidance_sweep_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    shots = m2.OUT / "figures"
    shots.mkdir(exist_ok=True)
    plots.guidance_curves(summary, shots / "guidance_curves.png")
    show = prompts[:3]
    grid_rows, labels = [], []
    for p in show:
        for k in range(min(2, args.seeds)):
            grid_rows.append([out_dir / f"w{w:g}" / f"{p['idx']:04d}_s{k}.png" for w in args.scales])
            grid_rows[-1] = [q if q.exists() else None for q in grid_rows[-1]]
            labels.append(f"{p['caption']} (seed {p['idx'] + k * m2.SEED_STRIDE})")
    plots.image_table(grid_rows, labels, [f"w = {w:g}" for w in args.scales], shots / "guidance_grid.png",
                      "Same prompt and seed, increasing guidance scale (DDIM, 30 steps)")


if __name__ == "__main__":
    main()
