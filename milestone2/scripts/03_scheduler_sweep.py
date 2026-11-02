"""Experiment 3: noise scheduler × number of sampling steps (guidance 7.5).

Schedulers: DDIM, DDPM, PNDM, Euler, Euler-ancestral, DPM-Solver++ (multistep), all built
from SD v1.5's trained beta schedule; steps ∈ {10, 25, 50}. Prompt i uses seed i everywhere.
Per configuration:
  - CLIP score (prompt alignment), mean ± 95% CI over prompts
  - seconds per image (wall clock, batch of 4)
  - convergence: mean |Δpixel| (0-255) and CLIP image similarity to the same scheduler's
    50-step image. Deterministic samplers (DDIM, PNDM, Euler, DPM++) should converge;
    stochastic ones (DDPM, Euler-a) add fresh noise each step and land on a different image.
  - mean |Δpixel| to the DDIM 50-step image (do all samplers reach the same picture?)

Writes results/scheduler_sweep.csv, results/scheduler_sweep_summary.json and
screenshots/{scheduler_heatmaps, scheduler_clip_vs_steps, scheduler_grid}.png.
Images go to outputs/schedulers/<scheduler>_<steps>/.

    python 03_scheduler_sweep.py
    python 03_scheduler_sweep.py --n 8 --schedulers ddim dpmpp --steps 10 50
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image

import m2common as m2
import m2plots as plots

STOCHASTIC = {"ddpm", "euler_a"}


def pixel_diff(a, b) -> float:
    return float(np.abs(np.asarray(Image.open(a).convert("RGB"), float) -
                        np.asarray(Image.open(b).convert("RGB"), float)).mean())


def main():
    cfg = m2.load_config()
    ec, dc = cfg["scheduler_sweep"], cfg["diffusion"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=ec["n_prompts"])
    ap.add_argument("--schedulers", nargs="+", default=ec["schedulers"])
    ap.add_argument("--steps", type=int, nargs="+", default=ec["steps"])
    ap.add_argument("--model-id", default=dc["model_id"])
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    dc["model_id"] = args.model_id
    steps_list = sorted(args.steps)
    ref_steps = steps_list[-1]

    out_dir = m2.resolve(cfg["outputs"]) / "schedulers"
    prompts = m2.load_prompts(cfg, args.n)

    def path(s, n, p):
        return out_dir / f"{s}_{n}" / f"{p['idx']:04d}.png"

    timing_path = m2.resolve(cfg["outputs"]) / "timing.json"
    if not args.score_only:
        pipe = m2.build_pipeline(cfg, args.schedulers[0])
        for s in args.schedulers:
            m2.set_scheduler(pipe, s)
            for n in steps_list:
                jobs = [{"path": path(s, n, p), "prompt": p["caption"], "seed": p["idx"]} for p in prompts]
                t = m2.run_jobs(pipe, jobs, n, ec["guidance"], cfg, f"{s} {n} steps")
                print(f"{s} {n} steps:", t)
                m2.merge_timing(timing_path, f"scheduler_{s}_{n}", t)
        del pipe
        torch.cuda.empty_cache()
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}

    scorer = m2.ClipScorer(cfg["clip_score"]["model"], cfg["clip_score"]["batch_size"])
    txt = scorer.texts([p["caption"] for p in prompts])
    emb, rows, summary = {}, [], []
    for s in args.schedulers:
        for n in steps_list:
            ok = [k for k, p in enumerate(prompts) if path(s, n, p).exists()]
            if ok:
                emb[(s, n)] = (ok, scorer.images([path(s, n, prompts[k]) for k in ok]))
    ddim_ref = ("ddim", ref_steps) if ("ddim", ref_steps) in emb else None
    for s in args.schedulers:
        for n in steps_list:
            if (s, n) not in emb:
                continue
            ok, z = emb[(s, n)]
            clip = scorer.score(z, txt[ok])
            ref = emb.get((s, ref_steps))
            px_ref, sim_ref, px_ddim = [], [], []
            for r, k in enumerate(ok):
                p = prompts[k]
                row = {"scheduler": s, "steps": n, "idx": p["idx"], "seed": p["idx"],
                       "file": f"{s}_{n}/{path(s, n, p).name}", "clip_score": round(float(clip[r]), 3),
                       **{key: round(v, 4) for key, v in m2.image_stats(path(s, n, p)).items()}}
                if ref is not None and n != ref_steps and k in ref[0]:
                    px_ref.append(pixel_diff(path(s, n, p), path(s, ref_steps, p)))
                    sim_ref.append(float(z[r] @ ref[1][ref[0].index(k)]))
                    row["pixel_diff_to_50"] = round(px_ref[-1], 3)
                if ddim_ref is not None and (s, n) != ddim_ref and k in emb[ddim_ref][0]:
                    px_ddim.append(pixel_diff(path(s, n, p), path("ddim", ref_steps, p)))
                    row["pixel_diff_to_ddim50"] = round(px_ddim[-1], 3)
                rows.append(row)
            m, e = m2.mean_ci(clip)
            summary.append({
                "scheduler": s, "steps": n, "stochastic": s in STOCHASTIC, "images": len(ok),
                "clip_score": round(m, 3), "clip_score_ci95": round(e, 3),
                "saturation": round(float(np.mean([r["saturation"] for r in rows[-len(ok):]])), 4),
                "seconds_per_image": timing.get(f"scheduler_{s}_{n}", {}).get("seconds_per_image"),
                "pixel_diff_to_50": round(float(np.mean(px_ref)), 2) if px_ref else (0.0 if n == ref_steps else None),
                "clip_img_sim_to_50": round(float(np.mean(sim_ref)), 4) if sim_ref else (1.0 if n == ref_steps else None),
                "pixel_diff_to_ddim50": round(float(np.mean(px_ddim)), 2) if px_ddim else None,
            })

    res = m2.ROOT / "results"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    m2.write_csv([{k: r.get(k, "") for k in keys} for r in rows], res / "scheduler_sweep.csv")
    out = {"model_id": dc["model_id"], "guidance": ec["guidance"], "prompts": len(prompts),
           "gpu": m2.gpu_name(), "batch_size": dc["batch_size"], "reference_steps": ref_steps,
           "per_config": summary}
    (res / "scheduler_sweep_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    shots = m2.ROOT / "screenshots"
    shots.mkdir(exist_ok=True)
    plots.scheduler_heatmaps(summary, args.schedulers, steps_list, shots / "scheduler_heatmaps.png")
    plots.scheduler_lines(summary, args.schedulers, shots / "scheduler_clip_vs_steps.png")
    show = prompts[:2]
    grid = [[path(s, n, p) if path(s, n, p).exists() else None for p in show for n in steps_list]
            for s in args.schedulers]
    cols = [f"{n} steps" + (" (A)" if i == 0 else " (B)") for i, _ in enumerate(show) for n in steps_list]
    title = "  |  ".join(f"{'AB'[i]}: {p['caption']}" for i, p in enumerate(show))
    plots.image_table(grid, args.schedulers, cols, shots / "scheduler_grid.png", title, cell=2.0, label_width=12)


if __name__ == "__main__":
    main()
