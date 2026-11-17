"""Experiment 4: shape of the inference noise schedule, sampler fixed (Euler, guidance 7.5).

SD v1.5 was trained with a 1,000-step scaled-linear beta schedule. At inference the sampler
visits only a few of those noise levels, and *which* ones is a choice:
  - leading:  t = 0, k, 2k, ... (+1 offset); SD v1.5's default; never starts at pure noise
  - linspace: evenly spaced from t = 999 to t = 0
  - trailing: t = 999, 999 - k, ...; starts at the highest noise level
  - karras:   sigma_i spaced as in Karras et al. (2022): more steps at low noise
Each is run at 10 and 25 steps. Per configuration:
  - CLIP score, mean ± 95% CI over prompts
  - seconds per image
  - mean |Δpixel| (0-255) and CLIP image similarity to the leading schedule at 50 steps
Also plots the sigma sequence of every schedule (no GPU needed for that part).

Writes outputs/milestone2/results/noise_schedule.csv, outputs/milestone2/results/noise_schedule_summary.json and
outputs/milestone2/figures/{noise_schedule_sigmas, noise_schedule_metrics, noise_schedule_grid}.png.
Images go to outputs/milestone2/images/noise_schedule/<schedule>_<steps>/.

    python scripts/milestone2/04_noise_schedule.py
    python scripts/milestone2/04_noise_schedule.py --n 8 --schedules leading karras --steps 10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # make the t2i package importable

from t2i import experiments as m2
from t2i import experiment_plots as plots

SCHEDULES = {
    "leading": {"timestep_spacing": "leading"},
    "linspace": {"timestep_spacing": "linspace"},
    "trailing": {"timestep_spacing": "trailing"},
    "karras": {"timestep_spacing": "leading", "use_karras_sigmas": True},
}


def pixel_diff(a, b) -> float:
    return float(np.abs(np.asarray(Image.open(a).convert("RGB"), float) -
                        np.asarray(Image.open(b).convert("RGB"), float)).mean())


def schedule_sigmas(sched_cls, base_config, overrides: dict, steps: int) -> list[float]:
    """Noise levels the sampler visits, highest first (final 0 dropped)."""
    s = sched_cls.from_config(base_config, **overrides)
    s.set_timesteps(steps)
    return [round(float(x), 4) for x in s.sigmas[:-1]]


def main():
    cfg = m2.load_config()
    ec, dc = cfg["noise_schedule"], cfg["diffusion"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=ec["n_prompts"])
    ap.add_argument("--schedules", nargs="+", default=ec["schedules"], choices=list(SCHEDULES))
    ap.add_argument("--steps", type=int, nargs="+", default=ec["steps"])
    ap.add_argument("--model-id", default=dc["model_id"])
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    dc["model_id"] = args.model_id
    steps_list, ref_n = sorted(args.steps), ec["reference_steps"]
    configs = [(s, n) for s in args.schedules for n in steps_list]
    ref = ("leading", ref_n)

    out_dir = m2.resolve(cfg["outputs"]) / "noise_schedule"
    prompts = m2.load_prompts(cfg, args.n)

    def path(s, n, p):
        return out_dir / f"{s}_{n}" / f"{p['idx']:04d}.png"

    timing_path = m2.resolve(cfg["outputs"]) / "timing.json"
    from t2i.generate import SCHEDULERS
    sched_cls = SCHEDULERS[ec["scheduler"]]
    if not args.score_only:
        pipe = m2.build_pipeline(cfg, ec["scheduler"])
        base = dict(pipe.scheduler.config)
        for s, n in [ref] + configs:
            m2.set_scheduler(pipe, ec["scheduler"], base, **SCHEDULES[s])
            jobs = [{"path": path(s, n, p), "prompt": p["caption"], "seed": p["idx"]} for p in prompts]
            t = m2.run_jobs(pipe, jobs, n, ec["guidance"], cfg, f"{s} {n} steps")
            print(f"{s} {n} steps:", t)
            m2.merge_timing(timing_path, f"noise_schedule_{s}_{n}", t)
        del pipe
        torch.cuda.empty_cache()
    else:
        from diffusers import PNDMScheduler
        base = dict(PNDMScheduler.load_config(dc["model_id"], subfolder="scheduler"))
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}

    sigmas = {s: {n: schedule_sigmas(sched_cls, base, SCHEDULES[s], n) for n in steps_list}
              for s in args.schedules}

    scorer = m2.ClipScorer(cfg["clip_score"]["model"], cfg["clip_score"]["batch_size"])
    txt = scorer.texts([p["caption"] for p in prompts])
    emb = {}
    for c in [ref] + configs:
        ok = [k for k, p in enumerate(prompts) if path(*c, p).exists()]
        if ok:
            emb[c] = (ok, scorer.images([path(*c, prompts[k]) for k in ok]))

    rows, summary = [], []
    for c in [ref] + configs:
        if c not in emb or (c == ref and c in configs):
            continue
        s, n = c
        ok, z = emb[c]
        clip = scorer.score(z, txt[ok])
        px, sim, first = [], [], len(rows)
        for r, k in enumerate(ok):
            p = prompts[k]
            row = {"schedule": s, "steps": n, "idx": p["idx"], "seed": p["idx"],
                   "file": f"{s}_{n}/{path(s, n, p).name}", "clip_score": round(float(clip[r]), 3),
                   **{key: round(v, 4) for key, v in m2.image_stats(path(s, n, p)).items()}}
            if c != ref and ref in emb and k in emb[ref][0]:
                px.append(pixel_diff(path(s, n, p), path(*ref, p)))
                sim.append(float(z[r] @ emb[ref][1][emb[ref][0].index(k)]))
                row["pixel_diff_to_ref"] = round(px[-1], 3)
                row["clip_img_sim_to_ref"] = round(sim[-1], 4)
            rows.append(row)
        m, e = m2.mean_ci(clip)
        summary.append({
            "schedule": s, "steps": n, "images": len(ok), "reference": c == ref,
            "clip_score": round(m, 3), "clip_score_ci95": round(e, 3),
            "saturation": round(float(np.mean([r["saturation"] for r in rows[first:]])), 4),
            "seconds_per_image": timing.get(f"noise_schedule_{s}_{n}", {}).get("seconds_per_image"),
            "pixel_diff_to_ref": round(float(np.mean(px)), 2) if px else (0.0 if c == ref else None),
            "clip_img_sim_to_ref": round(float(np.mean(sim)), 4) if sim else (1.0 if c == ref else None),
            "first_sigma": sigmas[s][n][0] if s in sigmas and n in sigmas[s] else None,
        })

    res = m2.OUT / "results"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    m2.write_csv([{k: r.get(k, "") for k in keys} for r in rows], res / "noise_schedule.csv")
    out = {"model_id": dc["model_id"], "scheduler": ec["scheduler"], "guidance": ec["guidance"],
           "prompts": len(prompts), "gpu": m2.gpu_name(), "batch_size": dc["batch_size"],
           "reference": f"{ref[0]} schedule, {ref_n} steps", "per_config": summary, "sigmas": sigmas}
    (res / "noise_schedule_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "sigmas"}, indent=2))

    shots = m2.OUT / "figures"
    shots.mkdir(exist_ok=True)
    plots.noise_schedule_sigmas(sigmas, shots / "noise_schedule_sigmas.png")
    plots.noise_schedule_metrics([x for x in summary if not x["reference"]], args.schedules,
                                 steps_list, shots / "noise_schedule_metrics.png")
    show = prompts[:2]
    grid = [[path(s, n, p) if path(s, n, p).exists() else None for p in show for n in steps_list]
            for s in args.schedules]
    cols = [f"{n} steps ({'AB'[i]})" for i, _ in enumerate(show) for n in steps_list]
    title = "  |  ".join(f"{'AB'[i]}: {p['caption']}" for i, p in enumerate(show))
    plots.image_table(grid, args.schedules, cols, shots / "noise_schedule_grid.png", title,
                      cell=2.0, label_width=12)


if __name__ == "__main__":
    main()
