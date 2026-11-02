"""Download COCO val2017, clean captions, take a subset and resize images to 512x512.

Writes data/processed/{images/, captions.jsonl, pairs.jsonl}, results/dataset_stats.json,
results/pairs_manifest.csv (one caption per image: the image-caption pair subset),
results/subset_checks.json and screenshots/{dataset_examples, caption_lengths}.png.
"""
import argparse
import csv
import json

import numpy as np
from transformers import CLIPTokenizer

from t2i.config import ROOT, load_config, resolve
from t2i.data import (check_pairs, download_coco, load_coco_captions, make_pairs,
                      preprocess_images, subset)
from t2i.plots import caption_lengths, dataset_examples


def main():
    cfg = load_config()
    d = cfg["data"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=d["num_images"])
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    root, out = resolve(d["root"]), resolve(d["processed"])
    if not args.skip_download:
        download_coco(root, d["split"])

    ann_file = root / "annotations" / f"captions_{d['split']}.json"
    raw_annotations = len(json.loads(ann_file.read_text())["annotations"])
    all_records = load_coco_captions(root, d["split"], min_words=0)
    records = load_coco_captions(root, d["split"], d["min_caption_words"])
    records = subset(records, args.num_images, d["seed"])
    kept = preprocess_images(records, root / d["split"], out, d["image_size"])

    pairs = make_pairs(kept, d["seed"])
    with open(out / "pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "pairs_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "caption_id", "file_name", "caption"])
        w.writeheader()
        w.writerows(pairs)
    checks = check_pairs(pairs, out / "images", d["image_size"])
    (ROOT / "results" / "subset_checks.json").write_text(json.dumps(checks, indent=2))
    print("pair subset checks:", json.dumps(checks))
    problems = {k: v for k, v in checks.items() if k.startswith(("missing", "empty", "placeholder", "duplicate")) and v}
    if problems:
        raise SystemExit(f"pair subset failed checks: {problems}")

    caps = [c for r in kept for c in r["captions"]]
    words = [len(c.split()) for c in caps]
    tok = CLIPTokenizer.from_pretrained(cfg["text_encoder"]["name"])
    tokens = [len(ids) for ids in tok(caps)["input_ids"]]
    stats = {
        "split": d["split"],
        "raw_images": len(all_records),
        "raw_annotations": raw_annotations,
        "raw_captions": sum(len(r["captions"]) for r in all_records),
        "images": len(kept),
        "captions": len(caps),
        "captions_per_image": round(len(caps) / len(kept), 2),
        "pairs": len(pairs),
        "caption_words_mean": round(float(np.mean(words)), 2),
        "caption_words_min_max": [min(words), max(words)],
        "clip_tokens_mean": round(float(np.mean(tokens)), 2),
        "clip_tokens_max": max(tokens),
        "captions_over_77_tokens": sum(t > 77 for t in tokens),
        "image_size": d["image_size"],
        "seed": d["seed"],
    }
    (ROOT / "results" / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    shots = ROOT / "screenshots"
    shots.mkdir(exist_ok=True)
    dataset_examples(kept, out / "images", shots / "dataset_examples.png")
    caption_lengths(words, tokens, shots / "caption_lengths.png")


if __name__ == "__main__":
    main()
