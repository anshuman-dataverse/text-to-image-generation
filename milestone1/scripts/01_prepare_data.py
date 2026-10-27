"""Download COCO val2017, clean captions, take a subset and resize images to 512x512.

Writes data/processed/{images/, captions.jsonl}, results/dataset_stats.json and
screenshots/{dataset_examples, caption_lengths}.png.
"""
import argparse
import json

import numpy as np
from transformers import CLIPTokenizer

from t2i.config import ROOT, load_config, resolve
from t2i.data import download_coco, load_coco_captions, preprocess_images, subset
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

    all_records = load_coco_captions(root, d["split"], min_words=0)
    records = load_coco_captions(root, d["split"], d["min_caption_words"])
    records = subset(records, args.num_images, d["seed"])
    kept = preprocess_images(records, root / d["split"], out, d["image_size"])

    caps = [c for r in kept for c in r["captions"]]
    words = [len(c.split()) for c in caps]
    tok = CLIPTokenizer.from_pretrained(cfg["text_encoder"]["name"])
    tokens = [len(ids) for ids in tok(caps)["input_ids"]]
    stats = {
        "split": d["split"],
        "raw_images": len(all_records),
        "raw_captions": sum(len(r["captions"]) for r in all_records),
        "images": len(kept),
        "captions": len(caps),
        "captions_per_image": round(len(caps) / len(kept), 2),
        "caption_words_mean": round(float(np.mean(words)), 2),
        "caption_words_min_max": [min(words), max(words)],
        "clip_tokens_mean": round(float(np.mean(tokens)), 2),
        "clip_tokens_max": max(tokens),
        "captions_over_77_tokens": sum(t > 77 for t in tokens),
        "image_size": d["image_size"],
        "seed": d["seed"],
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    shots = ROOT / "screenshots"
    shots.mkdir(exist_ok=True)
    dataset_examples(kept, out / "images", shots / "dataset_examples.png")
    caption_lengths(words, tokens, shots / "caption_lengths.png")


if __name__ == "__main__":
    main()
