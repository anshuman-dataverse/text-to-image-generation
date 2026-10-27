"""Download COCO val2017, clean captions, subset, and resize images.

    python scripts/prepare_data.py                  # full config (5k images)
    python scripts/prepare_data.py --num-images 1000
"""

import argparse
import json
from collections import Counter

from t2i.config import load_config, resolve
from t2i.data import download_coco, load_coco_captions, preprocess_images, subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--num-images", type=int, default=None)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    d = cfg["data"]
    root, out = resolve(d["root"]), resolve(d["processed"])
    n = args.num_images or d["num_images"]

    if not args.skip_download:
        download_coco(root, d["split"])

    records = load_coco_captions(root, d["split"], d["min_caption_words"])
    records = subset(records, n, d["seed"])
    kept = preprocess_images(records, root / d["split"], out, d["image_size"])

    caps = [c for r in kept for c in r["captions"]]
    lengths = Counter(len(c.split()) for c in caps)
    stats = {
        "images": len(kept),
        "captions": len(caps),
        "captions_per_image": round(len(caps) / max(len(kept), 1), 2),
        "mean_caption_words": round(sum(k * v for k, v in lengths.items()) / max(len(caps), 1), 2),
        "max_caption_words": max(lengths) if lengths else 0,
        "image_size": d["image_size"],
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
