"""Encode all processed captions with the configured text encoder and cache to disk.

    python scripts/encode_captions.py
    python scripts/encode_captions.py --encoder bert-base-uncased --kind bert
"""

import argparse

from t2i.config import load_config, resolve
from t2i.data import read_jsonl
from t2i.embeddings import TextEncoder, cache_embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--kind", default=None, choices=["clip", "bert"])
    ap.add_argument("--save-hidden", action="store_true",
                    help="also store full (N, 77, D) token embeddings (large)")
    args = ap.parse_args()

    cfg = load_config()
    te = cfg["text_encoder"]
    name, kind = args.encoder or te["name"], args.kind or te["type"]
    processed = resolve(cfg["data"]["processed"])

    records = read_jsonl(processed / "captions.jsonl")
    enc = TextEncoder(name, kind, te["max_length"])
    tag = name.split("/")[-1]
    out = processed / f"embeddings_{tag}.pt"
    res = cache_embeddings(records, enc, out, te["batch_size"], args.save_hidden)
    print(f"saved {res['pooled'].shape} pooled embeddings -> {out}")


if __name__ == "__main__":
    main()
