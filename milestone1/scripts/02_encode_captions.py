"""Encode every processed caption with a Transformer text encoder and cache the embeddings.

Writes data/processed/embeddings_<encoder>.pt (pooled embeddings for every caption).

    python 02_encode_captions.py
    python 02_encode_captions.py --encoder bert-base-uncased --kind bert
"""
import argparse

from t2i.config import load_config, resolve
from t2i.data import read_jsonl
from t2i.embeddings import TextEncoder, cache_embeddings


def main():
    te = load_config()["text_encoder"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=te["name"])
    ap.add_argument("--kind", default=te["type"], choices=["clip", "bert"])
    ap.add_argument("--save-hidden", action="store_true",
                    help="also store full (N, 77, D) token embeddings (~3 GB for 25k captions)")
    args = ap.parse_args()

    processed = resolve(load_config()["data"]["processed"])
    records = read_jsonl(processed / "captions.jsonl")
    enc = TextEncoder(args.encoder, args.kind, te["max_length"])
    out = processed / f"embeddings_{args.encoder.split('/')[-1]}.pt"
    res = cache_embeddings(records, enc, out, te["batch_size"], args.save_hidden)
    print(f"{len(records)} images, {len(res['texts'])} captions -> pooled {tuple(res['pooled'].shape)} -> {out}")


if __name__ == "__main__":
    main()
