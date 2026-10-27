"""Sanity-check the text embeddings before using them as diffusion conditioning.

1. Captions of the same image should be closer than captions of different images.
2. CLIP text->image retrieval: the shared embedding space should find the right image.

Writes results/embedding_checks.json and screenshots/{caption_similarity, retrieval_examples}.png.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from t2i.config import ROOT, load_config, resolve
from t2i.data import read_jsonl
from t2i.embeddings import default_device
from t2i.plots import retrieval_examples, similarity_matrix


def caption_similarity(emb: dict, n_images: int, seed: int) -> dict:
    """Same-image vs different-image cosine similarity of cached pooled embeddings."""
    ids = emb["image_ids"].numpy()
    uniq = np.unique(ids)
    pick = np.random.default_rng(seed).choice(uniq, size=min(n_images, len(uniq)), replace=False)
    mask = np.isin(ids, pick)
    z = F.normalize(emb["pooled"][mask].float(), dim=-1)
    owner = torch.from_numpy(ids[mask])
    sim = z @ z.T
    same = owner[:, None] == owner[None, :]
    off = ~torch.eye(len(z), dtype=torch.bool)
    s_same, s_diff = sim[same & off], sim[~same]
    g = torch.Generator().manual_seed(seed)
    a = s_same[torch.randint(len(s_same), (5000,), generator=g)]
    b = s_diff[torch.randint(len(s_diff), (5000,), generator=g)]
    return {
        "images": int(len(pick)),
        "captions": int(len(z)),
        "same_image_mean": round(s_same.mean().item(), 4),
        "different_image_mean": round(s_diff.mean().item(), 4),
        "gap": round((s_same.mean() - s_diff.mean()).item(), 4),
        # Probability a random same-image pair beats a random different-image pair
        "pairwise_auc": round((a[:, None] > b[None, :]).float().mean().item(), 4),
    }


@torch.no_grad()
def retrieval(records: list[dict], img_dir, name: str, n: int, batch: int = 64) -> tuple[dict, torch.Tensor]:
    dev = default_device()
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(name, torch_dtype=dtype).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(name)
    recs = records[:n]
    img_z, txt_z = [], []
    for i in tqdm(range(0, len(recs), batch), desc="clip retrieval"):
        b = recs[i:i + batch]
        px = proc(images=[Image.open(img_dir / r["file_name"]).convert("RGB") for r in b],
                  return_tensors="pt")["pixel_values"].to(dev, dtype)
        img_z.append(F.normalize(model.get_image_features(pixel_values=px).float(), dim=-1))
        t = proc(text=[r["captions"][0] for r in b], return_tensors="pt",
                 padding=True, truncation=True).to(dev)
        txt_z.append(F.normalize(model.get_text_features(**t).float(), dim=-1))
    sim = torch.cat(txt_z) @ torch.cat(img_z).T
    ranks = sim.argsort(dim=1, descending=True).cpu()
    gt = torch.arange(len(recs))
    pos = (ranks == gt[:, None]).float().argmax(1) + 1
    res = {"images": len(recs), "query": "first caption of each image"}
    for k in (1, 5, 10):
        res[f"recall@{k}"] = round((pos <= k).float().mean().item(), 4)
        res[f"chance@{k}"] = round(k / len(recs), 4)
    res["median_rank"] = int(pos.median().item())
    return res, ranks


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-images", type=int, default=1000)
    ap.add_argument("--retrieval-images", type=int, default=1000)
    args = ap.parse_args()

    processed = resolve(cfg["data"]["processed"])
    name = cfg["text_encoder"]["name"]
    records = read_jsonl(processed / "captions.jsonl")
    emb = torch.load(processed / f"embeddings_{name.split('/')[-1]}.pt")
    shots = ROOT / "screenshots"

    sim_res = caption_similarity(emb, args.sim_images, cfg["data"]["seed"])
    print("caption similarity:", sim_res)

    # Heatmap over the captions of 8 images
    sub = records[:8]
    owner = [k for k, r in enumerate(sub) for _ in r["captions"]]
    texts = [c for r in sub for c in r["captions"]]
    lookup = {t: i for i, t in enumerate(emb["texts"])}
    z = F.normalize(emb["pooled"][[lookup[t] for t in texts]].float(), dim=-1)
    similarity_matrix((z @ z.T).numpy(), owner, shots / "caption_similarity.png")

    ret_res, ranks = retrieval(records, processed / "images", name, args.retrieval_images)
    print("retrieval:", ret_res)
    q = [0, 1, 2, 3, 4]
    retrieval_examples([records[i]["captions"][0] for i in q],
                       [[records[j]["file_name"] for j in ranks[i, :5].tolist()] for i in q],
                       [[j == i for j in ranks[i, :5].tolist()] for i in q],
                       processed / "images", shots / "retrieval_examples.png")

    out = {"encoder": name, "caption_similarity": sim_res, "text_to_image_retrieval": ret_res}
    (ROOT / "results" / "embedding_checks.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
