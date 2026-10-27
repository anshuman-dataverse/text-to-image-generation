"""Transformer text encoders (CLIP / BERT) and embedding caching."""

from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TextEncoder:
    """Wraps a HF text transformer and returns per-token and pooled embeddings.

    For CLIP ViT-L/14 the per-token output (B, 77, 768) is exactly the
    `prompt_embeds` tensor Stable Diffusion v1.5 cross-attends to.
    """

    def __init__(self, name: str = "openai/clip-vit-large-patch14", kind: str = "clip",
                 max_length: int = 77, device: str | None = None, dtype=torch.float16):
        self.device = device or default_device()
        self.dtype = dtype if self.device == "cuda" else torch.float32
        self.kind = kind
        self.max_length = max_length
        if kind == "clip":
            self.tokenizer = CLIPTokenizer.from_pretrained(name)
            self.model = CLIPTextModel.from_pretrained(name, torch_dtype=self.dtype)
        elif kind == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.model = AutoModel.from_pretrained(name, torch_dtype=self.dtype)
        else:
            raise ValueError(f"unknown encoder kind: {kind}")
        self.model.to(self.device).eval()

    def tokenize(self, texts: list[str]) -> dict:
        return self.tokenizer(texts, padding="max_length", max_length=self.max_length,
                              truncation=True, return_tensors="pt")

    @torch.no_grad()
    def encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        tok = self.tokenize(texts).to(self.device)
        out = self.model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        hidden = out.last_hidden_state
        if self.kind == "clip":
            pooled = out.pooler_output  # EOS-token embedding
        else:  # mean-pool over real tokens
            m = tok["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(1) / m.sum(1)
        return {"hidden": hidden, "pooled": pooled, "attention_mask": tok["attention_mask"]}


def cache_embeddings(records: list[dict], encoder: TextEncoder, out_path: Path,
                     batch_size: int = 256, save_hidden: bool = False) -> dict:
    """Encode every caption of every record; save pooled (and optionally full) embeddings.

    Full hidden states for all 25k COCO val captions are ~3 GB in fp16, so they
    are off by default; pooled embeddings are enough for similarity analysis.
    """
    texts, image_ids = [], []
    for r in records:
        for c in r["captions"]:
            texts.append(c)
            image_ids.append(r["image_id"])

    pooled, hidden = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="encode"):
        out = encoder.encode(texts[i:i + batch_size])
        pooled.append(out["pooled"].half().cpu())
        if save_hidden:
            hidden.append(out["hidden"].half().cpu())

    result = {
        "texts": texts,
        "image_ids": torch.tensor(image_ids),
        "pooled": torch.cat(pooled),
        "encoder": encoder.model.name_or_path,
    }
    if save_hidden:
        result["hidden"] = torch.cat(hidden)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)
    return result
