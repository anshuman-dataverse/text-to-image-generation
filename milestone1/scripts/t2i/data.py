"""COCO Captions download, cleaning, subsetting and image preprocessing."""

import json
import random
import re
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

COCO_URLS = {
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def download_coco(root: Path, split: str = "val2017") -> None:
    """Download and extract COCO images for `split` plus caption annotations."""
    root.mkdir(parents=True, exist_ok=True)
    for key in (split, "annotations"):
        zip_path = root / Path(COCO_URLS[key]).name
        _download(COCO_URLS[key], zip_path)
        marker = root / (split if key == split else "annotations")
        if not marker.exists():
            print(f"Extracting {zip_path.name} ...")
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(root)


def clean_caption(text: str) -> str:
    text = text.strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(" .") + "."
    return text[0].upper() + text[1:]


def load_coco_captions(root: Path, split: str = "val2017", min_words: int = 3) -> list[dict]:
    """Return one record per image: {image_id, file_name, width, height, captions}."""
    with open(root / "annotations" / f"captions_{split}.json") as f:
        ann = json.load(f)

    caps = defaultdict(list)
    for a in ann["annotations"]:
        c = clean_caption(a["caption"])
        if len(c.split()) >= min_words:
            caps[a["image_id"]].append(c)

    records = []
    for img in ann["images"]:
        if caps[img["id"]]:
            records.append({
                "image_id": img["id"],
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "captions": sorted(set(caps[img["id"]])),
            })
    return sorted(records, key=lambda r: r["image_id"])


def subset(records: list[dict], n: int, seed: int = 42) -> list[dict]:
    if n >= len(records):
        return records
    rng = random.Random(seed)
    return sorted(rng.sample(records, n), key=lambda r: r["image_id"])


def resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    """Resize shorter side to `size`, then center-crop to size x size."""
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def preprocess_images(records: list[dict], src_dir: Path, out_dir: Path, size: int) -> list[dict]:
    """Write resized/cropped RGB JPEGs to out_dir/images and captions.jsonl."""
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    for r in tqdm(records, desc="preprocess"):
        dst = img_dir / r["file_name"]
        if not dst.exists():
            try:
                img = Image.open(src_dir / r["file_name"]).convert("RGB")
            except (OSError, FileNotFoundError) as e:
                print(f"skip {r['file_name']}: {e}")
                continue
            resize_center_crop(img, size).save(dst, quality=95)
        kept.append(r)

    with open(out_dir / "captions.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    return kept


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


class CocoCaptionDataset(Dataset):
    """Preprocessed images normalized to [-1, 1] (SD VAE input range) with a caption.

    caption_mode: "first" returns captions[0]; "random" samples one per access.
    """

    def __init__(self, processed_dir: Path, caption_mode: str = "first"):
        self.dir = Path(processed_dir)
        self.records = read_jsonl(self.dir / "captions.jsonl")
        self.caption_mode = caption_mode
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        img = Image.open(self.dir / "images" / r["file_name"]).convert("RGB")
        caps = r["captions"]
        cap = random.choice(caps) if self.caption_mode == "random" else caps[0]
        return {"pixel_values": self.tf(img), "caption": cap, "image_id": r["image_id"]}


def to_pil(t: torch.Tensor) -> Image.Image:
    """Inverse of the [-1, 1] normalization, for visualization."""
    arr = ((t.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)
