# Generative Deep Learning Project: From Words to Worlds

**IE 7615 · Northeastern University · Fall 2026 · Group 1**

Text-to-image generation with a Transformer text encoder (CLIP ViT-L/14) and a latent diffusion model (Stable Diffusion v1.5). We study noise schedulers, sampling steps, classifier-free guidance and text-encoder choice, and evaluate with FID, Inception Score and CLIP score on COCO Captions.

## Team

- Anshuman Singh (002590892)
- Rohan Prakash Krishna Prakash (002317798)
- Sree Ramya Pasala (002301508)

## Milestones

| Milestone | Task | Status | Report |
|---|---|---|---|
| [Milestone 1](milestone1/) | Dataset, text→embedding pipeline, first samples | Code ready, Colab run pending | [Markdown](milestone1/report/Milestone1_Report.md) |
| Milestone 2 | Baseline generation, guidance and scheduler tuning | Planned | |
| Milestone 3 | FID / IS / CLIP evaluation, sensitivity analysis | Planned | |
| Milestone 4 | Final report, gallery, presentation (Dec 13) | Planned | |

---

## Milestone 1: Dataset, text embeddings and first samples

- **Dataset:** COCO Captions 2017 val, 5,000 images with about 5 captions each. Captions are cleaned (whitespace, punctuation, duplicates, under 3 words dropped). Images are resized on the shorter side, center-cropped to 512×512 and normalized to [-1, 1].
- **Text encoder:** `openai/clip-vit-large-patch14`. Its per-token output (77×768) is the U-Net's cross-attention context, and the pooled output (768) is used for similarity checks. All caption embeddings are cached.
- **Sanity checks:** same-image vs. different-image caption similarity, and CLIP text→image retrieval Recall@K.
- **Diffusion:** `stable-diffusion-v1-5/stable-diffusion-v1-5` (🤗 Diffusers, fp16), with 6 swappable schedulers and classifier-free guidance. Five test prompts run with fixed seeds. A check confirms that embeddings from our own encoder condition the U-Net the same way as the built-in prompt path.

### Structure

```
milestone1/
├── configs/default.yaml          data, text-encoder and diffusion settings
├── notebooks/milestone1_colab.ipynb   whole milestone on Google Colab (data on Google Drive)
├── report/                       Milestone1_Report.md
├── results/                      dataset_stats.json, embedding_checks.json, conditioning_check.json,
│                                 samples/ (5 images + metadata.json)
├── runs/                         console logs
├── screenshots/                  figures used in the report
└── scripts/
    ├── 01_prepare_data.py        download COCO val2017, clean, subset, resize
    ├── 02_encode_captions.py     cache CLIP (or BERT) embeddings for every caption
    ├── 03_embedding_checks.py    caption similarity + text→image retrieval
    ├── 04_sample_prompts.py      5 prompt→image samples + external-embedding check
    └── t2i/                      shared code: config, data, embeddings, generate, plots
```

### Setup

A GPU is needed. A Colab T4 is enough: SD v1.5 in fp16 uses about 4 GB and takes about 3–5 s per 512×512 image at 30 steps.

**Google Colab:** open `milestone1/notebooks/milestone1_colab.ipynb` (*File → Open notebook → GitHub*), select a T4 GPU and run all cells. COCO is stored once in `MyDrive/IE7615_Project2/data`, and the outputs are zipped to `MyDrive/IE7615_Project2/milestone1_outputs.zip`.

**University GPU / Linux:**

```bash
git clone https://github.com/anshuman-dataverse/generative-deep-learning-project.git
cd generative-deep-learning-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

```bash
cd milestone1/scripts
python 01_prepare_data.py                 # ~1.2 GB download -> data/processed/
python 02_encode_captions.py              # -> data/processed/embeddings_clip-vit-large-patch14.pt
python 03_embedding_checks.py             # -> results/embedding_checks.json
python 04_sample_prompts.py               # -> results/samples/, screenshots/milestone1_samples.png
```

Options: `01_prepare_data.py --num-images 1000`, `02_encode_captions.py --encoder bert-base-uncased --kind bert`, `04_sample_prompts.py --scheduler dpmpp --steps 25 --guidance 5`.

The COCO data is not in the repository (`milestone1/data/` is git-ignored). Script 01 downloads it.
