# Generative Deep Learning Project: From Words to Worlds

**IE 7615 · Northeastern University · Fall 2026 · Group 1**

Text-to-image generation with a Transformer text encoder (CLIP ViT-L/14) and a latent diffusion model (Stable Diffusion v1.5). We study noise schedulers, sampling steps, classifier-free guidance and text-encoder choice, and evaluate with FID, Inception Score and CLIP score on COCO Captions.

## Team

| Member | NUID | Responsibilities |
|---|---|---|
| Anshuman Singh | 002590892 | model integration, diffusion image generation, repository management |
| Rohan Prakash Krishna Prakash | 002317798 | dataset collection, cleaning, preprocessing and validation |
| Sree Ramya Pasala | 002301508 | CLIP embedding pipeline, evaluation, visualizations, technical documentation |

## Milestones

| Milestone | Task | Status | Report |
|---|---|---|---|
| [Milestone 1](milestone1/) | Dataset, text→embedding pipeline, first samples | Done | [Report (Markdown)](milestone1/report/Milestone1_Report.md) · [Report (PDF)](milestone1/report/Milestone1_Report.pdf) · [One-page proposal (PDF)](milestone1/report/Milestone1_Proposal.pdf) · [Submission (proposal + report, PDF)](milestone1/report/Milestone1_Submission.pdf) |
| [Milestone 2](milestone2/) | Baseline generation, guidance and scheduler tuning | Code ready, Colab run pending | |
| Milestone 3 | FID / IS / CLIP evaluation, sensitivity analysis | Planned | |
| Milestone 4 | Final report, gallery, presentation (Dec 13) | Planned | |

---

## Milestone 1: Dataset, text embeddings and first samples

- **Dataset:** COCO Captions 2017 val, 5,000 images with 25,014 captions (about 5 each). Captions are cleaned (whitespace and punctuation normalized; captions under 3 words and exact duplicates dropped, leaving 25,000). Images are resized on the shorter side, center-cropped to 512×512 and normalized to [-1, 1].
- **Image–caption pair subset:** 5,000 pairs, one caption per image picked at random with seed 42, with annotator placeholders such as "I am unable to see an image above." skipped. The manifest is [`results/pairs_manifest.csv`](milestone1/results/pairs_manifest.csv) (image_id, caption_id, file_name, caption). [`results/subset_checks.json`](milestone1/results/subset_checks.json) records 0 missing images, 0 empty captions and 0 duplicate image or caption IDs.
- **Text encoder:** `openai/clip-vit-large-patch14`. Its per-token output (77×768) is the U-Net's cross-attention context, and the pooled output (768) is used for similarity checks. All caption embeddings are cached.
- **Sanity checks:** same-image vs. different-image caption similarity, and CLIP text→image retrieval Recall@K.
- **Diffusion:** `stable-diffusion-v1-5/stable-diffusion-v1-5` (🤗 Diffusers, fp16), with 6 swappable schedulers and classifier-free guidance. Five test prompts run with fixed seeds. A check confirms that embeddings from our own encoder condition the U-Net the same way as the built-in prompt path.

Pipeline: `text prompt → CLIP tokenizer → CLIP text encoder → 77×768 embeddings → U-Net + DDIM scheduler (classifier-free guidance) → VAE decoder → 512×512 image`

### Results

| Check | Result |
|---|---|
| Caption cosine similarity, same image vs. different images | 0.656 vs. 0.234 (pairwise AUC 0.985) |
| CLIP text→image Recall@1 / @5 / @10 (1,000 images) | 54.9% / 81.1% / 89.7% (chance 0.1% / 0.5% / 1%) |
| Our embeddings vs. SD's built-in encoding | identical (max abs diff 0.0), pixel-identical images |

### Five test runs

SD v1.5, DDIM, 30 steps, guidance 7.5, 512×512, fp16 on a Colab Tesla T4 (6.7 s/image). Metadata: [`results/samples/metadata.json`](milestone1/results/samples/metadata.json).

| Seed | Prompt | Image | Observation |
|---|---|---|---|
| 0 | A man riding a bicycle down a city street. | <img src="milestone1/results/samples/sample_00.png" width="180"> | correct and photorealistic |
| 1 | A plate of pasta with tomatoes and basil on a wooden table. | <img src="milestone1/results/samples/sample_01.png" width="180"> | all four objects present |
| 2 | Two dogs playing with a frisbee in a grassy park. | <img src="milestone1/results/samples/sample_02.png" width="180"> | two dogs, but the frisbee is missing (object omission) |
| 3 | A red double-decker bus parked next to a blue car. | <img src="milestone1/results/samples/sample_03.png" width="180"> | no blue car: the blue leaks onto a second bus (attribute leakage) |
| 4 | A cat sitting on a windowsill watching the rain at sunset. | <img src="milestone1/results/samples/sample_04.png" width="180"> | cat and rain are right; "sunset" is ignored |

Single-subject prompts work well. Quality drops as prompts add objects and attributes (omission, attribute binding, a dropped third concept), the known weakness of CLIP-conditioned diffusion on compositional prompts. Milestone 2 tests guidance and scheduler settings against it. Small faces, hands and any text in the image are also unreliable at 512×512.

### Structure

```
milestone1/
├── configs/default.yaml          data, text-encoder and diffusion settings
├── notebooks/milestone1_colab.ipynb   whole milestone on Google Colab (data on Google Drive)
├── report/                       Milestone1_Report.md/.pdf, Milestone1_Proposal.md/.pdf (one page)
├── results/                      dataset_stats.json, pairs_manifest.csv, subset_checks.json,
│                                 embedding_checks.json, conditioning_check.json,
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

**Google Colab (recommended):**

1. Open [`milestone1/notebooks/milestone1_colab.ipynb`](milestone1/notebooks/milestone1_colab.ipynb) in Colab (*File → Open notebook → GitHub*, paste the repository URL).
2. *Runtime → Change runtime type → T4 GPU*.
3. *Runtime → Run all*, and allow Google Drive access when asked. The notebook clones this repository, installs `requirements.txt`, downloads COCO (about 1 GB, stored once in `MyDrive/IE7615_Project2/data`) and runs scripts 01–04.
4. The outputs are zipped to `MyDrive/IE7615_Project2/milestone1_outputs.zip`. To update the repository, unzip it into `milestone1/` and commit.

No Hugging Face token is needed: CLIP and SD v1.5 are public.

**University GPU / Linux:**

```bash
git clone https://github.com/anshuman-dataverse/text-to-image-generation.git
cd text-to-image-generation
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

The COCO data is not in the repository (`milestone1/data/` is git-ignored). Script 01 downloads it. Script 01 needs no GPU, so the dataset step can be checked on any machine.

The report PDFs are built from the Markdown with `milestone1/report/build_pdfs.sh` (pandoc + Google Chrome).

---

## Milestone 2: baseline generation, guidance and scheduler sweeps

Three experiments on COCO captions from the Milestone 1 pair manifest. Prompt *i* uses seed *i* in every experiment, so only the varied setting changes. Every image is scored with CLIP score (ViT-L/14, 100 · max(cos, 0)) against its caption.

| Exp. | Script | Varies | Prompts × seeds | Measures |
|---|---|---|---|---|
| 1 | `01_baseline.py` | nothing (DDIM, 30 steps, w = 7.5) | 200 × 1 | CLIP score, vs. the real COCO photo |
| 2 | `02_guidance_sweep.py` | w ∈ {1, 3, 5, 7.5, 10, 15} | 32 × 2 | CLIP score, seed diversity, saturation |
| 3 | `03_scheduler_sweep.py` | DDIM, DDPM, PNDM, Euler, Euler-a, DPM++ × {10, 25, 50} steps | 24 × 1 | CLIP score, s/image, convergence to 50 steps |

**Run:** open [`milestone2/notebooks/milestone2_colab.ipynb`](milestone2/notebooks/milestone2_colab.ipynb) in Colab with a T4 GPU and *Run all* (about 1.5–2 h). Generated images go to `MyDrive/IE7615_Project2/milestone2_outputs` and are skipped if already there, so a disconnected session resumes. Results, figures and logs are zipped to `MyDrive/IE7615_Project2/milestone2_outputs.zip`; unzip it into `milestone2/` and commit.

Locally: `cd milestone2/scripts && python 01_baseline.py` (and 02, 03). `--n` sets the number of prompts and `--score-only` re-scores existing images without a GPU run. Settings are in `milestone2/configs/default.yaml`.
