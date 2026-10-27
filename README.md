# From Words to Worlds: Text-to-Image Generation with Transformers + Diffusion

A text-to-image pipeline that conditions a pre-trained latent diffusion model
(Stable Diffusion v1.5) on Transformer text embeddings (CLIP ViT-L/14). We study
noise schedulers, sampling steps, classifier-free guidance, and text-encoder choice,
and evaluate with FID / IS and CLIP-based text–image alignment on COCO Captions.

## Status

| Milestone | Scope | Status |
|---|---|---|
| 1. Setup & data | COCO subset, preprocessing, text→embedding pipeline, 5 test samples | ✅ code ready |
| 2. Integration & baseline | guidance / scheduler / step sweeps | ⏳ |
| 3. Evaluation | FID, IS, CLIP score, sensitivity plots | ⏳ |
| 4. Report & presentation | report, gallery, slides | ⏳ |

## Repository layout

```
configs/default.yaml        all hyper-parameters (data, encoder, diffusion)
src/t2i/
  data.py                   COCO download, caption cleaning, subsetting, resize/crop, Dataset
  embeddings.py             CLIP / BERT text encoder wrapper + embedding cache
  generate.py               SD pipeline, scheduler swapping, guided sampling, grids
scripts/
  prepare_data.py           download + preprocess COCO val2017 subset
  encode_captions.py        cache text embeddings for all captions
  sample_prompts.py         5 prompt→image test runs (Milestone 1)
notebooks/
  01_data_and_text_embeddings.ipynb
  02_baseline_generation.ipynb
docs/proposal.md            1-page project proposal
outputs/milestone1/         committed sample images + metadata
```

## Running it

A GPU is required for generation (a Colab T4 with 16 GB is enough: SD v1.5 in fp16 takes about 4 GB
and produces a 512×512 image in about 3–5 s at 30 steps).

### Google Colab
1. Push this repo to GitHub, then set `REPO_URL` in the first cell of each notebook.
2. Open a notebook in Colab (*File → Open notebook → GitHub*) and set *Runtime → T4 GPU*.
3. Run all cells. With `USE_DRIVE = True`, data and outputs persist in `MyDrive/words-to-worlds`.

### University GPU / any Linux machine
```bash
git clone <repo-url> && cd words-to-worlds
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python scripts/prepare_data.py            # ~1.2 GB download, 5k images → data/processed/
python scripts/encode_captions.py         # CLIP embeddings → data/processed/embeddings_*.pt
python scripts/sample_prompts.py          # 5 samples → outputs/milestone1/
```

Common options:
```bash
python scripts/prepare_data.py --num-images 1000
python scripts/encode_captions.py --encoder bert-base-uncased --kind bert
python scripts/sample_prompts.py --scheduler dpmpp --steps 25 --guidance 5
```

## Pipeline

1. **Data**: COCO Captions 2017 val (5,000 images, ~25k human captions). Captions are
   whitespace-normalized, punctuated, deduplicated, and dropped if shorter than 3 words.
   Images are resized on the shorter side, center-cropped to 512×512, and normalized
   to [-1, 1] (the SD VAE input range).
2. **Text encoder**: `openai/clip-vit-large-patch14`. Captions are padded/truncated to 77
   tokens. The per-token output `(B, 77, 768)` is the U-Net's cross-attention context;
   the pooled EOS embedding `(B, 768)` is used for similarity and retrieval analysis.
3. **Diffusion**: `stable-diffusion-v1-5/stable-diffusion-v1-5` through 🤗 Diffusers. Schedulers
   (DDIM, DDPM, PNDM, Euler, Euler-A, DPM-Solver++) are swappable at runtime and share the
   model's trained β-schedule. Classifier-free guidance:
   `ε̂ = ε(x_t, ∅) + w·(ε(x_t, c) − ε(x_t, ∅))`.
4. **External conditioning**: `t2i.generate.generate(..., prompt_embeds=...)` feeds embeddings
   from our own encoder straight to the U-Net, which lets us swap or modify the text representation.

## Reproducibility
- All settings live in `configs/default.yaml`. The data subset uses `seed: 42`.
- Each generated image uses its own `torch.Generator` seeded with `seed + i`.
- `outputs/*/metadata.json` records the model, scheduler, steps, guidance, and seed for every image.

## Team
See `docs/proposal.md` for roles.
