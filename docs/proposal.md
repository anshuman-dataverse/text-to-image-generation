# Project Proposal: From Words to Worlds
**Text-to-Image Generation Using Transformer and Diffusion Models**

**Team:** _Name 1, Name 2, Name 3_ · **Course:** Deep Learning · **Date:** _TBD_

## Goal
Build and analyze a text-to-image pipeline in which a Transformer text encoder
conditions a pre-trained latent diffusion model. Beyond producing images, we want to
measure how the conditioning choices (guidance strength, noise scheduler, number of
sampling steps, and text encoder) trade off image fidelity, diversity, and
prompt alignment.

## Dataset
**COCO Captions 2017 (val split).** It has 5,000 natural images, each with about 5
human-written captions (about 25k captions in total).
- *Why COCO:* it is the standard benchmark for text-to-image FID (reference statistics
  are comparable to published work), its everyday multi-object scenes test compositional
  alignment, and the 5k-image split fits the 1k–5k scope without the 18 GB train set.
- *Preprocessing:* caption cleaning (whitespace, punctuation, dedup, ≥3 words),
  resize + center-crop to 512×512, normalization to [-1, 1], and CLIP tokenization (77 tokens).
  Embeddings are cached to disk.
- *Usage:* captions serve as prompts. Real images serve as the reference set for
  FID and as positives for CLIP retrieval and alignment checks.

## Model
- **Text encoder:** CLIP ViT-L/14 (`openai/clip-vit-large-patch14`), a 12-layer
  Transformer whose per-token outputs form the cross-attention context of the U-Net.
  For comparison: BERT-base (mean-pooled), and optionally a GPT-style encoder (extension).
- **Image generator:** Stable Diffusion v1.5, a latent diffusion model (VAE with a
  4×64×64 latent and an 860M-parameter U-Net with cross-attention), run through 🤗 Diffusers in fp16.
- **Conditioning:** classifier-free guidance with scale *w*. Text embeddings can be passed
  directly as `prompt_embeds`, which lets us swap or modify the text representation.

## Experiments
1. Guidance scale *w* ∈ {1, 3, 5, 7.5, 10, 15}: fidelity vs. diversity.
2. Schedulers (DDPM, DDIM, PNDM, Euler-A, DPM-Solver++) × steps {10, 25, 50}: quality vs. speed.
3. Text encoder / conditioning: CLIP vs. modified embeddings (interpolation, prompt
   weighting); BERT embeddings via a learned projection (stretch goal).
4. *(Extension)* Style control through prompt modifiers or additional conditioning.

## Evaluation
- **Quantitative:** FID and Inception Score (torchmetrics) on about 1–2k generated images
  vs. COCO val references; CLIP score for text–image alignment; wall-clock time per image.
- **Qualitative:** a prompt-vs-image gallery, side-by-side parameter sweeps, and a
  failure-case analysis (counting, spatial relations, attribute binding, text rendering).

## Compute
Google Colab (T4/A100) and the university GPU cluster. SD v1.5 in fp16 takes about 4 GB of VRAM
and runs about 3–5 s per image on a T4, so 2k images per configuration is about 2 GPU-hours.

## Roles
| Member | Responsibilities |
|---|---|
| _Name 1_ | Data pipeline, preprocessing, text-encoder experiments |
| _Name 2_ | Diffusion integration, scheduler / guidance sweeps, gallery |
| _Name 3_ | Evaluation (FID / IS / CLIP score), analysis, ethics section |
| All | Report writing, presentation |

## Timeline
| Week | Milestone |
|---|---|
| 1–2 | Setup, COCO subset, text→embedding notebook, 5 test samples, repo |
| 3–4 | Baseline conditional generation; guidance and scheduler tuning |
| 5 | FID / IS / CLIP evaluation, sensitivity analysis, draft report |
| 7 (Dec 13) | Final report, gallery, slides, demo video |

## Ethical considerations (planned)
COCO's demographic and geographic skew; stereotypes LAION-trained models inherit;
misuse (deepfakes, non-consensual imagery); and copyright and artist-style questions. We keep the
safety checker in public demos and document observed biases in generated samples.
