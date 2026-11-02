# From Words to Worlds: Text-to-Image Generation with Transformers and Diffusion

**IE 7615 · Northeastern University · Fall 2026 · Group 1 · Milestone 1 Proposal**\
GitHub: <https://github.com/anshuman-dataverse/text-to-image-generation>

## 1. Objective

Build a text-to-image pipeline in which a Transformer text encoder conditions a pre-trained latent diffusion model, then measure how the conditioning choices (classifier-free guidance scale, noise scheduler, number of sampling steps, text encoder) trade off image fidelity, diversity and prompt alignment. We evaluate quantitatively (FID, Inception Score, CLIP score) and qualitatively, and discuss bias and misuse risks.

## 2. Dataset

**MS COCO Captions 2017, validation split.** We use 5,000 natural photos of everyday multi-object scenes (80 categories). They are the standard reference set for text-to-image FID.

- **Working subset: 5,000 image–caption pairs**, one caption per image picked at random with seed 42 (`results/pairs_manifest.csv` lists image ID, caption ID and caption). The full pool of 25,000 cleaned captions (about 5 per image) is kept for the embedding checks.
- **Cleaning:** whitespace and punctuation normalized, captions under 3 words and duplicates removed (25,014 → 25,000), and annotator placeholder captions excluded from the pairs. Verified: 0 missing images, 0 empty captions, 0 duplicate IDs.
- **Preprocessing:** shorter side resized to 512, center-cropped to 512×512 and normalized to [−1, 1]. Captions use CLIP BPE tokens padded to 77 (max 58, none truncated).

## 3. Model architecture

| Component | Model | Role |
|---|---|---|
| Text encoder | CLIP ViT-L/14 text transformer (123M) | caption → 77×768 token embeddings |
| Denoiser | Stable Diffusion v1.5 U-Net (860M) | predicts noise, cross-attending to the text embeddings |
| Autoencoder | SD v1.5 VAE | 512×512 image ↔ 64×64×4 latent |
| Sampling | DDIM, DDPM, PNDM, Euler, Euler-A, DPM-Solver++ + classifier-free guidance | reverse diffusion |

Pipeline: *text prompt → CLIP tokenizer → text encoder → embeddings → U-Net + scheduler → VAE decoder → image.*

## 4. Methodology

1. **Milestone 1 (done):** prepare data, set up the environment (PyTorch, 🤗 Diffusers, Colab T4), build the text→embedding pipeline and check it. Same-image caption similarity is 0.656 vs. 0.234 for different images, and text→image Recall@1/5/10 is 54.9 / 81.1 / 89.7% among 1,000 images. We ran 5 seeded test prompts, and our own embeddings drive the U-Net identically to SD's built-in encoder.
2. **Milestone 2:** baseline generation over COCO captions; sweep guidance scale w ∈ {1, 3, 5, 7.5, 10, 15} and schedulers × steps {10, 25, 50}.
3. **Milestone 3:** FID and IS (torchmetrics) on 1–2k generated images against COCO references, plus CLIP score, sensitivity plots and a text-encoder comparison (CLIP vs. BERT).
4. **Milestone 4:** final analysis of fidelity vs. diversity, failure modes (object omission, attribute leakage) and ethics.

## 5. Team and responsibilities

| Member | Responsibilities |
|---|---|
| Anshuman Singh (002590892) | model integration, diffusion image generation, repository management |
| Rohan Prakash Krishna Prakash (002317798) | dataset collection, cleaning, preprocessing and validation |
| Sree Ramya Pasala (002301508) | CLIP embedding pipeline, evaluation, visualizations, technical documentation |

## 6. Expected deliverables

Reproducible code and Colab notebook; cleaned 5,000-pair COCO subset manifest; generated image galleries for each configuration; FID / IS / CLIP-score tables and sensitivity plots; milestone reports, final report, slides and demo video (Dec 13).
