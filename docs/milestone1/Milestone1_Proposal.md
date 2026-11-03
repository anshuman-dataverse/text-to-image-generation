# From Words to Worlds: Text-to-Image Generation with Transformers and Diffusion

**Milestone 1 Proposal, IE 7615 (Fall 2026), Northeastern University, Group 1**\
GitHub: <https://github.com/anshuman-dataverse/text-to-image-generation>

## 1. Objective

We will build a text-to-image pipeline in which a Transformer text encoder conditions a pre-trained latent diffusion model, and then study how the conditioning settings change the results. The settings we plan to vary are the classifier-free guidance scale, the noise scheduler, the number of sampling steps and the text encoder. We will measure image quality with FID and Inception Score and prompt alignment with CLIP score, compare images by eye, and discuss dataset bias and misuse.

## 2. Dataset

We use the validation split of MS COCO Captions 2017: 5,000 photos of everyday scenes (80 object categories), each with about five human-written captions. It is the usual reference set for text-to-image FID, and its multi-object scenes test whether the model follows the whole prompt.

- Working subset: 5,000 image-caption pairs, using one caption per image picked at random (seed 42). The list is in `outputs/milestone1/results/pairs_manifest.csv`. All 25,000 cleaned captions are kept for the embedding checks.
- Cleaning: we normalized whitespace and punctuation, removed captions shorter than three words and exact duplicates (25,014 down to 25,000), and skipped annotator placeholder captions when choosing pairs. No image is missing and no caption is empty or repeated.
- Preprocessing: images are resized so the shorter side is 512, center-cropped to 512×512 and scaled to [-1, 1]. Captions are tokenized with CLIP's tokenizer and padded to 77 tokens. The longest caption has 58 tokens, so none are truncated.

## 3. Model

| Component | Model | Role |
|---|---|---|
| Text encoder | CLIP ViT-L/14 text encoder (123M) | turns a caption into 77×768 token embeddings |
| Denoiser | SD v1.5 U-Net (860M) | predicts the noise, attending to the text embeddings |
| Autoencoder | SD v1.5 VAE | maps a 512×512 image to a 64×64×4 latent and back |
| Sampler | DDIM, DDPM, PNDM, Euler, Euler-a, DPM++ | reverse diffusion with classifier-free guidance |

## 4. Plan

1. Milestone 1 (done): data preparation, environment (PyTorch, Diffusers and Transformers on a Colab T4), and the text-to-embedding pipeline with sanity checks. Captions of the same image are much closer than captions of different images (cosine 0.656 vs. 0.234), and CLIP finds the right image among 1,000 with Recall@1/5/10 of 54.9/81.1/89.7%. Five test images were generated, and our own embeddings reproduce SD's built-in prompt encoding exactly.
2. Milestone 2: baseline images for COCO captions, then sweeps over the guidance scale (1, 3, 5, 7.5, 10, 15) and over the six schedulers at 10, 25 and 50 steps.
3. Milestone 3: FID and IS (torchmetrics) on 1,000 to 2,000 generated images against COCO, CLIP score, sensitivity plots, and a comparison of CLIP and BERT as the text encoder.
4. Milestone 4: final analysis of fidelity against diversity, typical failures such as missing objects and colors on the wrong object, and ethics.

## 5. Team and responsibilities

| Member | Responsibilities |
|---|---|
| Anshuman Singh (002590892) | model integration, image generation, repository |
| Rohan Prakash Krishna Prakash (002317798) | dataset collection, cleaning, preprocessing and validation |
| Sree Ramya Pasala (002301508) | CLIP embedding pipeline, evaluation, visualizations, documentation |

## 6. Expected deliverables

Reproducible code and Colab notebooks, the 5,000-pair manifest, image galleries, FID, IS and CLIP score tables and plots, milestone reports, and the final report, slides and demo video (December 13).
