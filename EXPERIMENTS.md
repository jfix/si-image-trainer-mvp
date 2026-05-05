# Experiment log

## Goal

Build a pipeline that takes a flash image and a city code and returns the most likely Space Invader mosaic identifier (e.g. `PA_392`) with a confidence label.

---

## Architecture

Retrieval-based pipeline:
1. **Detect** — YOLOv8 detector crops the mosaic region from the flash photo
2. **Embed** — embed the cropped image into a vector
3. **Search** — find nearest neighbours in a city-scoped vector index of reference images
4. **Rank** — aggregate hits by invader ID, apply role weights, return top-k with confidence label

---

## Experiments

### 1. Baseline embedder (handcrafted features)
**What:** Colour histograms + edge gradients + pooled gradients. Deterministic, no model weights needed.  
**Result:** top-1 16%, top-5 57%, MRR 0.30 on reference-to-reference eval (210 queries).  
**Verdict:** Weak but fast. Surprisingly competitive for colour-heavy pixel-art style content.

---

### 2. Mosaic detector — YOLOv8
**What:** Trained YOLOv8s on bounding-box annotations of mosaics in flash photos.  
**Iterations:**
- v1: 49 images — mAP50 0.924 but only 9 val images, unreliable metric. Best epoch: 1 (immediate overfit).
- v2: 99 images — mAP50 0.837, stable but low. Best epoch: 1 (still overfitting).
- v3: 279 images — mAP50 **0.981**, best epoch: 61 (stable convergence). This is the production detector.

**Key learning:** needed ~250+ images before training stopped overfitting immediately.  
**Weights:** `outputs/models/mosaic_detector_v3.pt`  
**Wired into pipeline:** yes — detector crops the flash image before embedding. Falls back to full image if nothing detected. `"used_crop": true` appears in diagnostics.

---

### 3. DINOv2 pretrained (no fine-tuning)
**What:** Replaced handcrafted embedder with `facebook/dinov2-small` via HuggingFace transformers. 384-dim CLS token embedding.  
**Result:** top-1 **11%** — worse than the baseline.  
**Why it failed:** DINOv2 was trained on natural images and extracts semantic features that don't discriminate well between different coloured tile grids. The baseline's colour histograms were actually better suited for this pixel-art content.  
**Verdict:** Pretrained features alone are not enough. Fine-tuning required.

---

### 4. DINOv2 fine-tuned with triplet loss (no augmentation)
**What:** Fine-tuned last 2 transformer blocks of DINOv2-small using triplet margin loss on reference library images. 1734 train invaders / 305 val invaders. 20 epochs, LR 1e-5.  
**Result:** top-1 **77.6%**, top-5 93.8%, MRR 0.85 on reference-to-reference eval. Best epoch: 14.  
**Flash image test:** **0% accuracy** on 50 PA flash images via validation page.  
**Why it failed on flash:** Training used reference images on both sides of the triplet. The model learned reference↔reference matching well but never saw a flash image, so flash embeddings land in a completely different region of the space. Classic domain gap.  
**Weights:** `outputs/models/dinov2_finetuned/`

---

### 5. DINOv2 fine-tuned with triplet loss + augmentation (in progress)
**What:** Same as experiment 4 but the anchor image gets heavy augmentation simulating flash photo conditions: colour jitter, gaussian blur, perspective distortion, affine transform, JPEG compression. Positive gets light augmentation. Negative is clean.  
**Augmentation library:** `albumentations` (faster than torchvision for geometric transforms).  
**Training:** Running on Google Colab T4 GPU (~9 min/epoch). Split across two 90-min sessions (epochs 1–10, then 11–20).  
**Progress so far:** epoch 7 best, val loss 0.1181 and still improving.  
**Weights (when done):** `outputs/models/dinov2_finetuned_aug/`

---

## Confidence calibration

After fine-tuning without augmentation, recalibrated confidence thresholds to match the new score range (0.45–0.67 vs old 0.72–0.94):

| Label | Threshold | Precision (on ref-to-ref eval) | n |
|---|---|---|---|
| certainly | score ≥ 0.59, margin ≥ 0.15 | 100% | 30 |
| probably | score ≥ 0.53, margin ≥ 0.075 | 91% | 57 |
| maybe | score ≥ 0.48, margin ≥ 0.0375 | 79% | 61 |
| unknown | below thresholds | 53% | 62 |

These will need recalibrating again after the augmented model is trained, since score distributions will shift.

---

## Tools built

- **`siit predict`** — CLI prediction for a single flash image
- **`siit build-index`** — builds city-scoped vector indexes from reference manifest
- **`siit evaluate`** — runs offline retrieval eval on held-out reference images
- **`scripts/build_validation_page.py`** — generates a self-contained HTML page showing flash images side-by-side with predicted reference images and confidence scores. Used for manual validation.
- **`scripts/prepare_colab_data.py`** — packages reference images into a zip for Colab upload
- **`notebooks/train_colab.ipynb`** — self-contained Colab notebook with resume support (START_EPOCH / END_EPOCH)

---

## Known issues / limitations

- **Offline eval is misleading:** reference-to-reference accuracy (77.6%) is not a reliable proxy for flash-to-reference accuracy (0% before augmentation). Need labeled flash images for honest evaluation.
- **No labeled flash dataset:** validation is currently manual (look at the HTML page and judge). To measure accuracy properly, need ~50–100 flash images with known identifiers.
- **Confidence thresholds need recalibration** after each new embedder model.
- **PA is a hard city:** 1700+ mosaics in the index. Even small embedding errors lead to wrong top-1 results. Smaller cities will likely perform better.

---

## What to try next

### High priority
- [ ] **Evaluate augmented model on flash images** — once Colab run finishes, rebuild index and regenerate validation page. This is the key test.
- [ ] **Build a labeled flash dataset** — manually identify ~50 flash images with known invader IDs. Essential for honest accuracy measurement and future training.
- [ ] **Recalibrate confidence thresholds** — after augmented model is trained.

### Medium priority
- [ ] **Include labeled flash images in training** — once the labeled flash dataset exists, add (flash image, reference image) pairs as positive triplets. This directly teaches the flash↔reference mapping rather than relying on augmentation alone.
- [ ] **Hard negative mining** — instead of random negatives, use the nearest-neighbour negatives (mosaics that look similar but are different). Should improve discrimination between visually similar mosaics.
- [ ] **Unfreeze more layers** — currently only last 2 blocks fine-tuned. With more data, unfreezing all layers or using a lower LR throughout might help.

### Lower priority
- [ ] **Mosaic region quality** — some detector crops are partial or off-centre. A rectification step (homography estimation) could improve crop quality.
- [ ] **Multi-image aggregation** — some invaders have 3–4 reference images. Currently aggregated by max score; could try learned aggregation.
- [ ] **City-specific tuning** — PA has very different characteristics from smaller cities. City-specific index tuning or separate models might help.
