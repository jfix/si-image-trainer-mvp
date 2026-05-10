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

### 1. Baseline embedder (handcrafted features) — 2026-04-02
**What:** Colour histograms + edge gradients + pooled gradients. Deterministic, no model weights needed.  
**Result:** top-1 16%, top-5 57%, MRR 0.30 on reference-to-reference eval (210 queries).  
**Verdict:** Weak but fast. Surprisingly competitive for colour-heavy pixel-art style content.

---

### 2. Mosaic detector — YOLOv8 — 2026-05-03
**What:** Trained YOLOv8s on bounding-box annotations of mosaics in flash photos.  
**Iterations:**
- v1: 49 images — mAP50 0.924 but only 9 val images, unreliable metric. Best epoch: 1 (immediate overfit).
- v2: 99 images — mAP50 0.837, stable but low. Best epoch: 1 (still overfitting).
- v3: 279 images — mAP50 **0.981**, best epoch: 61 (stable convergence). This is the production detector.

**Key learning:** needed ~250+ images before training stopped overfitting immediately.  
**Weights:** `outputs/models/mosaic_detector_v3.pt`  
**Wired into pipeline:** yes — detector crops the flash image before embedding. Falls back to full image if nothing detected. `"used_crop": true` appears in diagnostics.

---

### 3. DINOv2 pretrained (no fine-tuning) — 2026-05-05
**What:** Replaced handcrafted embedder with `facebook/dinov2-small` via HuggingFace transformers. 384-dim CLS token embedding.  
**Result:** top-1 **11%** — worse than the baseline.  
**Why it failed:** DINOv2 was trained on natural images and extracts semantic features that don't discriminate well between different coloured tile grids. The baseline's colour histograms were actually better suited for this pixel-art content.  
**Verdict:** Pretrained features alone are not enough. Fine-tuning required.

---

### 4. DINOv2 fine-tuned with triplet loss (no augmentation) — 2026-05-05
**What:** Fine-tuned last 2 transformer blocks of DINOv2-small using triplet margin loss on reference library images. 1734 train invaders / 305 val invaders. 20 epochs, LR 1e-5.  
**Result:** top-1 **77.6%**, top-5 93.8%, MRR 0.85 on reference-to-reference eval. Best epoch: 14.  
**Flash image test:** **0% accuracy** on 50 PA flash images via validation page.  
**Why it failed on flash:** Training used reference images on both sides of the triplet. The model learned reference↔reference matching well but never saw a flash image, so flash embeddings land in a completely different region of the space. Classic domain gap.  
**Weights:** `outputs/models/dinov2_finetuned/`

---

### 5. DINOv2 fine-tuned with triplet loss + augmentation — 2026-05-05 to 2026-05-06
**What:** Same as experiment 4 but the anchor image gets heavy augmentation simulating flash photo conditions: colour jitter, gaussian blur, perspective distortion, affine transform, JPEG compression. Positive gets light augmentation. Negative is clean.  
**Augmentation library:** `albumentations` (faster than torchvision for geometric transforms).  
**Training:** Google Colab T4 GPU, 20 epochs in one session (~9 min/epoch). Best epoch: 19, val loss 0.0963.  
**Flash accuracy (full reference images in index):** ~0% — model trained but inference still broken.  
**Flash accuracy (cropped reference images in index):** **~23% (7/30 manual validation)** — major improvement.  
**Key finding:** The model was always learning something useful. The problem was an asymmetry: flash images were cropped to the mosaic region at inference, but reference images in the index were embedded as full photos. Fixing `build_index.py` to also run the detector on reference images and embed the crop eliminated this mismatch and unlocked the accuracy.  
**Weights:** `outputs/models/dinov2_finetuned_aug/`  
**Next:** Retrain with cropped reference images as positives/negatives (not full images) to close the remaining training/inference gap.

---

### 6. DINOv2 fine-tuned with crop triplets (crop both sides) — 2026-05-06 to 2026-05-07
**What:** Same augmentation as experiment 5, but reference images are also detector-cropped before embedding — both at index time and during training (all three legs of each triplet use the crop if found, falling back to full image). Starts from bare DINOv2-small, not from the previous fine-tuned model.  
**Training:** Google Colab T4 GPU, 20 epochs (~285s/epoch). Best epoch: 14, val loss 0.1249.  
**Val loss higher than exp 5** (0.1249 vs 0.0963) — expected, task is harder with variable crop sizes.  
**Flash accuracy:** ~30% (9/30) on one sample, ~43% (13/30) on a second independent sample. Real accuracy likely 35–40% on PA.  
**Key finding:** Crop-to-crop training further improved over the asymmetric model. PA is still hard (1700+ mosaics) but results are meaningful with no labeled flash data at all.  
**Weights:** `outputs/models/dinov2_finetuned_aug_crop/`  
**Next:** Try warm-starting from `dinov2_finetuned_aug` instead of bare DINOv2, and/or add labeled flash images to training.

---

### 7. DINOv2 fine-tuned with crop triplets + labeled flash images — 2026-05-08
**What:** Same as experiment 6, but training mixes two triplet types: (1) reference crop triplets with flash augmentation on anchor, and (2) real labeled flash image as anchor → reference crop of correct invader → reference crop of different invader. 52 labeled flash images oversampled 20× per epoch (~840 flash triplets, 11% of total).  
**Training:** Google Colab T4 GPU, 20 epochs (~310s/epoch). Best epoch: 20 (val loss still improving at the end — more epochs likely helpful). Best val loss: 0.1310.  
**Flash accuracy (seed 42, same as previous tests):** 16/30 (~53%) — significant jump from 9/30 with previous model.  
**Flash accuracy (seed 99, fresh images):** 14/30 (~47%) — modest improvement from 13/30.  
**Key finding:** Labeled data helps substantially on images similar to the training set, but generalises weakly with only 52 examples. The gap between seen (53%) and fresh (47%) results suggests partial memorisation. Need 200+ labeled images for robust generalisation.  
**Weights:** `outputs/models/dinov2_finetuned_aug_crop_labeled/`  
**Next:** Collect more labeled flash images (different seeds in build_labeling_page.py) and retrain.

---

### 9. DINOv2 fine-tuned with 291 labeled flash images across 4 cities — 2026-05-08
**What:** Same architecture as experiment 7, but labeled dataset grows from 52 → 291 images across PA (107), LDN (88), MARS (47), BXL (49). ~248 flash pairs used for training after val split. FLASH_OVERSAMPLE=20 → ~4960 flash triplets per epoch (~5× more than exp 7).  
**Training:** Google Colab T4 GPU, 20 epochs (~370s/epoch). Best epoch: 20 (val loss still improving — more epochs helpful). Best val loss: 0.1314.  
**Flash accuracy:**
- BXL: 24/30 (**80%**) — up from 55% in exp 8. Largest improvement, likely because BXL labels were added and index is small (42 mosaics).
- PA: 10/30 (**33%**) — appears lower than exp 7 (~43–53%) but high variance in 30-image samples makes direct comparison unreliable.
**Key finding:** Multi-city labeling dramatically improves smaller cities. PA accuracy is hard to measure reliably with 30-image samples; a fixed held-out eval set is needed. Val loss still trending down at epoch 20.  
**Weights:** `outputs/models/dinov2_finetuned_aug_crop_labeled/`  
**Next:** Run more epochs (resume from epoch 20); collect more PA labels to improve the hardest city.

---

### 10. DINOv2 fine-tuned with 413 labeled flash images across 6 cities — 2026-05-09
**What:** Same architecture. Labeled dataset grows from 291 → 413 images: PA (229), LDN (88), MARS (47), BXL (49), DJBA (50 — newly added to reference library), BAB (47). ~350 flash pairs after val split. FLASH_OVERSAMPLE=20.  
**Training:** Google Colab T4 GPU, 20 epochs (~410s/epoch). Best epoch: 19. Best val loss: **0.1287** — new overall best.  
**Flash accuracy:**
- PA: 19/50 (**38%**) — slight decline from 42% in exp 9, on same seed. Within noise but suggests PA is plateauing.
- BXL: 21/30 (**70%**) — vs 80% in exp 9 but on a different seed, so not directly comparable.
**Key finding:** Val loss improved significantly (0.1314 → 0.1287) but flash accuracy on PA did not follow. Training run only used 413 labels — ROM (47) and the newly collected DJBA/BAB labels were added after the zip was prepared. Next retrain will include 604 labels across 7 cities.  
**Weights:** `outputs/models/dinov2_finetuned_aug_crop_labeled/`  
**Next:** Retrain with 604 labels (adding ROM, more PA). More PA labels needed — 276 labeled vs 1568 mosaics is a weak ratio.

---

### 11. DINOv2 fine-tuned with 604 labeled flash images, 40 epochs — 2026-05-10
**What:** Same architecture. Labeled dataset grows to 604 images across PA (276), LDN (88), MARS (47), BXL (49), DJBA (50), BAB (47), ROM (47). Training extended to 40 epochs. ~515 flash pairs after val split. Colab zip filtered to cities with ≥200 reference images or flash labels (9 cities, 6438 images) to keep zip size manageable.  
**Training:** Google Colab T4 GPU, 40 epochs (~460s/epoch). Best epoch: 25. Best val loss: **0.1080** — new overall best by a large margin.  
**Flash accuracy:**
- PA: 37/50 (**74%**) — up from ~38% in exp 10. Largest PA improvement yet.
- BXL: 28/30 (**93%**) — essentially solved for a city this size.
**Key finding:** The combination of 604 diverse labeled images + 40 epochs produced a step-change in accuracy. Val loss 0.1080 is ~16% better than previous best (0.1287) and this time fully translated to flash accuracy. PA at 74% is a major milestone given 1568 mosaics in the index.  
**Weights:** `outputs/models/dinov2_finetuned_aug_crop_labeled/`  
**Next:** Test other cities (LDN, MARS, DJBA, ROM). Consider hard negative mining to push PA further. Label more cities.

---

### 8. Cross-city generalisation test — 2026-05-08
**What:** Tested experiment 7 model (trained exclusively on PA data) on London and Brussels flash images with no city-specific labeled data or fine-tuning.  
**Results:**

| City | Approx. mosaics in index | Accuracy (20 images) |
|------|--------------------------|----------------------|
| PA   | ~1700                    | ~50%                 |
| LDN  | ~600                     | 60%                  |
| BXL  | ~200                     | 55%                  |

**Key finding:** The model generalises across cities without any city-specific training. Smaller indexes perform better — fewer candidates means less competition. The system is usable today for smaller cities as-is.  
**Next:** More PA labeled data will improve PA accuracy and likely lift other cities too via better general embeddings.

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
- **`scripts/build_labeling_page.py`** — generates a labeling tool with pre-filled predictions, quick-select buttons, and a download button for exporting labels as JSONL.
- **`scripts/prepare_colab_data.py`** — packages reference images, detector weights, and labeled flash images into a zip for Colab upload
- **`notebooks/train_colab.ipynb`** — self-contained Colab notebook with resume support (START_EPOCH / END_EPOCH)

---

## Known issues / limitations

- **Offline eval is misleading:** reference-to-reference accuracy (77.6%) is not a reliable proxy for flash-to-reference accuracy (0% before augmentation). Need labeled flash images for honest evaluation.
- **Small labeled dataset:** 52 labeled flash images helps on seen images but generalises weakly to new ones. Need 200+ diverse labeled images for meaningful generalisation.
- **Validation is noisy:** 30-image manual samples have high variance (13–16/30 on the same model). Need a fixed held-out eval set to measure progress reliably.
- **Confidence thresholds need recalibration** after each new embedder model.
- **PA is a hard city:** 1700+ mosaics in the index. Even small embedding errors lead to wrong top-1 results. Smaller cities will likely perform better.

---

## What to try next

### High priority
- [x] **Evaluate augmented model on flash images** — done, ~23% on manual validation of 30 PA images.
- [x] **Retrain with cropped reference images** — done, see experiment 6. Improved to ~30–43% on PA.
- [x] **Build a labeled flash dataset** — 52 images labeled via build_labeling_page.py. Modest generalisation improvement; need 200+ for meaningful gains.
- [x] **Include labeled flash images in training** — done, see experiment 7. ~53% on seen images, ~47% on new images.
- [ ] **Collect more labeled flash images** — target 200+ diverse PA images. Each labeling session + retrain compounds. Use `build_labeling_page.py` with different seeds.
- [ ] **Recalibrate confidence thresholds** — after enough labeled data exists to measure calibration honestly.

### Medium priority
- [ ] **Hard negative mining** — instead of random negatives, use the nearest-neighbour negatives (mosaics that look similar but are different). Should improve discrimination between visually similar mosaics.
- [ ] **Hard negative mining** — instead of random negatives, use the nearest-neighbour negatives (mosaics that look similar but are different). Should improve discrimination between visually similar mosaics.
- [ ] **Unfreeze more layers** — currently only last 2 blocks fine-tuned. With more data, unfreezing all layers or using a lower LR throughout might help.

### Lower priority
- [ ] **Mosaic region quality** — some detector crops are partial or off-centre. A rectification step (homography estimation) could improve crop quality.
- [ ] **Multi-image aggregation** — some invaders have 3–4 reference images. Currently aggregated by max score; could try learned aggregation.
- [ ] **City-specific tuning** — PA has very different characteristics from smaller cities. City-specific index tuning or separate models might help.
