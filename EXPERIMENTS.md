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
| City | Index size | Accuracy |
|------|-----------|----------|
| DJBA | 58 | **100%** (30/30) |
| MARS | 97 | **97%** (29/30) |
| ROM | 75 | **93%** (28/30) |
| BXL | 42 | **93%** (28/30) |
| LDN | 188 | **87%** (26/30) |
| PA | 1568 | **74%** (37/50) |

**Key finding:** Step-change in accuracy across all cities. Val loss 0.1080 (~16% better than previous best 0.1287) fully translated to flash accuracy this time. The combination of 604 diverse labeled images + 40 epochs unlocked a new performance level. Smaller cities are essentially solved; PA at 74% is the remaining challenge given its 1568-mosaic index.  
**Weights:** `outputs/models/dinov2_finetuned_aug_crop_labeled/`  
**Next:** Test other cities (LDN, MARS, DJBA, ROM). Consider hard negative mining to push PA further. Label more cities.

---

### 12. Hard negative mining (mined from base model) — 2026-05-12
**What:** Same architecture as experiment 11, but with hard negative mining: embed all reference images, find top-30 nearest wrong neighbours per invader, use those as negatives instead of random ones. Warm-started from bare DINOv2-small (mistake — see below).  
**Training:** Google Colab T4 GPU, 40 epochs. Best epoch: 38, val loss **0.1298** — worse than exp 11 (0.1080).  
**Why it failed:** Hard negative mining ran on the untrained base DINOv2 model before any fine-tuning. Without task-specific embeddings, "hard" negatives were effectively random — no better than the random baseline, and the extra complexity may have hurt.  
**Fix:** Mine hard negatives from the exp 11 checkpoint (which already knows what looks similar in the mosaic space). See experiment 13.  
**Weights:** not saved (worse than exp 11).

---

### 13. Hard negative mining from exp 12 weights (warm-start confusion) — 2026-05-12
**What:** Same as experiment 12, but intended to warm-start from exp 11 and mine from exp 11 weights. In practice, `WARMSTART_FROM` pointed to the Drive checkpoint which held exp 12 weights (0.1298), not exp 11 (0.1080). Mining therefore ran on the same mediocre model as exp 12.  
**Training:** Google Colab T4 GPU, 40 epochs. Best val loss: **0.1298** (never beat exp 12 baseline). Nothing saved.  
**Root cause:** Drive checkpoint was overwritten by exp 12 before exp 13 ran. Exp 11 weights only existed locally.  
**Fix:** Remove hard negative mining entirely. Add explicit `BEST_VAL_LOSS_BASELINE = 0.1080` in notebook config so future runs always target exp 11's score regardless of what's on Drive.  
**Weights:** not saved.

---

### 14. Exp 11 recipe + 836 labeled images (warm-start from Drive) — 2026-05-14
**What:** Return to the exp 11 recipe (no hard negatives). Labeled dataset grows from 604 → 836 images (PA: 508, LDN: 88, MARS: 47, BXL: 49, DJBA: 50, BAB: 47, ROM: 47). Warm-started from Drive checkpoint (exp 12/13 weights = 0.1298). `BEST_VAL_LOSS_BASELINE = 0.1080`.  
**Training:** Google Colab T4 GPU, 40 epochs. Best achieved: 0.1193 (epoch 21). Never beat baseline 0.1080. Nothing saved.  
**Why it failed:** Drive checkpoint was contaminated by exp 12/13 (0.1298). Starting val loss was 0.1307 — already worse than exp 11, couldn't recover in 40 epochs.  
**Fix:** Train from base DINOv2 (`WARMSTART_FROM = ''`), same as exp 11 did. More data should push below 0.1080.

---

### 15. Exp 11 recipe from scratch + 930 labeled images — 2026-05-14
**What:** Same as experiment 11 but with 930 labeled images (vs 604). Trains from base `facebook/dinov2-small` — no warm-start. `BEST_VAL_LOSS_BASELINE = 0.1080`. Label breakdown: PA (602), LDN (88), MARS (47), BXL (49), DJBA (50), BAB (47), ROM (47). 326 more images than exp 11.  
**Training:** Google Colab T4 GPU, 40 epochs (~560s/epoch). Best epoch: 12, val loss **0.1143**. Never beat baseline 0.1080. Nothing saved.  
**Val loss trajectory:** Fast initial descent (0.1462 → 0.1182 by epoch 8), then plateaued and oscillated between 0.114–0.125 for the remaining 32 epochs. Train loss kept falling to 0.006, creating a large train/val gap — classic overfitting.  
**Why it failed to improve:** Same recipe + more data was not enough to break through. The model memorised training triplets but didn't generalise better. The training regime itself has hit its ceiling.  
**Weights:** not saved (never beat 0.1080).  
**Next:** Change something — hard negative mining (now with exp 11 embeddings, which are meaningful), lower LR, or more PA labels.

---

### 16. Lower LR + warm-start from exp 11 — 2026-05-15
**What:** Same data as exp 15 (930 labels). Two changes: (1) LR halved from 1e-5 → 5e-6 to address the val loss oscillation seen in exp 15; (2) warm-start from exp 11 weights (val 0.1080) instead of base DINOv2. 20 epochs.  
**Training:** Google Colab T4 GPU, 20 epochs (~600s/epoch). Best epoch: **1**, val loss **0.0706** — new overall best by a large margin (35% improvement over exp 11's 0.1080). Saved to Drive.  
**Val loss trajectory:** 0.0706 → 0.0752 → … → 0.0882 → 0.0772. Best at epoch 1, slowly degraded, partial recovery at epoch 20. Never improved on epoch 1.  
**Why it worked:** warm-start from exp 11 put the model in a well-structured embedding space. One pass over the 930 labels at low LR was enough to improve substantially. Continued training caused overfitting — the model had memorised training triplets before the val distribution could be learned further.  
**Key learning:** with warm-start, the value is in the first few epochs. Future runs should stop at 5–10 epochs, not 20–40.  
**Weights:** saved (epoch 1, val 0.0706).  
**Flash accuracy (PA, seed 42, 50 images):** 19/50 (**38%**) — dramatically worse than exp 11's 74% on the same seed.  
**Why it failed:** Val loss measures reference↔reference discrimination. Warm-starting from exp 11 and continuing training pushed the embedding further into reference-space, causing flash images to land further from their correct match. Lower val loss did not mean better flash accuracy — the opposite.  
**Critical finding:** Reference-only val loss is not a reliable proxy for flash accuracy. A model can achieve a low val loss by over-specialising in reference↔reference matching while losing the flash→reference bridging capability that augmentation was supposed to provide.  
**Fix needed:** Include labeled flash images in the val loss computation (flash anchor → correct reference should be measured directly). Until then, always verify flash accuracy manually after each run.  
**Production model:** revert to exp 11 (val 0.1080, flash 74% PA).

---

### 17. Flash-aware val loss, from scratch, 930 labels — 2026-05-16
**What:** Same data as exp 15/16 (930 labels). Key change: val loss is now a combined metric — 50% reference↔reference triplet loss + 50% flash→reference triplet loss. 15% of labeled flash images are held out as a flash val set (not used in training). Trains from base `facebook/dinov2-small`, LR 1e-5, 40 epochs. `BEST_VAL_LOSS_BASELINE` reset to 1.0 (new metric, not comparable to prior runs).  
**Why:** Exp 16 showed that reference-only val loss (0.0706) can be completely decoupled from flash accuracy (38%). The combined metric forces the model to optimise for what we actually care about.  
**Training:** in progress.

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
- **`scripts/build_labeling_page.py`** — generates a labeling tool with pre-filled predictions, quick-select buttons, and a download button for exporting labels as JSONL. Includes "Show skipped" filter toggle to review previously skipped images.
- **`scripts/build_labeling_page_r2.py`** — same output as above, but sources images from Cloudflare R2 (S3-compatible object store) + D1 (SQLite) instead of local disk. Queries D1 via REST API, downloads images locally to `collect-si-live-data/images/`, filters out already-labeled images.
- **`scripts/build_contact_sheet.py`** — searchable dark-themed grid of all reference mosaics for a city, sorted numerically. Click any card to copy the invader ID to clipboard (with toast confirmation). Useful for quickly looking up IDs during labeling.
- **`scripts/prepare_colab_data.py`** — packages reference images, detector weights, and labeled flash images into a zip for Colab upload
- **`notebooks/train_colab.ipynb`** — self-contained Colab notebook with resume support (START_EPOCH / END_EPOCH). Pre-commit hook auto-updates title with date + commit hash when notebook is staged.
- **`outputs/index.html`** — master index page listing all cities with mosaic counts, links to contact sheets and labeling pages, and labeled-image count badges. Regenerated by a one-off script from manifest + flash_labels + outputs directory.

---

## Known issues / limitations

- **Offline eval is misleading:** reference-to-reference accuracy is not a reliable proxy for flash-to-reference accuracy. Use labeled flash images + validation pages for honest evaluation.
- **Validation is noisy:** 30–50 image manual samples have meaningful variance. A fixed held-out eval set would give more reliable progress tracking.
- **Confidence thresholds need recalibration** after each new embedder model — current thresholds were set on an earlier model and score distributions have shifted.
- **PA is a hard city:** 1570 mosaics in the index. At 74% accuracy (exp 11) many mosaics are still confused with visually similar neighbours. 602 labeled flash images is still a weak ratio (~38%).
- **No proper eval metrics:** we only track val loss and manual spot-checks. P@1 and Recall@K on a fixed held-out set would give reliable, reproducible progress measurement.
- **flash_labels.jsonl concatenation bug:** labels appended without a trailing newline produce `{...}{...}` on a single line. The deduplication script (`scripts/deduplicate_flash_labels.py`) handles this with regex split, but the root cause should be fixed in the labeling page's export logic.

---

## What to try next

### High priority
- [x] **Evaluate augmented model on flash images** — done, ~23% on manual validation of 30 PA images.
- [x] **Retrain with cropped reference images** — done, see experiment 6. Improved to ~30–43% on PA.
- [x] **Build a labeled flash dataset** — done, 930 images across 7 cities (PA: 602, LDN: 88, MARS: 47, BXL: 49, DJBA: 50, BAB: 47, ROM: 47).
- [x] **Include labeled flash images in training** — done, see experiments 7–11. PA 74%, smaller cities 87–100%.
- [x] **Collect more labeled flash images** — 930 labels collected. Smaller cities largely solved; PA needs more.
- [x] **Hard negative mining** — tried in exp 12/13/14. All failed. Abandoned for now.
- [x] **Lower LR + warm-start recipe** — exp 16 achieved val 0.0706 (35% better than exp 11). Best epoch was 1 — key learning: warm-start needs only 5–10 epochs, not 20–40.
- [x] **Validate exp 16 on flash images** — done: 19/50 (38%) on PA seed 42. Worse than exp 11 (74%). Val loss is not a reliable proxy for flash accuracy.
- [ ] **Fix val loss metric** — include labeled flash pairs in val loss computation so the metric directly measures flash→reference matching, not just reference↔reference. This is the most important architectural change needed before the next training run.
- [ ] **Recalibrate confidence thresholds** — score distributions have shifted significantly (0.07 range vs old 0.10 range). Must redo before deploying.
- [ ] **Build a proper eval framework** — compute P@1 and Recall@K on a fixed held-out set (not a 30-image spot-check). Script should run in < 60s and produce a single score to track per experiment.

### Medium priority
- [ ] **Hard negative mining (high priority now)** — exp 15 confirmed the recipe has hit its ceiling. Mine hard negatives from the exp 11 checkpoint (val 0.1080, the best model we have). Unlike exp 12/13, the embedding space is now meaningful so mined negatives will actually be hard.
- [ ] **More PA labels** — 602 PA labels vs 1570 mosaics is ~38% coverage. Each labeling session + retrain has consistently improved accuracy; continue toward 1000+ PA labels.
- [ ] **Label more cities** — VRS (39), WN (56), LY (48), AVI (41), NY (192), TK (135), HK (132) all have flash images and reference mosaics but no labeled data. 200+ labels per city would improve generalisation.
- [ ] **Unfreeze more layers** — currently only last 2 transformer blocks fine-tuned. With 900+ labeled images, unfreezing additional blocks may help capacity-limited learning.
- [ ] **Fix flash_labels.jsonl append bug** — labeling page export sometimes omits trailing newline, causing `{...}{...}` concatenated lines. Fix the export download logic to always emit `\n` after each record.

### Lower priority
- [ ] **Upgrade to DINOv2-base** — ~4× more parameters, typically 5–10% accuracy gain on fine-grained retrieval. Feasible on Colab with gradient checkpointing once label count reaches ~1500+.
- [ ] **Fixed held-out eval set** — a labelled test set never used for training, for reliable reproducible measurement across experiments.
- [ ] **Mosaic region quality** — some detector crops are partial or off-centre. A rectification step (homography estimation) could improve crop quality.
- [ ] **Multi-image aggregation** — some invaders have 3–4 reference images. Currently aggregated by max score; could try learned aggregation.
