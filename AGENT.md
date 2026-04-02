# AGENTS.md

## Purpose

This repository exists to build and evaluate a model that identifies **Invader mosaics** from flash images.

The system should take as input:

- a flash image
- the city code, such as `PA`, `LDN`, `BRU`

and return:

- the most likely mosaic identifier, such as `PA_123`
- a ranked top-k list of candidate identifiers
- a confidence label: `certainly`, `probably`, `maybe`, or `unknown`
- diagnostics explaining why the result was chosen

The system must use a **retrieval-first** architecture. Do not start with a flat classifier as the default solution.

---

## Primary objective

Build an MVP that is:

- measurable offline
- modular
- reproducible
- easy to extend when new mosaic identifiers appear

The design should make it possible to add new reference images and new identifiers without retraining the whole system from scratch.

---

## Core technical approach

Implement a two-stage pipeline.

### Stage 1: candidate retrieval
Given a query image:

1. detect or crop the mosaic region if possible
2. compute an embedding vector
3. search a city-scoped vector index of reference embeddings
4. retrieve top N nearest neighbors

### Stage 2: re-ranking
From the top N candidates:

1. aggregate image-level hits by identifier
2. run a more precise comparison for the top M identifiers
3. compute a calibrated confidence score
4. return the final prediction plus alternatives

Do **not** implement slow brute-force pairwise comparison against every reference image at inference time unless explicitly needed for benchmarking.

---

## Guiding principles

1. Start simple, measure honestly, then improve.
2. Prefer retrieval over closed-set classification for the MVP.
3. Treat cropping, calibration, and label quality as first-class concerns.
4. Avoid data leakage and misleading evaluation.
5. Optimize only after a working baseline exists.
6. Make experiments reproducible.
7. Reject uncertain cases instead of forcing a wrong identifier.

---

## Non-goals for the first iteration

Do not spend major time initially on:

- large-scale scraping from social media
- production frontend work
- distributed infrastructure
- multi-city global search when city is already known
- expensive platform engineering
- polished active-learning interfaces

Get the offline modeling pipeline working first.

---

## Expected deliverables

Each agent working on this repository should contribute toward the following deliverables.

### 1. Working code
The repository should contain:

- dataset preparation pipeline
- training scripts
- embedding generation scripts
- vector index builder
- inference pipeline
- evaluation scripts
- config files
- at least one CLI entry point for prediction

### 2. Documentation
The repository should contain:

- setup instructions
- data format description
- training and evaluation procedure
- architecture notes
- limitations and next steps

### 3. Reproducible experiments
There must be at least:

- one baseline experiment using pretrained embeddings only
- one improved experiment using fine-tuning or metric learning
- a comparison of metrics

### 4. Artifacts
Generated artifacts should include:

- model checkpoints
- vector indexes
- evaluation reports
- calibration reports
- confusion analysis
- hard negative examples

---

## Suggested repository layout

Use a structure close to the following unless there is a strong reason not to:

```text
project/
  README.md
  AGENTS.md
  pyproject.toml
  configs/
    base.yaml
    train_metric.yaml
    infer.yaml
  data/
    raw/
    interim/
    processed/
  notebooks/
    01_data_audit.ipynb
    02_embedding_baseline.ipynb
    03_error_analysis.ipynb
  src/
    data/
      schemas.py
      prepare_references.py
      prepare_queries.py
      split_dataset.py
      deduplicate.py
      augment.py
    detection/
      crop_mosaic.py
      rectify.py
    models/
      embedder.py
      siamese.py
      losses.py
      calibrator.py
    indexing/
      build_index.py
      search.py
      aggregate.py
    training/
      train_metric.py
      train_reranker.py
    inference/
      predict.py
      pipeline.py
    evaluation/
      metrics.py
      evaluate_retrieval.py
      evaluate_calibration.py
      error_analysis.py
    utils/
      io.py
      image.py
      logging.py
  scripts/
    run_prepare.sh
    run_train.sh
    run_eval.sh
  outputs/
    models/
    indexes/
    reports/