# CLAUDE.md — si-image-trainer-mvp

## Project overview

This repo builds and evaluates a system that identifies **Invader mosaics** from street flash photos.

**Stack**: Python · DINOv2 (ViT-S, fine-tuned) · FAISS · Modal · Cloudflare R2

**Pipeline**: flash photo → YOLOv8 crop → DINOv2 embedding → FAISS city-scoped index → top-K candidates → confidence score → `PA_123`

**Entry point**: `.venv/bin/siit predict --city PA --image path/to/flash.jpg`

See `EXPERIMENTS.md` for the full training history and `docs/ingestion-job.md` for Phase 2.

---

## Crowd-labelling integration

This repo owns the nightly ingestion job that pulls crowd-verified
labels from the **web app repo** and grows the FAISS reference corpus.

- **Spec**: see `docs/ingestion-job.md`
- **Upstream contract**: `GET /api/export/confirmed` on
  https://si-image-wall.pages.dev — defined in the web app repo's
  `docs/api.md`. Treat as a stable contract.
- **Authoritative design**: the labelling system's overall design lives
  in the web app repo at `docs/labelling-design.md` (§6 covers the
  reference-corpus pipeline this repo implements).

### Conventions

- The ingestion job is the **only** writer to the FAISS reference index
  besides the existing one-shot training scripts. Any update must be
  atomic from the inference service's point of view (write to a new
  R2 key, then flip a pointer).
- Quality filters are applied **before** embedding to avoid burning GPU
  time on rejects.
- The cursor (`since` timestamp) for `/api/export/confirmed` is
  persisted in this repo, not in the web app's D1. The web app side is
  stateless w.r.t. ingestion.

### Gotchas

- Near-duplicate detection runs against existing references **for the
  same mosaic_id only** — not against the whole index. Whole-index
  similarity will reject legitimate references for visually similar
  mosaics.
- The shared secret for `/api/export/confirmed` lives in Modal secrets,
  not in this repo. Don't commit it.
