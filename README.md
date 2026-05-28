# si-image-trainer-mvp

Retrieval-first MVP for identifying Invader mosaics from flash images.

This repository starts with a baseline that:

- scans the existing reference library into a manifest
- scans live flash images into a query manifest
- computes deterministic local image embeddings without downloading model weights
- builds a city-scoped vector index
- returns ranked predictions with confidence labels and diagnostics
- evaluates retrieval on held-out reference images

## Current assumptions

- Reference data lives at `/Users/jakob/Projects/si-reference-library/references`.
- Live flash images live at `/Users/jakob/Projects/collect-si-live-data/images`.
- Live-event metadata lives at `/Users/jakob/Projects/collect-si-live-data/data`.
- Live queries inherit city names and `flash_id` values from NDJSON event files, then resolve city codes via `place-mappings.json`.
- Offline evaluation currently uses held-out reference images when an identifier has at least two usable images.
- The baseline embedder is handcrafted and deterministic so it works offline immediately. The repository is structured so a pretrained or fine-tuned embedder can replace it later without changing the pipeline shape.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

scripts/run_prepare.sh
scripts/run_eval.sh
```

Run a prediction:

```bash
siit predict \
  --config configs/infer.yaml \
  --city AIX \
  --image /Users/jakob/Projects/collect-si-live-data/images/2026/03/29/0/0GzudQ9S.jpg
```

## Repository layout

The repository follows the structure described in [`AGENT.md`](/Users/jakob/Projects/si-image-trainer-mvp/AGENT.md) with working baseline code under `src/si_image_trainer/`.

## Data format

### Reference manifest

Each record contains:

- `city_code`
- `invader_id`
- `image_path`
- `role`
- `status`
- `source_type`

### Query manifest

Each record contains:

- `query_id`
- `image_path`
- `city_code` when available
- `city_name`
- `flash_id`
- `player`
- `observed_at` parsed from the date folder
- `label_invader_id` only when an external label manifest is supplied

## Evaluation

Current baseline metrics:

- top-1 accuracy
- top-k accuracy
- mean reciprocal rank
- confidence-label distribution

Reports are written to `outputs/reports/`.

## Next steps

- replace the baseline embedder with a pretrained vision backbone
- add learned re-ranking from labeled positives and hard negatives
- add calibrated probability estimates from a labeled query set
- add mosaic-region detection or rectification when enough training data exists

## Modal Phase A automation (ingestion plumbing)

Phase A is now started with a Modal job scaffold:

- script: `scripts/modal_phase_a.py`
- app name: `si-corpus-refresh`
- state volume: `si-corpus-refresh-state`
- cursor store: `si-corpus-refresh-cursor`

Current behavior:

- fetches confirmed labels from `GET /api/export/confirmed?since=...`
- uses a persistent cursor in Modal Dict
- stages labels and run summaries into a Modal Volume
- optionally downloads flash images for downstream processing

Run once:

```bash
modal run scripts/modal_phase_a.py::run_once
```

Backfill from zero cursor:

```bash
modal run scripts/modal_phase_a.py::run_once --since 0
```

Inspect saved cursor/state:

```bash
modal run scripts/modal_phase_a.py::show_state
```

Deploy scheduled daily job:

```bash
modal deploy scripts/modal_phase_a.py
```

Required secret in Modal:

- `si-archive-admin` containing `LABELLING_SECRET`

Important:

- This Phase A implementation is ingestion plumbing only.
- It does not yet mutate/publish FAISS indexes automatically.
- It does not alter model training logic.
