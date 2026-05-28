# Reference-Corpus Ingestion Job

The nightly job that pulls crowd-verified labels from the web app and
grows the FAISS reference corpus. This is Phase 2 of the labelling
system roadmap and the single biggest lever for retrieval accuracy.

## Implementation status (28 May 2026)

Implemented in this repo:

- `scripts/modal_phase_a.py` with Modal app `si-corpus-refresh`
- persistent cursor handling via Modal Dict (`si-corpus-refresh-cursor`)
- daily scheduled ingestion function (`ingest_confirmed_labels`)
- run summaries + staged labels persisted to Modal Volume (`si-corpus-refresh-state`)
- optional flash-image download for downstream quality filtering/index updates

Not implemented yet:

- quality filters from this document
- near-duplicate checks
- atomic FAISS publication + pointer flip
- inference reload orchestration

This keeps Phase A focused on reliable ingestion plumbing before index mutation.

## Phase B implementation status (28 May 2026)

Implemented scaffold:

- `scripts/phase_b_refresh.py`

Current behavior:

1. Pulls crowd-confirmed labels from `/api/export/confirmed` with a persisted cursor.
2. Downloads flash images to `data/interim/confirmed_flash_images`.
3. Applies sharpness filter before embedding.
4. Applies same-`mosaic_id` near-duplicate rejection via cosine similarity.
5. Writes:
   - merged manifest: `data/processed/reference_manifest_phase_b.jsonl`
   - accepted/rejected ledgers under `data/automation/phase_b/`
   - run summary under `data/automation/phase_b/reports/<run_id>.json`
6. Optional index build mode:
   - versioned output: `outputs/indexes_versions/<run_id>/`
   - atomic pointer file: `outputs/indexes_current.json`

### Run commands

Basic refresh (no index build):

```bash
LABELLING_SECRET=... \
python scripts/phase_b_refresh.py --config configs/base.yaml
```

Refresh + versioned index build + pointer flip:

```bash
LABELLING_SECRET=... \
python scripts/phase_b_refresh.py --config configs/base.yaml --build-index
```

Backfill run from since=0 with bounded batch:

```bash
LABELLING_SECRET=... \
python scripts/phase_b_refresh.py --config configs/base.yaml --since 0 --max-labels 200
```

Offline plumbing test without export API:

```bash
python scripts/phase_b_refresh.py --config configs/base.yaml --labels-file tmp/sample_labels.json
```

## Inputs checklist (operator)

Keep these values handy before running automation jobs.

### Required now (Phase A)

- Modal secret name: `si-archive-admin`
- Secret key in that secret: `LABELLING_SECRET`
- `LABELLING_SECRET` must match the web app export secret used by:
   - `GET /api/export/confirmed`
   - request header: `X-Export-Secret`

### Already used by deployment metadata flow

- `IMAGE_WALL_META_SECRET` (trainer side env var)
- image-wall Pages secret: `MODEL_META_SECRET`
- both must have the same value for `/api/model-meta` publish to succeed

### Planned for Phase B (expected additional inputs)

- Canonical reference-library source path/repo (for manifest refresh input)
- Policy values for quality filters (bbox area threshold, sharpness threshold)
- Dedup threshold (cosine similarity within the same `mosaic_id`)
- Publication target for refreshed indexes (where to write/version and pointer key)
- Inference reload strategy after pointer flip (polling interval vs explicit bounce)

Use this checklist as the single source for required runtime inputs when running or debugging corpus automation.

## Why this matters

Most retrieval failures in the current system are a **coverage**
problem, not an embedding-quality problem: each mosaic typically has
one reference image (often head-on, well-lit), while queries are
angled, low-light, or partially occluded "flash" photos. Adding 3–5
diverse references per mosaic typically beats anything you'd get from
fine-tuning at the current data scale.

## Contract with the web app

**Upstream**: `GET https://si-image-wall.pages.dev/api/export/confirmed?since=<unix_ts>`

Auth: shared secret in the `X-Admin-Secret` header. Secret is stored
in Modal secrets, not in source.

**Response shape** (per the web app repo's `docs/api.md`):
```json
{
  "labels": [
    {
      "flash_id": "f_abc",
      "mosaic_id": "PA_123",
      "flash_url": "https://r2.../flashes/abc.jpg",
      "confirmed_at": 1731792000
    }
  ],
  "cursor": 1731792345
}
```

**Reporting back**: after processing each label, POST to
`/api/poweruser/ref-status` (same auth) with:
```json
{ "flash_id": "f_abc", "status": "in_corpus" | "rejected", "reject_reason": "..." }
```

## Cursor handling

Persist the last successful `cursor` value in Modal storage (a small
`modal.Dict` or `modal.Volume` file). At job start:

```python
since = cursor_store.get("since", 0)
resp = fetch_confirmed(since=since)
# ...process labels...
cursor_store["since"] = resp["cursor"]   # only after successful processing
```

Idempotency: if the job dies mid-batch, re-fetching from the old
cursor is safe — duplicate ingest attempts must be no-ops because each
`flash_id` is already either `in_corpus`, `rejected`, or unknown.
Check before embedding.

## Quality filters

Applied **before** embedding (GPU time is the expensive bit).

1. **Bbox area**: use the existing vision-API bounding box already
   computed for the flash. Reject if `bbox_area / image_area < 0.05`
   (mosaic too small in frame).
2. **Sharpness**: variance of Laplacian on the bbox crop. Reject if
   below an empirically-tuned threshold (start at 100 on a 256×256
   crop, iterate). Use `cv2.Laplacian(gray, cv2.CV_64F).var()`.
3. **Near-duplicate within the same mosaic_id**: embed the candidate,
   compute cosine similarity against existing references **for this
   mosaic_id only**, reject if max sim > 0.95. Crucially **do not**
   compare against the whole index — visually similar but distinct
   mosaics would block each other.

Track reject counts by reason in a Modal-side metric so you can tune
thresholds.

## FAISS index update

The inference service holds the FAISS index in memory and serves
top-K queries. Updating it must be **atomic from the inference
service's point of view**.

Recommended approach with R2:

1. Read the current index from `r2://faiss/index.bin` and metadata
   from `r2://faiss/index.meta.json` (mosaic_id mapping).
2. Append new vectors and update the metadata.
3. Write to `r2://faiss/index-<timestamp>.bin` and `.meta.json`.
4. Flip a pointer: `r2://faiss/current.txt` contains the active
   filename. Inference service polls or is notified to reload.
5. After a grace period, garbage-collect old indices.

Avoid:
- Mutating `index.bin` in place — inference can read a half-written file.
- Long lock-holding patterns that block inference reads.

If your inference cls uses `@modal.enter()` to load the index at
container start, the simplest "reload" is to bounce the inference
container. With Modal's keep-warm pool, a rolling restart works:
spin up new containers reading `current.txt`, drain old ones.

## Schedule

```python
@app.function(
    schedule=modal.Period(days=1),
    secrets=[modal.Secret.from_name("si-archive-admin")],
    gpu="A10G",  # cheapest GPU that fits DINOv2 ViT-L inference
    timeout=3600,
)
def ingest_confirmed_labels():
    ...
```

Run at off-peak hours (e.g. 03:00 UTC). Keep `gpu` minimal — this is
embedding-only, no training.

## Backfill

One-shot run separate from the schedule: pull *all* historic confirmed
labels (since=0) and process them. Useful when first turning the
system on, or after extending the schema. Make it a separate
`@app.local_entrypoint()` rather than mangling the scheduled function.

## Metrics

Track per-run:
- Labels fetched
- Accepted (embedded + added)
- Rejected (broken out by reason)
- Index size before / after
- Wall time / GPU seconds

After each run, evaluate on a held-out gold set (curated separately,
not the same set used for training trust) and log top-1 accuracy.
That's how you know this is actually working.

## Out of scope here

- Training a projection head — Phase 4, separate job/file.
- The verification UI, consensus, trust scoring — all in the web app repo.
- Generating new reference images for mosaics that have *zero* refs —
  needs the poweruser flow in the web app to mint new IDs first.
