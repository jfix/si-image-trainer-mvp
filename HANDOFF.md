# Training Repo — Handoff

Two files to merge into the training repo:

```
<training-repo-root>/
├── CLAUDE.md                       # merge CLAUDE.md.snippet into this
└── docs/
    └── ingestion-job.md            # drop in as-is
```

## Merging the snippet

If the training repo already has a `CLAUDE.md`, append the snippet as a
new top-level section (or fit it into your existing structure — the
content is self-contained).

If there's no `CLAUDE.md` yet, treat `CLAUDE.md.snippet` as the seed
for a new one. Rename to `CLAUDE.md` and add a short project overview
at the top (project name, stack: Modal + Python + DINOv2 + FAISS, etc.)
so Claude Code has the broader context.

## First Claude Code session in this repo

```
cd <training-repo>
git checkout -b phase-2-corpus-ingest
claude
```

Suggested opening prompt:

> Read CLAUDE.md and `docs/ingestion-job.md`. Then implement the
> nightly ingestion job step by step: cursor handling first, then
> quality filters, then the FAISS atomic update. Reuse existing
> embedder/FAISS utilities from this repo; don't reimplement them.
> Pause for review after each step.

## What's elsewhere

- The labelling system's design doc, schema, API contract, and
  verification UI all live in the **web app repo**.
- This repo only implements Phase 2 (corpus ingestion) and eventually
  Phase 4 (projection head).
- The boundary is the `/api/export/confirmed` HTTP endpoint. Don't
  reach into the web app's D1 directly.
