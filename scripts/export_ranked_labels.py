#!/usr/bin/env python
"""Build the rank-aware flash dataset from verify-UI confirmations.

The verify UI shows the production model's score-ordered top-5; the crowd taps
the correct one. When the correct answer was NOT rank-1 ("it's the second/third
one"), the candidate(s) ranked *above* it are precisely the confusers the model
wrongly preferred — the gold hard negatives. flash_labels.jsonl flattens this
away (a rank-less bag of rejected_candidates); this script recovers it from
si-image-wall's D1 (tasks.candidates_json is score-ordered + labels.mosaic_id is
the correct answer) and joins it to the local flash images.

Output: data/processed/flash_ranked.jsonl, one row per confirmed flash:
  {flash_id, city, image_path, correct, correct_rank,
   candidates: [mosaic_id...],            # score-ordered
   outranking_confusers: [mosaic_id...]}  # ranked above the correct answer
"""
from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--si-image-wall-dir", default=str(REPO.parent / "si-image-wall"))
    p.add_argument("--d1-db", default="si-events")
    p.add_argument("--flash-zip", default=str(REPO / "outputs/colab_flash_data.zip"))
    p.add_argument("--ref-manifest", default=str(REPO / "data/processed/reference_manifest.jsonl"))
    p.add_argument("--cached-d1", default=None, help="reuse a saved D1 --json dump instead of querying")
    p.add_argument("--out", default=str(REPO / "data/processed/flash_ranked.jsonl"))
    return p.parse_args()


def query_d1(si_dir: str, db: str) -> list[dict]:
    sql = ("SELECT t.flash_id, t.city, t.candidates_json, l.mosaic_id AS correct "
           "FROM tasks t JOIN labels l ON l.flash_id = t.flash_id WHERE t.status='confirmed'")
    out = subprocess.run(
        ["npx", "wrangler", "d1", "execute", db, "--remote", "--json", "--command", sql],
        cwd=si_dir, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)[0]["results"]


def main():
    args = parse_args()

    # flash_id -> (image_path, city) from the local crowd-confirmed labels
    fid_meta: dict[str, dict] = {}
    with zipfile.ZipFile(args.flash_zip) as z:
        for line in z.read("flash_labels.jsonl").decode().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("flash_id"):
                fid_meta[str(r["flash_id"])] = {"image_path": r["image_path"], "city": r["city_code"]}

    ref_inv = set()
    for line in Path(args.ref_manifest).read_text().splitlines():
        if line.strip():
            ref_inv.add(json.loads(line)["invader_id"])

    rows = (json.load(open(args.cached_d1))[0]["results"] if args.cached_d1
            else query_d1(args.si_image_wall_dir, args.d1_db))

    out_rows, ranks = [], {}
    n_err = n_pairs = 0
    for r in rows:
        fid = str(r["flash_id"])
        meta = fid_meta.get(fid)
        if not meta:
            continue  # no local flash image
        cands = [c["mosaic_id"] for c in json.loads(r["candidates_json"])]  # score-ordered
        if r["correct"] not in cands:
            continue
        rank = cands.index(r["correct"]) + 1
        ranks[rank] = ranks.get(rank, 0) + 1
        confusers = [c for c in cands[:rank - 1] if c in ref_inv]
        if rank >= 2:
            n_err += 1
            n_pairs += len(confusers)
        out_rows.append({
            "flash_id": fid,
            "city": meta["city"],
            "image_path": meta["image_path"],
            "correct": r["correct"],
            "correct_rank": rank,
            "candidates": cands,
            "outranking_confusers": confusers,
        })

    out_path = Path(args.out)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in out_rows))
    print(f"wrote {len(out_rows)} rows → {out_path}")
    print("rank distribution:", {k: ranks[k] for k in sorted(ranks)})
    print(f"error cases (rank>=2): {n_err}  |  rank-aware hard-neg pairs: {n_pairs}")


if __name__ == "__main__":
    main()
