#!/usr/bin/env python3
"""Package reference images + manifest into a zip for Colab training."""
import json
import shutil
import sys
import zipfile
from pathlib import Path

MANIFEST = Path("data/processed/reference_manifest.jsonl")
OUTPUT_ZIP = Path("outputs/colab_training_data.zip")


def main() -> None:
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    print(f"Manifest rows: {len(rows)}")

    missing = [r for r in rows if not Path(r["image_path"]).exists()]
    if missing:
        print(f"Warning: {len(missing)} images not found locally, skipping.")
        rows = [r for r in rows if Path(r["image_path"]).exists()]

    # Remap paths to colab-friendly relative paths: city/invader_id/filename
    remapped = []
    for r in rows:
        p = Path(r["image_path"])
        rel = f"{r['city_code']}/{r['invader_id']}/{p.name}"
        remapped.append({**r, "image_path": rel})

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (row, rrow) in enumerate(zip(rows, remapped), 1):
            if i % 500 == 0 or i == total:
                print(f"  {i}/{total}", end="\r")
            zf.write(row["image_path"], arcname=f"images/{rrow['image_path']}")

        manifest_content = "\n".join(json.dumps(r) for r in remapped)
        zf.writestr("reference_manifest.jsonl", manifest_content)

    size_mb = OUTPUT_ZIP.stat().st_size / 1_000_000
    print(f"\nWrote {OUTPUT_ZIP}  ({size_mb:.0f} MB)  — {total} images")


if __name__ == "__main__":
    main()
