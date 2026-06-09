#!/usr/bin/env python3
"""Build reference zip for Colab training.

Run once; only re-run when the reference manifest or detector changes.
A hash of the inputs is stored alongside the zip — if nothing changed, the
script exits immediately without touching the zip.
"""
import hashlib
import json
import zipfile
from pathlib import Path

MANIFEST       = Path("data/processed/reference_manifest.jsonl")
FLASH_LABELS   = Path("data/processed/flash_labels.jsonl")
PHASE_B_LABELS = Path("data/automation/phase_b/last_accepted.jsonl")
DETECTOR       = Path("outputs/models/mosaic_detector_v4.pt")
OUTPUT_ZIP     = Path("outputs/colab_reference_data.zip")
HASH_FILE      = Path("outputs/colab_reference_data.zip.hash")

MIN_IMAGES_PER_CITY = 200


def input_hash() -> str:
    h = hashlib.sha256()
    h.update(MANIFEST.read_bytes())
    if DETECTOR.exists():
        s = DETECTOR.stat()
        h.update(f"{s.st_size}:{s.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def main() -> None:
    current = input_hash()
    if OUTPUT_ZIP.exists() and HASH_FILE.exists() and HASH_FILE.read_text().strip() == current:
        size_mb = OUTPUT_ZIP.stat().st_size / 1e6
        print(f"Reference zip is up to date ({size_mb:.0f} MB, hash {current}). Nothing to do.")
        print(f"  Upload {OUTPUT_ZIP} to Drive only if you haven't already.")
        return

    print(f"Building {OUTPUT_ZIP} (hash {current})...")

    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    print(f"Manifest rows: {len(rows)}")

    from collections import Counter
    city_counts = Counter(r["city_code"] for r in rows)
    keep_cities = {c for c, n in city_counts.items() if n >= MIN_IMAGES_PER_CITY}

    # Only curated labels expand keep_cities — crowd-confirmed (phase_b) labels don't add
    # reference images for new cities since those cities rarely have enough mosaics for triplets.
    if FLASH_LABELS.exists():
        for r in [json.loads(l) for l in FLASH_LABELS.read_text().splitlines() if l.strip()]:
            keep_cities.add(r["city_code"])

    before = len(rows)
    rows = [r for r in rows if r["city_code"] in keep_cities and Path(r["image_path"]).exists()]
    print(f"Keeping {len(keep_cities)} cities (≥{MIN_IMAGES_PER_CITY} images or labeled): {len(rows)} rows (dropped {before - len(rows)})")

    remapped = [{**r, "image_path": f"{r['city_code']}/{r['invader_id']}/{Path(r['image_path']).name}"} for r in rows]

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (row, rrow) in enumerate(zip(rows, remapped), 1):
            if i % 500 == 0 or i == total:
                print(f"  {i}/{total}", end="\r")
            zf.write(row["image_path"], arcname=f"images/{rrow['image_path']}")

        zf.writestr("reference_manifest.jsonl", "\n".join(json.dumps(r) for r in remapped))

        if DETECTOR.exists():
            zf.write(DETECTOR, arcname=DETECTOR.name)
            print(f"\nAdded detector ({DETECTOR.stat().st_size / 1e6:.1f} MB)")
        else:
            print("\nWarning: detector weights not found, skipping")

    size_mb = OUTPUT_ZIP.stat().st_size / 1e6
    HASH_FILE.write_text(current)
    print(f"Wrote {OUTPUT_ZIP}  ({size_mb:.0f} MB)")
    print(f"Hash: {current}")
    print(f"\nUpload {OUTPUT_ZIP} to your Google Drive root.")


if __name__ == "__main__":
    main()
