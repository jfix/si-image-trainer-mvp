#!/usr/bin/env python3
"""Package reference images + manifest into a zip for Colab training."""
import json
import shutil
import sys
import zipfile
from pathlib import Path

MANIFEST      = Path("data/processed/reference_manifest.jsonl")
FLASH_LABELS  = Path("data/processed/flash_labels.jsonl")
PHASE_B_LABELS = Path("data/automation/phase_b/last_accepted.jsonl")
PHASE_B_IMAGES = Path("data/interim/confirmed_flash_images")
OUTPUT_ZIP    = Path("outputs/colab_training_data.zip")


MIN_IMAGES_PER_CITY = 200  # keep only cities with enough reference images to contribute useful triplets


def main() -> None:
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    print(f"Manifest rows: {len(rows)}")

    from collections import Counter
    city_img_counts = Counter(r["city_code"] for r in rows)
    keep_cities = {c for c, n in city_img_counts.items() if n >= MIN_IMAGES_PER_CITY}

    # Always keep cities with flash labels regardless of size
    if FLASH_LABELS.exists():
        flash_rows = [json.loads(l) for l in FLASH_LABELS.read_text().splitlines() if l.strip()]
        for r in flash_rows:
            keep_cities.add(r["city_code"])

    before = len(rows)
    rows = [r for r in rows if r["city_code"] in keep_cities]
    print(f"Keeping {len(keep_cities)} cities (≥{MIN_IMAGES_PER_CITY} images or labeled): {len(rows)} rows (dropped {before - len(rows)})")

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

        # Labeled flash images — merge curated flash_labels.jsonl + phase_b crowd-confirmed images.
        # Deduplicate by image filename so the same photo isn't added twice.
        all_flash_rows: list[dict] = []
        if FLASH_LABELS.exists():
            all_flash_rows += [json.loads(l) for l in FLASH_LABELS.read_text().splitlines() if l.strip()]
        if PHASE_B_LABELS.exists():
            all_flash_rows += [json.loads(l) for l in PHASE_B_LABELS.read_text().splitlines() if l.strip()]

        seen_filenames: set[str] = set()
        flash_remapped: list[dict] = []
        missing_flash = 0
        for r in all_flash_rows:
            p = Path(r["image_path"])
            if not p.exists():
                missing_flash += 1
                continue
            if p.name in seen_filenames:
                continue
            seen_filenames.add(p.name)
            rel = f"flash/{p.name}"
            zf.write(str(p), arcname=f"flash_images/{rel}")
            flash_remapped.append({**r, "image_path": rel})

        flash_manifest = "\n".join(json.dumps(r) for r in flash_remapped)
        zf.writestr("flash_labels.jsonl", flash_manifest)
        n_curated = sum(1 for r in flash_remapped if "flash_id" not in r)
        n_phase_b  = len(flash_remapped) - n_curated
        print(f"\nAdded {len(flash_remapped)} flash images ({missing_flash} missing, {n_curated} curated + {n_phase_b} phase-b)")

        detector_path = Path("outputs/models/mosaic_detector_v4.pt")
        if detector_path.exists():
            zf.write(detector_path, arcname="mosaic_detector_v4.pt")
            print(f"Added detector weights ({detector_path.stat().st_size / 1_000_000:.1f} MB)")
        else:
            print("Warning: detector weights not found, skipping")

    size_mb = OUTPUT_ZIP.stat().st_size / 1_000_000
    print(f"\nWrote {OUTPUT_ZIP}  ({size_mb:.0f} MB)  — {total} images")


if __name__ == "__main__":
    main()
