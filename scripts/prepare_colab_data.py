#!/usr/bin/env python3
"""Package labeled flash images for Colab training.

Run before each training session. Produces a small zip (~500 MB) containing
only flash images + labels. The larger reference zip (built once by
prepare_colab_reference.py) does not need to be re-uploaded each time.
"""
import json
import zipfile
from pathlib import Path

FLASH_LABELS   = Path("data/processed/flash_labels.jsonl")
PHASE_B_LABELS = Path("data/automation/phase_b/last_accepted.jsonl")
OUTPUT_ZIP     = Path("outputs/colab_flash_data.zip")


def main() -> None:
    all_rows: list[dict] = []
    if FLASH_LABELS.exists():
        all_rows += [json.loads(l) for l in FLASH_LABELS.read_text().splitlines() if l.strip()]
    if PHASE_B_LABELS.exists():
        all_rows += [json.loads(l) for l in PHASE_B_LABELS.read_text().splitlines() if l.strip()]

    seen: set[str] = set()
    to_pack: list[tuple[Path, dict]] = []
    missing = 0
    for r in all_rows:
        p = Path(r["image_path"])
        if not p.exists():
            missing += 1
            continue
        if p.name in seen:
            continue
        seen.add(p.name)
        to_pack.append((p, {**r, "image_path": f"flash/{p.name}"}))

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    total = len(to_pack)
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (p, rr) in enumerate(to_pack, 1):
            if i % 500 == 0 or i == total:
                print(f"  {i}/{total}", end="\r")
            zf.write(p, arcname=f"flash_images/{rr['image_path']}")
        zf.writestr("flash_labels.jsonl", "\n".join(json.dumps(rr) for _, rr in to_pack))

    n_curated = sum(1 for _, rr in to_pack if "flash_id" not in rr)
    n_phase_b = total - n_curated
    size_mb = OUTPUT_ZIP.stat().st_size / 1e6
    print(f"\n{total} flash images ({missing} missing, {n_curated} curated + {n_phase_b} phase-b)")
    print(f"Wrote {OUTPUT_ZIP}  ({size_mb:.0f} MB)")
    print(f"\nUpload {OUTPUT_ZIP} to your Google Drive root.")


if __name__ == "__main__":
    main()
