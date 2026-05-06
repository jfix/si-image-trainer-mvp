from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from si_image_trainer.models.detector import MosaicDetector
from si_image_trainer.models.embedder import make_embedder
from si_image_trainer.utils.io import ensure_parent, read_jsonl


def build_indexes(
    reference_manifest_path: str | Path,
    output_dir: str | Path,
    embedding_config: dict[str, int],
    retrieval_config: dict[str, object] | None = None,
    exclude_manifest_path: str | Path | None = None,
    detector_config: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    rows = read_jsonl(reference_manifest_path)
    excluded = {row["image_path"] for row in read_jsonl(exclude_manifest_path)} if exclude_manifest_path else set()
    rows = [row for row in rows if row["image_path"] not in excluded]
    allowed_roles = {
        str(role)
        for role in (retrieval_config or {}).get("stage1_allowed_roles", ["reference", "flash-reference"])
    }
    rows = [row for row in rows if str(row.get("role", "reference")) in allowed_roles]
    output_root = Path(output_dir)
    embedder = make_embedder(embedding_config)
    detector = MosaicDetector(**detector_config) if detector_config else None

    def embed_row(row: dict) -> np.ndarray:
        if detector:
            crop = detector.crop(row["image_path"])
            if crop is not None:
                return embedder.embed_image(crop)
        return embedder.embed_path(row["image_path"])

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["city_code"], []).append(row)

    manifests: list[dict[str, object]] = []
    for city_code, group in sorted(grouped.items()):
        print(f"  {city_code}: {len(group)} images", end="\r")
        vectors = np.vstack([embed_row(row) for row in group]).astype(np.float32)
        city_dir = output_root / city_code
        ensure_parent(city_dir / "placeholder")
        np.save(city_dir / "embeddings.npy", vectors)
        np.savez_compressed(city_dir / "index.npz", embeddings=vectors)
        (city_dir / "metadata.json").write_text(json.dumps(group, indent=2) + "\n", encoding="utf-8")
        manifests.append({"city_code": city_code, "count": len(group), "path": str(city_dir / "embeddings.npy")})
    return manifests
