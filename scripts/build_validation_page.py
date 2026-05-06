#!/usr/bin/env python3
"""Build a self-contained HTML page for validating flash image predictions."""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from si_image_trainer.inference.predict import predict_one
from si_image_trainer.models.detector import MosaicDetector
from si_image_trainer.utils.io import load_yaml


REFERENCE_ROOT = Path("/Users/jakob/Projects/si-reference-library/references")


def pil_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{data}"


def img_to_b64(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    suffix = p.suffix.lower().lstrip(".")
    mime = "png" if suffix == "png" else "jpeg"
    data = base64.b64encode(p.read_bytes()).decode()
    return f"data:image/{mime};base64,{data}"


def best_reference_image(invader_id: str) -> Path | None:
    meta_path = REFERENCE_ROOT / invader_id.split("_")[0] / invader_id / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    # Prefer grosplan, then reference photo
    for role in ("grosplan", "reference"):
        for img in meta.get("images", []):
            if img.get("role") == role:
                candidate = REFERENCE_ROOT.parent / img["local_path"]
                if candidate.exists():
                    return candidate
    return None


def build_page(config: dict, queries: list[dict], output_path: Path) -> None:
    detector = MosaicDetector(**config["detector"]) if config.get("detector") else None
    print(f"Running predictions on {len(queries)} images...")
    results = []
    for i, q in enumerate(queries, 1):
        print(f"  {i}/{len(queries)}  {Path(q['image_path']).name}", end="\r")
        pred = predict_one(config, q["image_path"], q["city_code"])
        invader_id = pred.get("prediction")
        ref_path = best_reference_image(invader_id) if invader_id else None
        crop = detector.crop(q["image_path"]) if detector else None
        results.append({
            "query": q,
            "pred": pred,
            "flash_b64": img_to_b64(q["image_path"]),
            "crop_b64": pil_to_b64(crop) if crop else None,
            "ref_b64": img_to_b64(ref_path) if ref_path else None,
            "ref_path": str(ref_path) if ref_path else None,
        })
    print(f"\nBuilding HTML...")

    cards = []
    for r in results:
        pred = r["pred"]
        invader_id = pred.get("prediction") or "—"
        score = pred["diagnostics"]["top_score"]
        margin = pred["diagnostics"]["margin_to_second"]
        confidence = pred.get("confidence_label", "unknown")
        used_crop = pred["diagnostics"].get("used_crop", False)
        city = r["query"].get("city_code", "")
        flash_name = Path(r["query"]["image_path"]).name

        top_k_html = "".join(
            f'<tr><td>{c["invader_id"]}</td><td>{c["score"]:.4f}</td></tr>'
            for c in pred.get("top_k", [])[:5]
        )

        flash_img = (
            f'<img src="{r["flash_b64"]}" alt="flash">'
            if r["flash_b64"]
            else '<div class="no-img">no image</div>'
        )
        crop_img = (
            f'<img src="{r["crop_b64"]}" alt="crop">'
            if r.get("crop_b64")
            else '<div class="no-img">no crop</div>'
        )
        ref_img = (
            f'<img src="{r["ref_b64"]}" alt="reference">'
            if r["ref_b64"]
            else '<div class="no-img">no ref</div>'
        )

        confidence_class = confidence.replace(" ", "-")

        cards.append(f"""
        <div class="card">
          <div class="card-header">
            <span class="city">{city}</span>
            <span class="filename">{flash_name}</span>
            <span class="confidence {confidence_class}">{confidence}</span>
          </div>
          <div class="images">
            <div class="img-block">
              <div class="label">Flash</div>
              {flash_img}
            </div>
            <div class="img-block">
              <div class="label">Crop used for matching</div>
              {crop_img}
            </div>
            <div class="img-block">
              <div class="label">Reference: {invader_id}</div>
              {ref_img}
            </div>
          </div>
          <div class="scores">
            <table>
              <tr><th>Invader</th><th>Score</th></tr>
              {top_k_html}
            </table>
            <div class="meta">score={score:.4f} margin={margin:.4f}</div>
          </div>
          <div class="verdict">
            <label><input type="radio" name="v_{flash_name}" value="correct"> ✓ Correct</label>
            <label><input type="radio" name="v_{flash_name}" value="wrong"> ✗ Wrong</label>
            <label><input type="radio" name="v_{flash_name}" value="unsure"> ? Unsure</label>
          </div>
        </div>""")

    cards_html = "\n".join(cards)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Flash Image Validation</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }}
  h1 {{ font-size: 1.2rem; margin-bottom: 16px; color: #aaa; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 16px; }}
  .card {{ background: #1e1e1e; border-radius: 8px; padding: 12px; border: 1px solid #333; }}
  .card-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 10px; font-size: 0.85rem; }}
  .city {{ background: #333; padding: 2px 6px; border-radius: 4px; font-weight: bold; }}
  .filename {{ color: #888; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .confidence {{ padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }}
  .certainly {{ background: #1a5c2a; color: #4caf50; }}
  .probably  {{ background: #1a3f5c; color: #42a5f5; }}
  .maybe     {{ background: #4a3a00; color: #ffc107; }}
  .unknown   {{ background: #3a1a1a; color: #ef5350; }}
  .images {{ display: flex; gap: 10px; margin-bottom: 10px; }}
  .img-block {{ flex: 1; }}
  .img-block img {{ width: 100%; border-radius: 4px; display: block; }}
  .label {{ font-size: 0.75rem; color: #888; margin-bottom: 4px; }}
  .no-img {{ height: 120px; background: #2a2a2a; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #555; font-size: 0.8rem; }}
  .scores table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-bottom: 4px; }}
  .scores th {{ text-align: left; color: #888; padding: 2px 4px; border-bottom: 1px solid #333; }}
  .scores td {{ padding: 2px 4px; }}
  .scores tr:first-child td {{ font-weight: bold; color: #fff; }}
  .meta {{ font-size: 0.75rem; color: #666; }}
  .verdict {{ display: flex; gap: 16px; margin-top: 10px; font-size: 0.85rem; }}
  .verdict label {{ cursor: pointer; display: flex; align-items: center; gap: 4px; }}
  #summary {{ position: fixed; bottom: 16px; right: 16px; background: #1e1e1e; border: 1px solid #444; border-radius: 8px; padding: 12px 16px; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>Flash image validation — {len(results)} predictions</h1>
<div class="grid">
{cards_html}
</div>
<div id="summary">
  <strong>Verdicts:</strong>
  <span id="correct-count">0</span> correct &nbsp;
  <span id="wrong-count">0</span> wrong &nbsp;
  <span id="unsure-count">0</span> unsure
</div>
<script>
  document.querySelectorAll('input[type=radio]').forEach(r => r.addEventListener('change', update));
  function update() {{
    document.getElementById('correct-count').textContent = document.querySelectorAll('input[value=correct]:checked').length;
    document.getElementById('wrong-count').textContent   = document.querySelectorAll('input[value=wrong]:checked').length;
    document.getElementById('unsure-count').textContent  = document.querySelectorAll('input[value=unsure]:checked').length;
  }}
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--city", default=None, help="Filter to a specific city code")
    parser.add_argument("--output", default="outputs/validation.html")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_yaml(args.config)
    queries = [json.loads(l) for l in open(config["paths"]["query_manifest"])]
    queries = [q for q in queries if q.get("city_code")]
    if args.city:
        queries = [q for q in queries if q["city_code"] == args.city]

    random.seed(args.seed)
    sample = random.sample(queries, min(args.n, len(queries)))

    build_page(config, sample, Path(args.output))
    import subprocess
    subprocess.Popen(["open", args.output])


if __name__ == "__main__":
    main()
