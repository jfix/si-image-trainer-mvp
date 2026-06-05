#!/usr/bin/env python3
"""Build a side-by-side HTML comparison page for two model configs.

Usage:
  python scripts/build_comparison_page.py \\
    --config-a configs/exp17.yaml \\
    --config-b configs/exp19.yaml \\
    --n 50 --city PA \\
    --output outputs/comparison_exp17_vs_exp19.html

The script will auto-build the index for config-b if not already present,
for only the cities being tested (much faster than a full rebuild).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from si_image_trainer.indexing.build_index import build_indexes
from si_image_trainer.inference.predict import predict_one
from si_image_trainer.models.detector import MosaicDetector
from si_image_trainer.utils.io import load_yaml


REFERENCE_ROOT = Path("/Users/jakob/Projects/si-reference-library/references")


def pil_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def img_to_b64(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    suffix = p.suffix.lower().lstrip(".")
    mime = "png" if suffix == "png" else "jpeg"
    return f"data:image/{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def best_reference_image(invader_id: str) -> Path | None:
    if not invader_id:
        return None
    meta_path = REFERENCE_ROOT / invader_id.split("_")[0] / invader_id / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    for role in ("grosplan", "reference"):
        for img in meta.get("images", []):
            if img.get("role") == role:
                candidate = REFERENCE_ROOT.parent / img["local_path"]
                if candidate.exists():
                    return candidate
    return None


def maybe_build_index(config: dict, label: str, cities: set[str]) -> None:
    """Build city indexes for config if they don't exist yet."""
    index_dir = Path(config["paths"]["index_dir"])
    missing = [c for c in sorted(cities) if not (index_dir / c / "embeddings.npy").exists()]
    if not missing:
        print(f"[{label}] Index already present for all test cities — skipping build.")
        return

    print(f"[{label}] Building index for cities: {missing}")
    print(f"[{label}] Index dir: {index_dir}")

    # Filter reference manifest to only the missing cities
    allowed_roles = set(
        config.get("retrieval", {}).get("stage1_allowed_roles", ["reference", "flash-reference", "grosplan"])
    )
    rows = [json.loads(l) for l in open(config["paths"]["reference_manifest"])]
    rows = [r for r in rows if r["city_code"] in missing and r.get("role", "reference") in allowed_roles]

    tmp = Path("outputs/tmp_build_index_manifest.jsonl")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    try:
        build_indexes(
            reference_manifest_path=tmp,
            output_dir=index_dir,
            embedding_config=config["embedding"],
            retrieval_config=config.get("retrieval"),
            detector_config=config.get("detector"),
        )
        print(f"[{label}] Index built successfully.")
    finally:
        tmp.unlink(missing_ok=True)


def run_predictions(config: dict, queries: list[dict], label: str) -> list[dict]:
    detector = MosaicDetector(**config["detector"]) if config.get("detector") else None
    results = []
    for i, q in enumerate(queries, 1):
        print(f"  [{label}] {i}/{len(queries)}  {Path(q['image_path']).name}", end="\r")
        try:
            pred = predict_one(config, q["image_path"], q["city_code"])
        except Exception as exc:
            pred = {"prediction": None, "confidence_label": "error", "diagnostics": {"top_score": 0, "margin_to_second": 0}, "top_k": [], "error": str(exc)}
        crop = detector.crop(q["image_path"]) if detector else None
        results.append({
            "query": q,
            "pred": pred,
            "crop_b64": pil_to_b64(crop) if crop else None,
        })
    print()
    return results


def pred_html(pred: dict, label: str) -> str:
    invader_id = pred.get("prediction") or "—"
    confidence = pred.get("confidence_label", "unknown")
    top_score = pred.get("diagnostics", {}).get("top_score", 0)
    margin = pred.get("diagnostics", {}).get("margin_to_second", 0)
    ref_path = best_reference_image(invader_id) if invader_id != "—" else None
    ref_b64 = img_to_b64(ref_path) if ref_path else None

    ref_img = (
        f'<img src="{ref_b64}" alt="ref">'
        if ref_b64
        else '<div class="no-img">no ref</div>'
    )

    top_k_html = "".join(
        f'<tr><td>{c["invader_id"]}</td><td>{c["score"]:.4f}</td></tr>'
        for c in pred.get("top_k", [])[:5]
    )

    confidence_class = confidence.replace(" ", "-")

    return f"""
      <div class="pred-col">
        <div class="pred-label">{label}</div>
        <span class="confidence {confidence_class}">{confidence}</span>
        <div class="ref-img">{ref_img}</div>
        <div class="pred-id">{invader_id}</div>
        <table class="scores-table">
          <tr><th>ID</th><th>Score</th></tr>
          {top_k_html}
        </table>
        <div class="meta">score={top_score:.4f} margin={margin:.4f}</div>
      </div>"""


def build_page(
    queries: list[dict],
    results_a: list[dict],
    results_b: list[dict],
    label_a: str,
    label_b: str,
    output_path: Path,
) -> None:
    cards = []
    for ra, rb in zip(results_a, results_b):
        q = ra["query"]
        city = q.get("city_code", "")
        flash_name = Path(q["image_path"]).name
        flash_b64 = img_to_b64(q["image_path"])

        flash_img = (
            f'<img src="{flash_b64}" alt="flash">'
            if flash_b64
            else '<div class="no-img">no image</div>'
        )
        crop_img = (
            f'<img src="{ra["crop_b64"]}" alt="crop">'
            if ra.get("crop_b64")
            else '<div class="no-img">no crop</div>'
        )

        pred_a = pred_html(ra["pred"], label_a)
        pred_b = pred_html(rb["pred"], label_b)

        uid = flash_name.replace(".", "_")

        same_pred = ra["pred"].get("prediction") == rb["pred"].get("prediction")
        agreement_badge = '<span class="badge-same">same prediction</span>' if same_pred else ""

        cards.append(f"""
        <div class="card">
          <div class="card-header">
            <span class="city">{city}</span>
            <span class="filename">{flash_name}</span>
            {agreement_badge}
          </div>
          <div class="card-body">
            {pred_a}
            <div class="center-col">
              <div class="img-label">Flash</div>
              {flash_img}
              <div class="img-label" style="margin-top:8px">Crop</div>
              {crop_img}
            </div>
            {pred_b}
          </div>
          <div class="verdict">
            <label><input type="radio" name="v_{uid}" value="a_better"> ✓ {label_a} better</label>
            <label><input type="radio" name="v_{uid}" value="b_better"> ✓ {label_b} better</label>
            <label><input type="radio" name="v_{uid}" value="same"> = Same</label>
            <label><input type="radio" name="v_{uid}" value="wrong"> ✗ Both wrong</label>
          </div>
        </div>""")

    cards_html = "\n".join(cards)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Model Comparison: {label_a} vs {label_b}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }}
  h1 {{ font-size: 1.1rem; margin-bottom: 4px; color: #aaa; }}
  .subtitle {{ font-size: 0.85rem; color: #666; margin-bottom: 16px; }}
  .grid {{ display: flex; flex-direction: column; gap: 16px; }}
  .card {{ background: #1e1e1e; border-radius: 8px; padding: 12px; border: 1px solid #333; }}
  .card-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 10px; font-size: 0.85rem; }}
  .city {{ background: #333; padding: 2px 6px; border-radius: 4px; font-weight: bold; }}
  .filename {{ color: #888; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .badge-same {{ background: #2a3a2a; color: #7cb87c; font-size: 0.75rem; padding: 2px 6px; border-radius: 4px; }}
  .card-body {{ display: grid; grid-template-columns: 1fr 180px 1fr; gap: 12px; align-items: start; }}
  .pred-col {{ display: flex; flex-direction: column; gap: 6px; }}
  .pred-label {{ font-size: 0.8rem; font-weight: bold; color: #aaa; text-transform: uppercase; letter-spacing: 0.05em; }}
  .center-col {{ display: flex; flex-direction: column; }}
  .center-col img {{ width: 100%; border-radius: 4px; display: block; }}
  .img-label {{ font-size: 0.7rem; color: #666; margin-bottom: 2px; }}
  .ref-img img {{ width: 100%; border-radius: 4px; display: block; max-height: 160px; object-fit: contain; background: #111; }}
  .pred-id {{ font-size: 0.9rem; font-weight: bold; }}
  .scores-table {{ width: 100%; border-collapse: collapse; font-size: 0.75rem; }}
  .scores-table th {{ text-align: left; color: #888; padding: 2px 4px; border-bottom: 1px solid #333; }}
  .scores-table td {{ padding: 2px 4px; }}
  .scores-table tr:nth-child(2) td {{ font-weight: bold; color: #fff; }}
  .meta {{ font-size: 0.7rem; color: #555; }}
  .confidence {{ padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; display: inline-block; }}
  .certainly {{ background: #1a5c2a; color: #4caf50; }}
  .probably  {{ background: #1a3f5c; color: #42a5f5; }}
  .maybe     {{ background: #4a3a00; color: #ffc107; }}
  .unknown   {{ background: #3a1a1a; color: #ef5350; }}
  .no-img {{ height: 100px; background: #2a2a2a; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #555; font-size: 0.8rem; }}
  .verdict {{ display: flex; gap: 16px; margin-top: 10px; font-size: 0.8rem; flex-wrap: wrap; }}
  .verdict label {{ cursor: pointer; display: flex; align-items: center; gap: 4px; }}
  #summary {{ position: fixed; bottom: 16px; right: 16px; background: #1e1e1e; border: 1px solid #444; border-radius: 8px; padding: 12px 16px; font-size: 0.85rem; min-width: 200px; }}
  #summary strong {{ display: block; margin-bottom: 6px; }}
  .sum-row {{ display: flex; justify-content: space-between; gap: 16px; }}
</style>
</head>
<body>
<h1>Model comparison: <strong>{label_a}</strong> vs <strong>{label_b}</strong></h1>
<p class="subtitle">{len(results_a)} flash images — radio buttons below each card to record verdict</p>
<div class="grid">
{cards_html}
</div>
<div id="summary">
  <strong>Verdicts ({len(results_a)} total)</strong>
  <div class="sum-row"><span>{label_a} better:</span> <span id="cnt-a">0</span></div>
  <div class="sum-row"><span>{label_b} better:</span> <span id="cnt-b">0</span></div>
  <div class="sum-row"><span>Same:</span>              <span id="cnt-same">0</span></div>
  <div class="sum-row"><span>Both wrong:</span>        <span id="cnt-wrong">0</span></div>
  <div class="sum-row"><span>Unreviewed:</span>        <span id="cnt-open">{len(results_a)}</span></div>
</div>
<script>
  function update() {{
    const a    = document.querySelectorAll('input[value=a_better]:checked').length;
    const b    = document.querySelectorAll('input[value=b_better]:checked').length;
    const same = document.querySelectorAll('input[value=same]:checked').length;
    const wrong= document.querySelectorAll('input[value=wrong]:checked').length;
    const total= document.querySelectorAll('input[type=radio]').length / 4;
    document.getElementById('cnt-a').textContent    = a;
    document.getElementById('cnt-b').textContent    = b;
    document.getElementById('cnt-same').textContent = same;
    document.getElementById('cnt-wrong').textContent= wrong;
    document.getElementById('cnt-open').textContent = total - a - b - same - wrong;
  }}
  document.querySelectorAll('input[type=radio]').forEach(r => r.addEventListener('change', update));
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-a", default="configs/exp17.yaml", help="Baseline config (already has index)")
    parser.add_argument("--config-b", default="configs/exp19.yaml", help="Challenger config (index built on demand)")
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--city", default="PA", help="City code to sample flash images from")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-index-build", action="store_true", help="Skip index build even if missing")
    args = parser.parse_args()

    config_a = load_yaml(args.config_a)
    config_b = load_yaml(args.config_b)

    label_a = args.label_a or Path(args.config_a).stem
    label_b = args.label_b or Path(args.config_b).stem

    output_path = Path(args.output) if args.output else Path(f"outputs/comparison_{label_a}_vs_{label_b}.html")

    queries_all = [json.loads(l) for l in open(config_a["paths"]["query_manifest"])]
    queries_all = [q for q in queries_all if q.get("city_code") == args.city]

    random.seed(args.seed)
    sample = random.sample(queries_all, min(args.n, len(queries_all)))

    test_cities = {q["city_code"] for q in sample}

    if not args.skip_index_build:
        maybe_build_index(config_b, label_b, test_cities)

    print(f"Running {label_a} predictions...")
    results_a = run_predictions(config_a, sample, label_a)

    print(f"Running {label_b} predictions...")
    results_b = run_predictions(config_b, sample, label_b)

    print("Building HTML...")
    build_page(sample, results_a, results_b, label_a, label_b, output_path)

    subprocess.Popen(["open", str(output_path)])


if __name__ == "__main__":
    main()
