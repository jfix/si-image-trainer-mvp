#!/usr/bin/env python3
"""Build a self-contained HTML labeling page for creating a flash image ground-truth dataset."""
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
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def img_to_b64(path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    suffix = p.suffix.lower().lstrip(".")
    mime = "png" if suffix == "png" else "jpeg"
    return f"data:image/{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def best_reference_image(invader_id: str) -> Path | None:
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


def build_page(config: dict, queries: list[dict], output_path: Path) -> None:
    detector = MosaicDetector(**config["detector"]) if config.get("detector") else None
    print(f"Running predictions on {len(queries)} images...")
    results = []
    for i, q in enumerate(queries, 1):
        print(f"  {i}/{len(queries)}  {Path(q['image_path']).name}", end="\r")
        pred = predict_one(config, q["image_path"], q["city_code"])
        crop = detector.crop(q["image_path"]) if detector else None
        # Skip images where detector found nothing — harder to label without a clean crop
        if crop is None:
            continue
        top_k = pred.get("top_k", [])[:5]
        invader_id = pred.get("prediction") or ""
        ref_path = best_reference_image(invader_id) if invader_id else None
        results.append({
            "query": q,
            "pred": pred,
            "top_k": top_k,
            "flash_b64": img_to_b64(q["image_path"]),
            "crop_b64": pil_to_b64(crop),
            "ref_b64": img_to_b64(ref_path) if ref_path else None,
        })
    print(f"\nKept {len(results)} images with detected crops. Building HTML...")

    cards = []
    for r in results:
        flash_name = Path(r["query"]["image_path"]).name
        image_path = r["query"]["image_path"]
        city_code  = r["query"]["city_code"]
        prediction = r["pred"].get("prediction") or ""
        score      = r["pred"]["diagnostics"]["top_score"]

        quick_btns = "".join(
            f'<button class="quick-btn" onclick="setLabel(\'{flash_name}\', \'{c["invader_id"]}\')">'
            f'{c["invader_id"]} <span class="btn-score">{c["score"]:.3f}</span></button>'
            for c in r["top_k"]
        )
        ref_img = (
            f'<img src="{r["ref_b64"]}" alt="reference" id="ref_{flash_name}">'
            if r["ref_b64"]
            else f'<div class="no-img" id="ref_{flash_name}">no ref</div>'
        )

        # Encode top_k as JSON for JS lookup
        top_k_json = json.dumps({c["invader_id"]: c["score"] for c in r["top_k"]})

        cards.append(f"""
        <div class="card" id="card_{flash_name}">
          <div class="card-header">
            <span class="city">{city_code}</span>
            <span class="filename">{flash_name}</span>
            <span class="model-pred">model: {prediction} ({score:.3f})</span>
          </div>
          <div class="images">
            <div class="img-block">
              <div class="label">Flash</div>
              <img src="{r['flash_b64']}" alt="flash">
            </div>
            <div class="img-block">
              <div class="label">Crop</div>
              <img src="{r['crop_b64']}" alt="crop">
            </div>
            <div class="img-block">
              <div class="label" id="reflabel_{flash_name}">Reference: {prediction}</div>
              {ref_img}
            </div>
          </div>
          <div class="label-row">
            <div class="quick-btns">{quick_btns}</div>
            <div class="input-row">
              <input type="text" class="id-input" id="input_{flash_name}"
                     placeholder="e.g. PA_392" value="{prediction}"
                     oninput="onInput('{flash_name}', this.value)"
                     data-image-path="{image_path}"
                     data-city="{city_code}"
                     data-topk='{top_k_json}'>
              <button class="skip-btn" onclick="skipCard('{flash_name}')">skip</button>
            </div>
          </div>
        </div>""")

    cards_html = "\n".join(cards)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Flash Image Labeling</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }}
  h1 {{ font-size: 1.2rem; margin-bottom: 4px; color: #aaa; }}
  .subtitle {{ font-size: 0.85rem; color: #666; margin-bottom: 16px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); gap: 16px; }}
  .card {{ background: #1e1e1e; border-radius: 8px; padding: 12px; border: 1px solid #333; transition: border-color 0.2s; }}
  .card.labeled {{ border-color: #2e7d32; }}
  .card.skipped {{ opacity: 0.4; }}
  .card-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 10px; font-size: 0.85rem; }}
  .city {{ background: #333; padding: 2px 6px; border-radius: 4px; font-weight: bold; }}
  .filename {{ color: #888; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .model-pred {{ font-size: 0.75rem; color: #666; }}
  .images {{ display: flex; gap: 8px; margin-bottom: 10px; }}
  .img-block {{ flex: 1; }}
  .img-block img {{ width: 100%; border-radius: 4px; display: block; }}
  .label {{ font-size: 0.75rem; color: #888; margin-bottom: 4px; }}
  .no-img {{ height: 100px; background: #2a2a2a; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #555; font-size: 0.8rem; }}
  .label-row {{ display: flex; flex-direction: column; gap: 8px; }}
  .quick-btns {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .quick-btn {{ background: #2a2a2a; border: 1px solid #444; color: #ccc; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.8rem; }}
  .quick-btn:hover {{ background: #333; border-color: #666; }}
  .quick-btn.active {{ background: #1a3a1a; border-color: #4caf50; color: #4caf50; }}
  .btn-score {{ color: #666; font-size: 0.75rem; }}
  .input-row {{ display: flex; gap: 8px; align-items: center; }}
  .id-input {{ background: #2a2a2a; border: 1px solid #444; color: #fff; padding: 6px 10px; border-radius: 4px; font-size: 0.9rem; width: 140px; }}
  .id-input:focus {{ outline: none; border-color: #4caf50; }}
  .id-input.confirmed {{ border-color: #4caf50; background: #1a3a1a; }}
  .skip-btn {{ background: transparent; border: 1px solid #444; color: #666; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 0.8rem; }}
  .skip-btn:hover {{ border-color: #888; color: #aaa; }}
  #toolbar {{ position: fixed; bottom: 16px; right: 16px; background: #1e1e1e; border: 1px solid #444; border-radius: 8px; padding: 12px 16px; display: flex; align-items: center; gap: 16px; }}
  #counter {{ font-size: 0.9rem; color: #aaa; }}
  #counter span {{ color: #4caf50; font-weight: bold; }}
  #download-btn {{ background: #2e7d32; border: none; color: #fff; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: bold; }}
  #download-btn:hover {{ background: #388e3c; }}
  #download-btn:disabled {{ background: #333; color: #666; cursor: default; }}
</style>
</head>
<body>
<h1>Flash image labeling — {len(results)} images</h1>
<p class="subtitle">For each image, confirm or correct the invader ID. Use the quick-select buttons or type directly. Skip anything uncertain.</p>
<div class="grid">
{cards_html}
</div>
<div id="toolbar">
  <div id="counter"><span id="label-count">0</span> labeled</div>
  <button id="download-btn" onclick="downloadLabels()" disabled>Download labels.jsonl</button>
</div>
<script>
const labels = {{}};
const refData = {{}};

// Pre-populate with current predictions so they show as labeled
document.querySelectorAll('.id-input').forEach(inp => {{
  const name = inp.id.replace('input_', '');
  if (inp.value) {{
    labels[name] = {{
      image_path: inp.dataset.imagePath,
      city_code: inp.dataset.city,
      invader_id: inp.value,
      source: 'model'
    }};
  }}
}});

function updateCounter() {{
  const confirmed = Object.values(labels).filter(l => l.source === 'user').length;
  document.getElementById('label-count').textContent = confirmed;
  document.getElementById('download-btn').disabled = confirmed === 0;
}}

function setLabel(filename, invaderId) {{
  const inp = document.getElementById('input_' + filename);
  inp.value = invaderId;
  inp.classList.add('confirmed');
  document.querySelectorAll('#card_' + filename + ' .quick-btn').forEach(b => b.classList.remove('active'));
  const activeBtn = Array.from(document.querySelectorAll('#card_' + filename + ' .quick-btn'))
    .find(b => b.textContent.startsWith(invaderId));
  if (activeBtn) activeBtn.classList.add('active');
  labels[filename] = {{
    image_path: inp.dataset.imagePath,
    city_code: inp.dataset.city,
    invader_id: invaderId,
    source: 'user'
  }};
  document.getElementById('card_' + filename).classList.add('labeled');
  updateCounter();
}}

function onInput(filename, value) {{
  const inp = document.getElementById('input_' + filename);
  const cleaned = value.trim().toUpperCase();
  if (cleaned.length >= 4) {{
    labels[filename] = {{
      image_path: inp.dataset.imagePath,
      city_code: inp.dataset.city,
      invader_id: cleaned,
      source: 'user'
    }};
    inp.classList.add('confirmed');
    document.getElementById('card_' + filename).classList.add('labeled');
  }} else {{
    delete labels[filename];
    inp.classList.remove('confirmed');
    document.getElementById('card_' + filename).classList.remove('labeled');
  }}
  updateCounter();
}}

function skipCard(filename) {{
  delete labels[filename];
  document.getElementById('input_' + filename).value = '';
  document.getElementById('card_' + filename).classList.add('skipped');
  document.getElementById('card_' + filename).classList.remove('labeled');
  updateCounter();
}}

function downloadLabels() {{
  const userLabels = Object.values(labels).filter(l => l.source === 'user');
  const lines = userLabels.map(l => JSON.stringify({{
    image_path: l.image_path,
    city_code: l.city_code,
    invader_id: l.invader_id
  }}));
  const blob = new Blob([lines.join('\\n') + '\\n'], {{type: 'application/x-ndjson'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'flash_labels.jsonl';
  a.click();
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
    parser.add_argument("--city", default="PA")
    parser.add_argument("--output", default="outputs/labeling.html")
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
