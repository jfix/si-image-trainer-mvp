#!/usr/bin/env python3
"""Generate a searchable contact-sheet HTML page for a city's reference mosaics."""
import argparse
import json
import re
from pathlib import Path

MANIFEST = Path("data/processed/reference_manifest.jsonl")
ROLE_PRIORITY = ["grosplan", "reference", "flash-reference"]


def numeric_key(invader_id: str) -> int:
    m = re.search(r"(\d+)$", invader_id)
    return int(m.group(1)) if m else 0


def best_image(images):
    for role in ROLE_PRIORITY:
        for img in images:
            if img["role"] == role:
                return img
    return images[0] if images else None


def build_page(city: str, output: Path) -> None:
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    city_rows = [r for r in rows if r["city_code"] == city]
    if not city_rows:
        raise SystemExit(f"No rows found for city '{city}'")

    by_id: dict[str, list] = {}
    for r in city_rows:
        by_id.setdefault(r["invader_id"], []).append(r)

    invaders = sorted(by_id.keys(), key=numeric_key)
    print(f"{city}: {len(invaders)} invaders")

    cards = []
    missing = 0
    for inv_id in invaders:
        img = best_image(by_id[inv_id])
        if img is None:
            missing += 1
            continue
        p = Path(img["image_path"])
        if not p.exists():
            missing += 1
            continue
        file_url = f"file://{p.resolve()}"
        cards.append((inv_id, file_url))

    if missing:
        print(f"  ({missing} invaders skipped — image not found)")

    cards_js = json.dumps([{"id": c[0], "src": c[1]} for c in cards])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{city} contact sheet</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: sans-serif; background: #111; color: #eee; }}
  #toolbar {{
    position: sticky; top: 0; z-index: 10;
    background: #222; padding: 10px 16px;
    display: flex; align-items: center; gap: 12px;
    border-bottom: 1px solid #444;
  }}
  #toolbar h1 {{ font-size: 1rem; white-space: nowrap; }}
  #search {{
    flex: 1; padding: 6px 10px; border-radius: 6px;
    border: 1px solid #555; background: #333; color: #eee;
    font-size: 0.9rem;
  }}
  #count {{ font-size: 0.85rem; color: #aaa; white-space: nowrap; }}
  #grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
    gap: 6px;
    padding: 10px;
  }}
  .card {{
    background: #1e1e1e;
    border-radius: 6px;
    overflow: hidden;
    cursor: pointer;
    border: 2px solid transparent;
    transition: border-color 0.15s;
  }}
  .card:hover {{ border-color: #888; }}
  .card img {{
    width: 100%; aspect-ratio: 1;
    object-fit: cover; display: block;
    background: #2a2a2a;
  }}
  .card .label {{
    text-align: center; font-size: 0.7rem;
    padding: 3px 2px; color: #ccc;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .card.hidden {{ display: none; }}
  #copy-toast {{
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: #2e7d32; color: #fff; padding: 7px 18px; border-radius: 20px;
    font-size: 0.85rem; opacity: 0; pointer-events: none;
    transition: opacity 0.15s;
  }}
  #copy-toast.show {{ opacity: 1; }}
</style>
</head>
<body>

<div id="toolbar">
  <h1>{city} — {len(cards)} mosaics</h1>
  <input id="search" type="text" placeholder="Search by ID e.g. {city}_42 …" autofocus>
  <span id="count"></span>
</div>

<div id="grid"></div>

<div id="copy-toast"></div>

<script>
const CARDS = {cards_js};

const grid = document.getElementById('grid');
const search = document.getElementById('search');
const countEl = document.getElementById('count');

function render(filter) {{
  const q = filter.trim().toLowerCase();
  let shown = 0;
  grid.querySelectorAll('.card').forEach(el => {{
    const match = !q || el.dataset.id.toLowerCase().includes(q);
    el.classList.toggle('hidden', !match);
    if (match) shown++;
  }});
  countEl.textContent = q ? `${{shown}} / ${{CARDS.length}}` : '';
}}

CARDS.forEach(c => {{
  const div = document.createElement('div');
  div.className = 'card';
  div.dataset.id = c.id;
  div.innerHTML = `<img src="${{c.src}}" loading="lazy" title="${{c.id}}">
                   <div class="label">${{c.id}}</div>`;
  div.addEventListener('click', () => copyId(c.id));
  grid.appendChild(div);
}});

search.addEventListener('input', () => render(search.value));

let toastTimer;
function copyId(id) {{
  navigator.clipboard.writeText(id).then(() => {{
    const toast = document.getElementById('copy-toast');
    toast.textContent = `Copied ${{id}}`;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 1500);
  }});
}}
</script>
</body>
</html>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Saved to {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    output = args.output or Path(f"outputs/contact_sheet_{args.city}.html")
    build_page(args.city, output)


if __name__ == "__main__":
    main()
