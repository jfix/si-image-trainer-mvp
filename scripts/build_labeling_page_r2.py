#!/usr/bin/env python3
"""Build a labeling page using flash images downloaded from Cloudflare R2/D1."""
import argparse
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import urllib.request
import urllib.error

import boto3
from botocore.config import Config

ACCOUNT_ID   = "44fdf4e51d97c076209cb254bf9b728f"
BUCKET       = "si-images"
DB_ID        = "1603bece-c161-4619-b0b4-1f294f0ceb1d"
ENV_FILE     = Path("/Users/jakob/Projects/collect-si-worker/.env")
IMAGES_DIR   = Path("/Users/jakob/Projects/collect-si-live-data/images")
FLASH_LABELS = Path("data/processed/flash_labels.jsonl")


def load_env() -> dict:
    env = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def query_d1(sql: str, env: dict) -> list:
    url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/d1/database/{DB_ID}/query"
    body = json.dumps({"sql": sql}).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={
            "Authorization": f"Bearer {env['CLOUDFLARE_API_TOKEN']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    if not data.get("success"):
        print("D1 error:", data.get("errors"), file=sys.stderr)
        sys.exit(1)
    return data["result"][0]["results"]


def make_r2_client(env: dict):
    return boto3.client(
        "s3",
        endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=env["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    env = load_env()
    output = args.output or Path(f"outputs/labeling_{args.city}_r2_seed{args.seed}.html")

    # Load already-labeled filenames to exclude
    labeled_filenames = set()
    if FLASH_LABELS.exists():
        for line in FLASH_LABELS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("city_code") == args.city:
                    labeled_filenames.add(Path(r["image_path"]).name)
    print(f"Excluding {len(labeled_filenames)} already-labeled {args.city} images")

    # Query D1 — fetch more than needed to allow for filtering
    print(f"Querying D1 for {args.city} images...")
    sql = f"""
        SELECT flash_id, img
        FROM events
        WHERE place_id = '{args.city}'
          AND img IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {args.n * 5}
    """
    rows = query_d1(sql, env)
    print(f"Got {len(rows)} candidates from D1")

    rows = [r for r in rows if Path(r["img"]).name not in labeled_filenames]
    rows = rows[:args.n]
    print(f"Using {len(rows)} images after filtering")

    if not rows:
        print("No new images to label.")
        sys.exit(0)

    # Download missing images from R2
    r2 = make_r2_client(env)
    queries = []
    downloaded = 0
    missing = 0
    for i, row in enumerate(rows, 1):
        local_path = IMAGES_DIR / row["img"]
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                r2.download_file(BUCKET, row["img"], str(local_path))
                downloaded += 1
            except Exception:
                missing += 1
                continue
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} ({downloaded} downloaded, {missing} missing)", end="\r")
        queries.append({"image_path": str(local_path), "city_code": args.city})
    print(f"\nReady: {len(queries)} images ({downloaded} new, {missing} missing from R2)")

    # Load config and call build_page directly
    spec = importlib.util.spec_from_file_location(
        "build_labeling_page", Path(__file__).parent / "build_labeling_page.py"
    )
    blp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(blp)

    config = blp.load_yaml(args.config)
    output.parent.mkdir(parents=True, exist_ok=True)
    blp.build_page(config, queries, output)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
