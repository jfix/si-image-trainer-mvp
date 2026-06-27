#!/usr/bin/env python
"""Rank-aware hard-negative experiment (the "it's the second one" signal).

Baseline = InfoNCE flash→reference with in-batch negatives (same as
local_infonce.py). Treatment (--rank-hardneg 1) adds a pairwise margin term on
the verify-UI rank errors: for each flash whose correct answer was NOT the
model's rank-1, push sim(flash, correct) above sim(flash, confuser) for the
specific candidate(s) that outranked it. Reads data/processed/flash_ranked.jsonl
(built by export_ranked_labels.py).

Eval is R@1 on HELD-OUT error cases (rank>=2) against a large distractor bank —
a naturally-hard, sensitive metric (unlike the saturated random-flash R@1).

    .venv/bin/python scripts/local_rank.py --rank-hardneg 0 --tag A
    .venv/bin/python scripts/local_rank.py --rank-hardneg 1 --tag B
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_infonce import CropCache, encode, embed_references, flash_recall, info_nce_loss, LIGHT_AUG  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cities", nargs="+", default=["PA", "LDN"])
    p.add_argument("--rank-hardneg", type=int, choices=[0, 1], default=0)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--rank-lambda", type=float, default=1.0, help="weight of the pairwise term")
    p.add_argument("--rank-start", type=int, default=2, help="epoch to phase the rank term in")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--unfreeze-n", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--val-err-frac", type=float, default=0.25, help="fraction of error cases held out for eval")
    p.add_argument("--max-train-rows", type=int, default=2200,
                   help="cap training rows (keeps all error cases, samples non-error to fill) for speed")
    p.add_argument("--eval-bank-max", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-detector", action="store_true")
    p.add_argument("--ranked", default=str(REPO / "data/processed/flash_ranked.jsonl"))
    p.add_argument("--flash-zip", default=str(REPO / "outputs/colab_flash_data.zip"))
    p.add_argument("--ref-manifest", default=str(REPO / "data/processed/reference_manifest.jsonl"))
    p.add_argument("--detector", default=str(REPO / "outputs/models/mosaic_detector_v4.pt"))
    p.add_argument("--base-model", default="facebook/dinov2-small")
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=str(REPO / "outputs/local"))
    return p.parse_args()


class RankDataset(Dataset):
    """anchor = aug flash crop; positive = correct reference crop;
    confuser = reference crop of the candidate that outranked the truth (or None)."""

    def __init__(self, rows, ref_by_id, cache, processor):
        self.rows = rows
        self.ref_by_id = ref_by_id
        self.cache = cache
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def _proc(self, img):
        return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

    def __getitem__(self, idx):
        r = self.rows[idx]
        anchor = LIGHT_AUG(image=np.array(self.cache.get(r["flash_path"])))["image"]
        pos = self.cache.get(random.choice(self.ref_by_id[r["correct"]]))
        conf = None
        if r["confuser"] is not None:
            conf = self._proc(self.cache.get(random.choice(self.ref_by_id[r["confuser"]])))
        return self._proc(anchor), self._proc(pos), conf, r["correct"]


def collate_rank(batch):
    anchors = torch.stack([b[0] for b in batch])
    positives = torch.stack([b[1] for b in batch])
    ids = [b[3] for b in batch]
    conf_rows = [i for i, b in enumerate(batch) if b[2] is not None]
    confs = torch.stack([batch[i][2] for i in conf_rows]) if conf_rows else torch.empty(0)
    return anchors, positives, ids, confs, conf_rows


def prepare(args):
    random.seed(args.seed)
    cities = set(args.cities)

    ref_all: dict[str, list[str]] = {}
    for line in Path(args.ref_manifest).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["city_code"] in cities:
                ref_all.setdefault(r["invader_id"], []).append(r["image_path"])

    flash_dir = REPO / "tmp/flash_data"
    if not (flash_dir / "flash_labels.jsonl").exists():
        flash_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.flash_zip) as z:
            z.extractall(flash_dir)

    rows = []
    for line in Path(args.ranked).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["city"] not in cities or r["correct"] not in ref_all:
            continue
        fp = str(flash_dir / "flash_images" / r["image_path"])
        if not Path(fp).exists():
            continue
        confuser = next((c for c in r["outranking_confusers"] if c in ref_all), None)
        rows.append({"flash_path": fp, "correct": r["correct"],
                     "rank": r["correct_rank"], "confuser": confuser})

    # split: hold out error cases (rank>=2) for the sensitive eval
    err = [r for r in rows if r["rank"] >= 2]
    random.shuffle(err)
    n_val = int(len(err) * args.val_err_frac)
    val_err = err[:n_val]
    val_paths = {r["flash_path"] for r in val_err}
    train_rows = [r for r in rows if r["flash_path"] not in val_paths]
    # cap for speed: keep ALL training error cases (the signal), sample non-error to fill
    if len(train_rows) > args.max_train_rows:
        t_err = [r for r in train_rows if r["rank"] >= 2]
        t_non = [r for r in train_rows if r["rank"] < 2]
        random.shuffle(t_non)
        train_rows = t_err + t_non[: max(0, args.max_train_rows - len(t_err))]
        random.shuffle(train_rows)
    val_pairs = [(r["flash_path"], r["correct"]) for r in val_err]

    # eval bank: correct invaders in play + distractors
    train_inv = {r["correct"] for r in rows}
    bank_ids = list(train_inv)
    extras = sorted(i for i in ref_all if i not in train_inv)
    random.Random(args.seed).shuffle(extras)
    bank_ids += extras[: max(0, args.eval_bank_max - len(bank_ids))]
    bank_by_id = {i: ref_all[i] for i in bank_ids}
    ref_by_id = {i: ref_all[i] for i in train_inv | {r["confuser"] for r in rows if r["confuser"]}}

    stats = {"rows": len(rows), "train_rows": len(train_rows),
             "train_err": sum(1 for r in train_rows if r["rank"] >= 2),
             "train_with_confuser": sum(1 for r in train_rows if r["confuser"]),
             "val_err": len(val_pairs), "eval_bank": len(bank_by_id)}
    return ref_by_id, bank_by_id, train_rows, val_pairs, stats


def main():
    args = parse_args()
    tag = args.tag or ("B_rank" if args.rank_hardneg else "A_base")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"=== local rank-aware — tag={tag} rank_hardneg={bool(args.rank_hardneg)} device={device} ===")

    ref_by_id, bank_by_id, train_rows, val_pairs, stats = prepare(args)
    print("data:", json.dumps(stats))

    cache = CropCache(None if args.no_detector else args.detector, REPO / "tmp/crop_cache")
    paths = {p for ps in bank_by_id.values() for p in ps}
    paths |= {r["flash_path"] for r in train_rows} | {p for p, _ in val_pairs}
    print(f"warming crop cache for {len(paths)} images…")
    t0 = time.time()
    for i, p in enumerate(sorted(paths), 1):
        cache.get(p)
        if i % 400 == 0:
            print(f"  {i}/{len(paths)}  ({time.time()-t0:.0f}s)")
    print(f"crop cache ready ({time.time()-t0:.0f}s)")

    processor = AutoImageProcessor.from_pretrained(args.base_model)
    backbone = AutoModel.from_pretrained(args.base_model).to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    for blk in backbone.encoder.layer[-args.unfreeze_n:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in backbone.layernorm.parameters():
        p.requires_grad = True
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, backbone.parameters()), lr=args.lr)

    ds = RankDataset(train_rows, ref_by_id, cache, processor)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                        collate_fn=collate_rank, drop_last=True)
    print(f"train rows: {len(ds)} · {len(loader)} steps · val error cases: {len(val_pairs)}")

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        backbone.train()
        rank_on = bool(args.rank_hardneg) and epoch >= args.rank_start
        loss_sum = margin_sum = 0.0
        for anchors, positives, ids, confs, conf_rows in loader:
            anchors, positives = anchors.to(device), positives.to(device)
            a, p = encode(backbone, anchors), encode(backbone, positives)
            loss = info_nce_loss(a, p, ids, args.temperature)
            if rank_on and len(conf_rows) > 0:
                c = encode(backbone, confs.to(device))
                idx = torch.tensor(conf_rows, device=device)
                ap = (a[idx] * p[idx]).sum(-1)
                ac = (a[idx] * c).sum(-1)
                margin = F.relu(args.margin - ap + ac).mean()
                loss = loss + args.rank_lambda * margin
                margin_sum += float(margin)
            optim.zero_grad(); loss.backward(); optim.step()
            loss_sum += float(loss)
        bank_mat, bank_ids = embed_references(backbone, bank_by_id, cache, processor, device)
        rec = flash_recall(backbone, val_pairs, bank_mat, bank_ids, cache, processor, device)
        if device == "mps":
            torch.mps.empty_cache()
        best = max(best, rec[1])
        row = {"epoch": epoch, "loss": loss_sum / len(loader),
               "margin": margin_sum / max(1, len(loader)), "R@1": rec[1], "R@5": rec[5],
               "rank_on": rank_on, "secs": round(time.time() - t0)}
        history.append(row)
        print(f"  ep{epoch:2d}/{args.epochs}  loss={row['loss']:.4f}  "
              f"errR@1={rec[1]:.3f} errR@5={rec[5]:.3f}  {row['secs']}s{'  +rank' if rank_on else ''}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(
        {"tag": tag, "args": vars(args), "stats": stats, "best_errR@1": best, "history": history}, indent=2))
    print(f"\nDONE tag={tag}  best error-case R@1={best:.3f}  → {out / (tag + '.json')}")


if __name__ == "__main__":
    main()
