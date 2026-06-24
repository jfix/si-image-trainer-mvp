#!/usr/bin/env python
"""Small, local InfoNCE + hard-negative validation (the exp24 recipe).

Ports the Colab exp24 recipe — InfoNCE / NT-Xent with in-batch negatives,
cleaned verify-UI hard negatives, and a fine-tuned DINOv2-small backbone — to a
fast local MPS run on a PA+LDN subset. The point is to A/B whether hard
negatives actually help (R@1 on held-out flash images) before paying for a full
Colab/Modal run. Run it twice with the same --seed:

    .venv/bin/python scripts/local_infonce.py --hard-negatives 0 --tag A
    .venv/bin/python scripts/local_infonce.py --hard-negatives 1 --tag B

Crops are cached to tmp/crop_cache so the second run is fast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import zipfile
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

REPO = Path(__file__).resolve().parent.parent


# ── Args ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cities", nargs="+", default=["PA", "LDN"])
    p.add_argument("--per-city", nargs="+", type=int, default=[90, 60],
                   help="invaders to sample per city (aligned with --cities)")
    p.add_argument("--hard-negatives", type=int, choices=[0, 1], default=0)
    p.add_argument("--hardneg-start", type=int, default=3,
                   help="epoch to phase hard negatives in")
    p.add_argument("--hardneg-drop-cos", type=float, default=0.90,
                   help="drop a hard neg if its ref is this close to the positive")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--unfreeze-n", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--flash-oversample", type=int, default=3)
    p.add_argument("--max-flash-per-invader", type=int, default=8,
                   help="cap train flash per invader to bound runtime")
    p.add_argument("--val-flash-frac", type=float, default=0.20)
    p.add_argument("--eval-bank-max", type=int, default=1200,
                   help="distractor reference invaders in the retrieval bank "
                        "(train set is included; larger = harder, more discriminative)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-detector", action="store_true")
    p.add_argument("--mps-mem-fraction", type=float, default=0.0,
                   help="optionally cap MPS allocations to this fraction of recommended max "
                        "(0=off; too-low values OOM. Prefer lowering --batch to save memory)")
    p.add_argument("--flash-zip", default=str(REPO / "outputs/colab_flash_data.zip"))
    p.add_argument("--ref-manifest", default=str(REPO / "data/processed/reference_manifest.jsonl"))
    p.add_argument("--detector", default=str(REPO / "outputs/models/mosaic_detector_v4.pt"))
    p.add_argument("--base-model", default="facebook/dinov2-small")
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=str(REPO / "outputs/local"))
    return p.parse_args()


# ── Image / crop helpers ─────────────────────────────────────────────────────
def open_image(path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


FLASH_AUG = A.Compose([
    A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.4, hue=0.1, p=0.9),
    A.GaussianBlur(blur_limit=(3, 7), p=0.6),
    A.Perspective(scale=(0.05, 0.15), p=0.5),
    A.Affine(rotate=(-20, 20), translate_percent=(-0.1, 0.1), scale=(0.75, 1.25), p=0.6),
    A.RandomBrightnessContrast(p=0.4),
])
LIGHT_AUG = A.Compose([
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])


class CropCache:
    """Detector crops cached to disk by source-path hash (shared across A/B runs).

    Crops are loaded ON DEMAND (only the current batch lives in RAM) and bounded
    to `max_side` px so memory stays flat regardless of subset size.
    """

    def __init__(self, detector_path: str | None, cache_dir: Path, max_side: int = 384):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_side = max_side
        self.model = None
        if detector_path:
            from ultralytics import YOLO
            self.model = YOLO(detector_path)

    def _bound(self, img: Image.Image) -> Image.Image:
        if max(img.size) > self.max_side:
            img = img.copy()
            img.thumbnail((self.max_side, self.max_side))
        return img

    def get(self, path) -> Image.Image:
        path = str(path)
        key = hashlib.sha1(path.encode()).hexdigest()[:16]
        cached = self.cache_dir / f"{key}.png"
        if cached.exists():
            with Image.open(cached) as im:
                return self._bound(im.convert("RGB"))
        img = open_image(path)
        crop = img
        if self.model is not None:
            res = self.model(img, conf=0.25, verbose=False)
            boxes = res[0].boxes
            if boxes is not None and len(boxes) > 0:
                i = int(boxes.conf.argmax())
                x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i].tolist())
                if x2 > x1 and y2 > y1:
                    crop = img.crop((x1, y1, x2, y2))
        crop = self._bound(crop)
        crop.save(cached)
        return crop


# ── Model ────────────────────────────────────────────────────────────────────
def encode(backbone, pixel_values):
    feats = backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
    return F.normalize(feats, dim=-1)


def info_nce_loss(anchor, pos, ids, temperature, extra_neg=None):
    B = anchor.shape[0]
    logits = anchor @ pos.T
    id_idx = {v: i for i, v in enumerate(sorted(set(ids)))}
    id_t = torch.tensor([id_idx[x] for x in ids], device=anchor.device)
    same = id_t[:, None] == id_t[None, :]
    eye = torch.eye(B, dtype=torch.bool, device=anchor.device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))
    if extra_neg is not None and extra_neg.numel() > 0:
        logits = torch.cat([logits, anchor @ extra_neg.T], dim=1)
    logits = logits / temperature
    targets = torch.arange(B, device=anchor.device)
    return F.cross_entropy(logits, targets)


# ── Dataset ──────────────────────────────────────────────────────────────────
class PairDataset(Dataset):
    """anchor = aug view of an invader (flash crop or aug ref crop);
    positive = a different clean reference crop of the SAME invader."""

    def __init__(self, ref_by_id, flash_pairs, cache, processor, flash_oversample):
        self.ref_by_id = {k: v for k, v in ref_by_id.items() if len(v) >= 1}
        self.ids = list(self.ref_by_id.keys())
        self.flash_pairs = flash_pairs
        self.cache = cache
        self.processor = processor
        # ref items: invaders with >=2 refs can form ref-ref pairs
        self.ref_ids = [i for i in self.ids if len(self.ref_by_id[i]) >= 2]
        self.ref_len = len(self.ref_ids)
        self.flash_len = len(flash_pairs) * flash_oversample

    def __len__(self):
        return self.ref_len + self.flash_len

    def _proc(self, img):
        return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

    def __getitem__(self, idx):
        if idx < self.ref_len:
            inv = self.ref_ids[idx % self.ref_len]
            a_path, p_path = random.sample(self.ref_by_id[inv], 2)
            a = LIGHT_AUG(image=np.array(self.cache.get(a_path)))["image"]
            return self._proc(a), self._proc(self.cache.get(p_path)), inv
        flash_path, inv = self.flash_pairs[(idx - self.ref_len) % len(self.flash_pairs)]
        p_path = random.choice(self.ref_by_id[inv])
        a = LIGHT_AUG(image=np.array(self.cache.get(flash_path)))["image"]
        return self._proc(a), self._proc(self.cache.get(p_path)), inv


def collate(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            [b[2] for b in batch])


# ── Eval ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def embed_references(backbone, ref_by_id, cache, processor, device, batch=64):
    backbone.eval()
    items = [(iid, p) for iid, paths in ref_by_id.items() for p in paths]
    embs, keys = [], []
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        pv = processor(images=[cache.get(p) for _, p in chunk], return_tensors="pt")["pixel_values"].to(device)
        embs.append(encode(backbone, pv).cpu())
        keys.extend(iid for iid, _ in chunk)
    embs = torch.cat(embs)
    by = {}
    for k, e in zip(keys, embs):
        by.setdefault(k, []).append(e)
    ids = list(by.keys())
    mat = F.normalize(torch.stack([torch.stack(by[i]).mean(0) for i in ids]), dim=-1)
    return mat, ids


@torch.no_grad()
def flash_recall(backbone, flash_pairs, ref_mat, ref_ids, cache, processor, device, ks=(1, 5), batch=64):
    backbone.eval()
    if not flash_pairs:
        return {k: float("nan") for k in ks}
    q_emb, q_ids = [], []
    for s in range(0, len(flash_pairs), batch):
        chunk = flash_pairs[s:s + batch]
        pv = processor(images=[cache.get(p) for p, _ in chunk], return_tensors="pt")["pixel_values"].to(device)
        q_emb.append(encode(backbone, pv).cpu())
        q_ids.extend(iid for _, iid in chunk)
    q_emb = torch.cat(q_emb)
    sims = q_emb @ ref_mat.T
    topk = sims.topk(max(ks), dim=1).indices
    out = {}
    for k in ks:
        hits = sum(q_ids[q] in {ref_ids[j] for j in topk[q, :k].tolist()} for q in range(q_emb.shape[0]))
        out[k] = hits / q_emb.shape[0]
    return out


def build_hardneg_emb(hard_negs, ref_mat, ref_ids, pos_lookup, drop_cos, device):
    id_to_row = {iid: i for i, iid in enumerate(ref_ids)}
    keep = set()
    for flash_path, pos_id in pos_lookup.items():
        if pos_id not in id_to_row:
            continue
        pos_vec = ref_mat[id_to_row[pos_id]]
        for neg_id in hard_negs.get(flash_path, []):
            if neg_id not in id_to_row:
                continue
            if float(pos_vec @ ref_mat[id_to_row[neg_id]]) < drop_cos:
                keep.add(neg_id)
    rows = [id_to_row[i] for i in keep]
    if not rows:
        return None
    return ref_mat[rows].to(device)


# ── Data prep ────────────────────────────────────────────────────────────────
def prepare_data(args):
    random.seed(args.seed)
    cities = set(args.cities)

    # reference manifest → invader -> [abs ref image paths]
    ref_by_id_all: dict[str, list[str]] = {}
    inv_city: dict[str, str] = {}
    for line in Path(args.ref_manifest).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["city_code"] not in cities:
            continue
        ref_by_id_all.setdefault(r["invader_id"], []).append(r["image_path"])
        inv_city[r["invader_id"]] = r["city_code"]

    # flash labels (with hard negs) from the zip
    flash_dir = REPO / "tmp/flash_data"
    if not (flash_dir / "flash_labels.jsonl").exists():
        flash_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.flash_zip) as z:
            z.extractall(flash_dir)
    flash_root = flash_dir
    by_inv_flash: dict[str, list[str]] = {}
    hard_negs_raw: dict[str, list[str]] = {}
    for line in (flash_dir / "flash_labels.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["city_code"] not in cities:
            continue
        inv = r["invader_id"]
        fp = str(flash_root / "flash_images" / "flash" / Path(r["image_path"]).name)
        if not Path(fp).exists() or inv not in ref_by_id_all:
            continue
        by_inv_flash.setdefault(inv, []).append(fp)
        if r.get("rejected_candidates"):
            hard_negs_raw[fp] = r["rejected_candidates"]

    # select ~per-city invaders: RANDOM among those with >=2 flash (sorting by
    # most-flash biases toward popular, distinctive, easy invaders → saturated R@1).
    selected: list[str] = []
    for city, n in zip(args.cities, args.per_city):
        cands = [i for i in by_inv_flash if inv_city.get(i) == city and len(by_inv_flash[i]) >= 2]
        random.Random(args.seed).shuffle(cands)
        selected.extend(cands[:n])
    selected_set = set(selected)

    ref_by_id = {i: ref_by_id_all[i] for i in selected}
    # split flash into train/val per invader
    train_flash, val_flash = [], []
    for inv in selected:
        imgs = list(by_inv_flash[inv])
        random.shuffle(imgs)
        n_val = min(5, max(1, int(len(imgs) * args.val_flash_frac)))
        for p in imgs[:n_val]:
            val_flash.append((p, inv))
        for p in imgs[n_val:n_val + args.max_flash_per_invader]:
            train_flash.append((p, inv))
    # hard negs restricted to selected invaders (must be in the reference bank)
    hard_negs = {fp: [n for n in negs if n in selected_set]
                 for fp, negs in hard_negs_raw.items() if fp in dict(train_flash)}
    hard_negs = {k: v for k, v in hard_negs.items() if v}
    # "hard" val flash = those the verify UI flagged as confusable (had rejected
    # candidates). This is where hard negatives are expected to help.
    hard_val = {p for p, _ in val_flash if p in hard_negs_raw}

    # Eval bank: the 150 train invaders + extra reference invaders as distractors
    # (held-out flash must be retrieved among many candidates → discriminative R@1).
    bank_ids = list(selected)
    extras = sorted(i for i in ref_by_id_all if i not in selected_set)
    random.Random(args.seed).shuffle(extras)
    bank_ids += extras[: max(0, args.eval_bank_max - len(bank_ids))]
    bank_by_id = {i: ref_by_id_all[i] for i in bank_ids}

    stats = {
        "invaders": len(selected),
        "per_city": {c: sum(1 for i in selected if inv_city.get(i) == c) for c in args.cities},
        "train_flash": len(train_flash),
        "val_flash": len(val_flash),
        "val_flash_hard": len(hard_val),
        "eval_bank": len(bank_by_id),
        "flash_with_hardneg": len(hard_negs),
        "total_hardneg_pairs": sum(len(v) for v in hard_negs.values()),
    }
    return ref_by_id, bank_by_id, train_flash, val_flash, hard_val, hard_negs, stats


# ── Train ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    tag = args.tag or ("B_hardneg" if args.hard_negatives else "A_baseline")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device == "mps" and args.mps_mem_fraction > 0:
        try:
            torch.mps.set_per_process_memory_fraction(args.mps_mem_fraction)
        except Exception as e:
            print(f"(could not set MPS memory fraction: {e})")

    print(f"=== local InfoNCE — tag={tag} hard_negatives={bool(args.hard_negatives)} device={device} ===")
    ref_by_id, bank_by_id, train_flash, val_flash, hard_val, hard_negs, stats = prepare_data(args)
    print("data:", json.dumps(stats))
    hard_pairs = [(p, i) for p, i in val_flash if p in hard_val]

    # pre-warm the on-disk crop cache (run YOLO once up front; crops are then
    # loaded on demand per batch so RAM stays flat regardless of subset size)
    cache = CropCache(None if args.no_detector else args.detector, REPO / "tmp/crop_cache")
    all_paths = {p for paths in bank_by_id.values() for p in paths}
    all_paths |= {p for p, _ in train_flash} | {p for p, _ in val_flash}
    print(f"warming crop cache for {len(all_paths)} images…")
    t0 = time.time()
    for i, p in enumerate(sorted(all_paths), 1):
        cache.get(p)  # crops + saves to disk if missing; result discarded
        if i % 300 == 0:
            print(f"  {i}/{len(all_paths)}  ({time.time()-t0:.0f}s)")
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
    n_train = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"trainable params: {n_train:,}")

    ds = PairDataset(ref_by_id, train_flash, cache, processor, args.flash_oversample)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                        collate_fn=collate, drop_last=True)
    print(f"train items/epoch: {len(ds)} ({ds.ref_len} ref + {ds.flash_len} flash) · {len(loader)} steps")

    use_hn = bool(args.hard_negatives)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        backbone.train()
        extra_neg = None
        hn_on = use_hn and epoch >= args.hardneg_start and hard_negs
        if hn_on:
            ref_mat, ref_ids = embed_references(backbone, ref_by_id, cache, processor, device)
            pos_lookup = {p: i for p, i in train_flash}
            extra_neg = build_hardneg_emb(hard_negs, ref_mat, ref_ids, pos_lookup, args.hardneg_drop_cos, device)
            backbone.train()
        loss_sum = 0.0
        for anchor, pos, ids in loader:
            anchor, pos = anchor.to(device), pos.to(device)
            a, p = encode(backbone, anchor), encode(backbone, pos)
            loss = info_nce_loss(a, p, ids, args.temperature, extra_neg)
            optim.zero_grad(); loss.backward(); optim.step()
            loss_sum += loss.item()
        bank_mat, bank_ids = embed_references(backbone, bank_by_id, cache, processor, device)
        rec = flash_recall(backbone, val_flash, bank_mat, bank_ids, cache, processor, device)
        rec_h = flash_recall(backbone, hard_pairs, bank_mat, bank_ids, cache, processor, device)
        if device == "mps":
            torch.mps.empty_cache()
        row = {"epoch": epoch, "loss": loss_sum / len(loader),
               "R@1": rec[1], "R@5": rec[5], "hardR@1": rec_h[1], "hardR@5": rec_h[5],
               "hardneg": hn_on, "secs": round(time.time() - t0)}
        history.append(row)
        best = max(best, rec_h[1])  # track the discriminative (hard-subset) metric
        print(f"  ep{epoch:2d}/{args.epochs}  loss={row['loss']:.4f}  "
              f"R@1={rec[1]:.3f}  hardR@1={rec_h[1]:.3f} (n={len(hard_pairs)})  "
              f"{row['secs']}s{'  +hardneg' if hn_on else ''}")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(
        {"tag": tag, "args": vars(args), "stats": stats, "best_R@1": best, "history": history}, indent=2))
    print(f"\nDONE tag={tag}  best R@1={best:.3f}  → {out_dir / (tag + '.json')}")


if __name__ == "__main__":
    main()
