from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

from si_image_trainer.utils.image import open_image
from si_image_trainer.utils.io import read_jsonl

# Aggressive augmentation applied to the anchor — simulates flash photo conditions:
# variable lighting, motion blur, perspective distortion, distance variation.
_FLASH_AUGMENT = A.Compose([
    A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.4, hue=0.1, p=0.9),
    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 3.0), p=0.6),
    A.Perspective(scale=(0.05, 0.15), p=0.5),
    A.Affine(rotate=(-20, 20), translate_percent=(-0.1, 0.1), scale=(0.75, 1.25), p=0.6),
    A.RandomBrightnessContrast(p=0.4),
    A.Sharpen(alpha=(0, 0.5), lightness=(0.5, 1.0), p=0.3),
    A.ToGray(p=0.05),
    A.ImageCompression(quality_range=(60, 95), p=0.3),
])

# Light augmentation applied to the positive — adds variety without heavy distortion.
_LIGHT_AUGMENT = A.Compose([
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])


class TripletDataset(Dataset):
    def __init__(self, invader_images: dict[str, list[str]], processor, triplets_per_invader: int = 4, augment: bool = False) -> None:
        self._by_id = {k: v for k, v in invader_images.items() if len(v) >= 2}
        self._ids = list(self._by_id.keys())
        self._processor = processor
        self._augment = augment
        self._length = len(self._ids) * triplets_per_invader

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_id = self._ids[idx % len(self._ids)]
        anchor_path, pos_path = random.sample(self._by_id[anchor_id], 2)
        neg_id = random.choice([i for i in self._ids if i != anchor_id])
        neg_path = random.choice(self._by_id[neg_id])

        def load(path: str, aug=None) -> torch.Tensor:
            img = open_image(path)
            if aug is not None:
                img = aug(image=np.array(img))["image"]
            return self._processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        if self._augment:
            return load(anchor_path, _FLASH_AUGMENT), load(pos_path, _LIGHT_AUGMENT), load(neg_path)
        return load(anchor_path), load(pos_path), load(neg_path)


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _split_by_invader(rows: list[dict], val_fraction: float = 0.15) -> tuple[dict, dict]:
    by_id: dict[str, list[str]] = {}
    for row in rows:
        by_id.setdefault(row["invader_id"], []).append(row["image_path"])
    trainable = [k for k, v in by_id.items() if len(v) >= 2]
    random.seed(42)
    random.shuffle(trainable)
    n_val = max(1, int(len(trainable) * val_fraction))
    val_ids = set(trainable[:n_val])
    train = {k: v for k, v in by_id.items() if k not in val_ids and len(v) >= 2}
    val = {k: v for k, v in by_id.items() if k in val_ids}
    return train, val


def _encode(model: AutoModel, pixel_values: torch.Tensor) -> torch.Tensor:
    out = model(pixel_values=pixel_values)
    cls = out.last_hidden_state[:, 0]
    return F.normalize(cls, dim=-1)


def train_metric(config: dict[str, Any]) -> dict[str, Any]:
    train_cfg = config.get("training", {})
    model_name = train_cfg.get("model_name", "facebook/dinov2-small")
    output_dir = Path(train_cfg.get("output_dir", "outputs/models/dinov2_finetuned"))
    epochs = int(train_cfg.get("epochs", 20))
    batch_size = int(train_cfg.get("batch_size", 16))
    lr = float(train_cfg.get("lr", 1e-5))
    margin = float(train_cfg.get("margin", 0.3))
    unfreeze_last_n = int(train_cfg.get("unfreeze_last_n_blocks", 2))

    device = _get_device()
    print(f"Device: {device}")

    rows = read_jsonl(config["paths"]["reference_manifest"])
    train_inv, val_inv = _split_by_invader(rows)
    print(f"Train invaders: {len(train_inv)}  Val invaders: {len(val_inv)}")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Freeze all, then unfreeze the last N transformer blocks + layernorm
    for p in model.parameters():
        p.requires_grad = False
    for block in model.encoder.layer[-unfreeze_last_n:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.layernorm.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    model.to(device)

    augment = bool(train_cfg.get("augment", True))
    print(f"Augmentation: {'on' if augment else 'off'}")
    train_ds = TripletDataset(train_inv, processor, augment=augment)
    val_ds = TripletDataset(val_inv, processor, triplets_per_invader=8, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    history: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for anchor, positive, negative in train_loader:
            pixel_values = torch.cat([anchor, positive, negative], dim=0).to(device)
            emb = _encode(model, pixel_values)
            b = anchor.shape[0]
            loss = F.triplet_margin_loss(emb[:b], emb[b:2*b], emb[2*b:], margin=margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                pixel_values = torch.cat([anchor, positive, negative], dim=0).to(device)
                emb = _encode(model, pixel_values)
                b = anchor.shape[0]
                val_loss += F.triplet_margin_loss(emb[:b], emb[b:2*b], emb[2*b:], margin=margin).item()
        val_loss /= len(val_loader)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  {elapsed:.0f}s")
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4)})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            print(f"  -> saved (best val loss so far)")

    return {"best_val_loss": round(best_val_loss, 4), "history": history, "output_dir": str(output_dir)}
