from __future__ import annotations

import numpy as np

from si_image_trainer.utils.image import normalize_vector, open_image, resize_gray, resize_rgb


class BaselineEmbedder:
    def __init__(self, image_size: int = 96, color_bins: int = 12, edge_bins: int = 16, gradient_size: int = 24) -> None:
        self.image_size = image_size
        self.color_bins = color_bins
        self.edge_bins = edge_bins
        self.gradient_size = gradient_size

    def embed_path(self, image_path: str) -> np.ndarray:
        return self.embed_image(open_image(image_path))

    def embed_image(self, image) -> np.ndarray:
        rgb = resize_rgb(image, self.image_size)
        gray = resize_gray(image, self.gradient_size)

        color_features: list[np.ndarray] = []
        for channel in range(3):
            hist, _ = np.histogram(rgb[:, :, channel], bins=self.color_bins, range=(0.0, 1.0), density=True)
            color_features.append(hist.astype(np.float32))

        gx = np.diff(gray, axis=1, append=gray[:, -1:])
        gy = np.diff(gray, axis=0, append=gray[-1:, :])
        magnitude = np.sqrt(gx * gx + gy * gy)
        orientation = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
        edge_hist, _ = np.histogram(
            orientation[magnitude > magnitude.mean()],
            bins=self.edge_bins,
            range=(0.0, 1.0),
            density=True,
        )
        if np.isnan(edge_hist).any():
            edge_hist = np.zeros(self.edge_bins, dtype=np.float32)

        pooled = gray.reshape(6, self.gradient_size // 6, 6, self.gradient_size // 6).mean(axis=(1, 3)).flatten()

        features = np.concatenate([*color_features, edge_hist.astype(np.float32), pooled.astype(np.float32)])
        return normalize_vector(features.astype(np.float32))


class PretrainedEmbedder:
    def __init__(self, model_name: str = "facebook/dinov2-small") -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._model.to(self._device)
        self._torch = torch

    def embed_path(self, image_path: str) -> np.ndarray:
        return self.embed_image(open_image(image_path))

    def embed_image(self, image) -> np.ndarray:
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0].squeeze()
        return normalize_vector(cls_token.cpu().numpy().astype(np.float32))


def make_embedder(config: dict) -> BaselineEmbedder | PretrainedEmbedder:
    if config.get("type") == "pretrained":
        return PretrainedEmbedder(model_name=config.get("model_name", "dinov2_vits14"))
    return BaselineEmbedder(
        image_size=config.get("image_size", 96),
        color_bins=config.get("color_bins", 12),
        edge_bins=config.get("edge_bins", 16),
        gradient_size=config.get("gradient_size", 24),
    )
