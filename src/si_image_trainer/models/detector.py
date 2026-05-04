from __future__ import annotations

from pathlib import Path

from PIL import Image

from si_image_trainer.utils.image import open_image


class MosaicDetector:
    def __init__(self, model_path: str | Path, conf: float = 0.25) -> None:
        from ultralytics import YOLO
        self._model = YOLO(str(model_path))
        self._conf = conf

    def crop(self, image_path: str | Path) -> Image.Image | None:
        """Return a cropped PIL image of the best detection, or None if nothing found."""
        results = self._model(str(image_path), conf=self._conf, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        # Pick highest-confidence box
        best = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best].tolist()
        image = open_image(image_path)
        return image.crop((x1, y1, x2, y2))
