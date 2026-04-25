"""Person occlusion masking for court floor logo compositing.

Uses torchvision's Mask R-CNN to detect people in each frame,
producing a binary mask that prevents logos from rendering on top
of players.
"""

from __future__ import annotations

import numpy as np
import torch


class PersonMasker:
    """Detect people in a frame and return a binary occlusion mask."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model: torch.nn.Module | None = None

    def _load_model(self) -> None:
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_Weights,
            maskrcnn_resnet50_fpn,
        )

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self._model = maskrcnn_resnet50_fpn(weights=weights)
        self._model.to(self.device)
        self._model.eval()

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a (H, W) float32 mask where 1.0 = person pixel.

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR frame (H, W, 3) uint8.

        Returns
        -------
        np.ndarray
            (H, W) float32 mask, 0.0 = background, 1.0 = person.
        """
        if self._model is None:
            self._load_model()
        assert self._model is not None

        h, w = frame_bgr.shape[:2]
        # BGR → RGB, normalize to [0, 1]
        rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)

        with torch.no_grad():
            predictions = self._model([tensor])[0]

        # Filter for person class (COCO label 1) above confidence threshold.
        person_mask = np.zeros((h, w), dtype=np.float32)
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        masks = predictions["masks"].cpu().numpy()

        for i, (label, score) in enumerate(zip(labels, scores, strict=False)):
            if label == 1 and score >= self.confidence_threshold:
                instance_mask = masks[i, 0]  # (H, W) float
                person_mask = np.maximum(person_mask, instance_mask)

        # Binarize at 0.5 for clean edges.
        return (person_mask > 0.5).astype(np.float32)
