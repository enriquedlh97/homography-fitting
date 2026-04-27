"""Person occlusion masking for court floor logo compositing.

Three masker backends:
- PersonMasker: Mask R-CNN per-frame (binary masks, legacy)
- RVMMasker: RobustVideoMatting (continuous alpha mattes)
- SAM2VideoPersonMasker: SAM2 video propagation (pre-computed, best quality)

MaskSmoother: temporal post-processor (morph close + dropout protection).
Exact reproduction of tennis-virtual-ads mask_smoother.py.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
import torch


class PersonMasker:
    """Detect people via Mask R-CNN. Returns binary mask."""

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
        """Return (H, W) float32 mask, 1.0 = person pixel."""
        if self._model is None:
            self._load_model()
        assert self._model is not None

        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)

        with torch.no_grad():
            predictions = self._model([tensor])[0]

        person_mask = np.zeros((h, w), dtype=np.float32)
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        masks = predictions["masks"].cpu().numpy()

        for i, (label, score) in enumerate(zip(labels, scores, strict=False)):
            if label == 1 and score >= self.confidence_threshold:
                instance_mask = masks[i, 0]
                person_mask = np.maximum(person_mask, instance_mask)

        return (person_mask > 0.5).astype(np.float32)


class RVMMasker:
    """Soft alpha matting via RobustVideoMatting.

    Produces continuous float32 alpha values in [0, 1] instead of
    binary masks. Built-in temporal consistency via ConvGRU recurrent
    state — no flicker between frames.

    Loaded via torch.hub (auto-downloads ~9MB weights on first run).
    """

    def __init__(
        self,
        backbone: str = "mobilenetv3",
        downsample_ratio: float = 0.25,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._backbone = backbone
        self._downsample_ratio = downsample_ratio
        self._model: torch.nn.Module | None = None
        self._rec: list[Any] = [None, None, None, None]

    def _load_model(self) -> None:
        print(f"[RVM] Loading RobustVideoMatting (backbone={self._backbone}) on {self.device} ...")
        # Monkey-patch for PyTorch 2.7+ compatibility.
        # RVM passes list instead of tuple for scale_factors in upsample.
        import torch.nn.functional as F

        _orig_interpolate = F.interpolate

        def _patched_interpolate(
            input,
            size=None,
            scale_factor=None,
            mode="nearest",
            align_corners=None,
            recompute_scale_factor=None,
            antialias=False,
        ):
            if isinstance(scale_factor, list):
                scale_factor = tuple(scale_factor)
            return _orig_interpolate(
                input,
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners,
                recompute_scale_factor=recompute_scale_factor,
                antialias=antialias,
            )

        F.interpolate = _patched_interpolate

        self._model = torch.hub.load(
            "PeterL1n/RobustVideoMatting",
            self._backbone,
            trust_repo=True,
        )
        self._model.to(self.device)
        self._model.eval()
        print("[RVM] Model loaded.")

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return (H, W) float32 continuous alpha matte, 0.0=bg, 1.0=person.

        Unlike Mask R-CNN, this returns soft alpha values — a motion-blurred
        limb or semi-transparent shoe edge naturally gets alpha ~0.5.
        """
        if self._model is None:
            self._load_model()
        assert self._model is not None

        rgb = frame_bgr[:, :, ::-1].copy()
        src = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        src = src.to(self.device)
        downsample = torch.tensor([self._downsample_ratio], device=self.device)

        with torch.no_grad():
            fgr, pha, *self._rec = self._model(src, *self._rec, downsample)

        alpha: np.ndarray = pha[0, 0].cpu().numpy()  # (H, W) float32 [0, 1]
        return alpha

    def reset(self) -> None:
        """Reset recurrent state (call on scene cuts)."""
        self._rec = [None, None, None, None]


class SAM2VideoPersonMasker:
    """Pre-compute player masks for all frames via SAM2 video propagation.

    Detects persons on a prompt frame using Mask R-CNN, converts each
    detection to a center-point SAM2 prompt, and propagates masks across
    the full video. All masks are stored at init time — per-frame lookup
    is O(1).

    Reuses an existing frame directory (from text segmentation) to avoid
    re-extracting video frames.
    """

    def __init__(
        self,
        frame_dir: str,
        frame_names: list[str],
        checkpoint: str = "sam2/checkpoints/sam2.1_hiera_tiny.pt",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        confidence_threshold: float = 0.5,
        prompt_frame_idx: int = 0,
        device: str | None = None,
    ) -> None:
        self._frame_dir = frame_dir
        self._frame_names = frame_names
        self._masks: dict[int, np.ndarray] = {}

        # Read the prompt frame for person detection.
        frame_path = os.path.join(frame_dir, frame_names[prompt_frame_idx])
        frame_bgr = cv2.imread(frame_path)
        if frame_bgr is None:
            raise RuntimeError(f"Cannot read frame {prompt_frame_idx}: {frame_path}")

        fh, fw = frame_bgr.shape[:2]

        # Detect persons on the prompt frame.
        boxes = self._detect_persons(frame_bgr, confidence_threshold, device)
        if not boxes:
            print("[SAM2PersonMasker] No persons detected — all frames get empty masks")
            for i in range(len(frame_names)):
                self._masks[i] = np.zeros((fh, fw), dtype=np.uint8)
            return

        print(f"[SAM2PersonMasker] Detected {len(boxes)} persons on frame {prompt_frame_idx}")

        # Propagate player masks across the full video.
        self._propagate(
            boxes,
            prompt_frame_idx,
            checkpoint,
            model_cfg,
            device,
            (fh, fw),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _detect_persons(
        frame_bgr: np.ndarray,
        confidence_threshold: float,
        device: str | None,
    ) -> list[list[float]]:
        """Detect persons using Mask R-CNN. Returns list of [x1,y1,x2,y2]."""
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_Weights,
            maskrcnn_resnet50_fpn,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(weights=weights)
        model.to(device)
        model.eval()

        rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device)

        with torch.no_grad():
            predictions = model([tensor])[0]

        boxes: list[list[float]] = []
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        pred_boxes = predictions["boxes"].cpu().numpy()

        for i, (label, score) in enumerate(zip(labels, scores, strict=False)):
            if label == 1 and score >= confidence_threshold:
                boxes.append(pred_boxes[i].tolist())

        # Free Mask R-CNN from GPU immediately.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return boxes

    # ------------------------------------------------------------------

    def _propagate(
        self,
        boxes: list[list[float]],
        prompt_frame_idx: int,
        checkpoint: str,
        model_cfg: str,
        device: str | None,
        frame_shape: tuple[int, int],
    ) -> None:
        """Build SAM2 video predictor, add person prompts, propagate."""
        from banner_pipeline.device import detect_device, load_sam2_video_predictor

        dev = detect_device(device or "auto")
        predictor = load_sam2_video_predictor(checkpoint, model_cfg, dev)
        inference_state = predictor.init_state(video_path=self._frame_dir)

        # Use BOX prompts (not just center points) — gives SAM2 much
        # better context about the full person extent, especially feet.
        for obj_id, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box
            box_arr = np.array([x1, y1, x2, y2], dtype=np.float32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=prompt_frame_idx,
                obj_id=obj_id,
                box=box_arr,
            )
            print(f"[SAM2PersonMasker] Player {obj_id}: box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

        # Propagate masks across all frames.
        print("[SAM2PersonMasker] Propagating player masks …")
        fh, fw = frame_shape
        with torch.inference_mode():
            for fidx, _obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
                # Union all per-player masks into a single binary mask.
                binary = (mask_logits > 0.0).squeeze(1)  # (N, H, W)
                union = binary.any(dim=0)  # (H, W)
                self._masks[fidx] = union.cpu().numpy().astype(np.uint8)

        # Note: bbox floor extension removed — it creates a wide
        # rectangular mask that restores too much original text near
        # the player. With box prompts, SAM2 should track feet well
        # enough. The dilation in _erase_original_text provides the
        # remaining foot coverage.

        # Fill any missing frames with empty masks.
        for i in range(len(self._frame_names)):
            if i not in self._masks:
                self._masks[i] = np.zeros((fh, fw), dtype=np.uint8)

        # Cleanup.
        predictor.reset_state(inference_state)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        n_nonempty = sum(1 for m in self._masks.values() if m.any())
        print(
            f"[SAM2PersonMasker] Done — {len(self._masks)} frames, "
            f"{n_nonempty} with detected players"
        )

    # ------------------------------------------------------------------

    def mask(self, frame_idx: int) -> np.ndarray:
        """Return (H, W) float32 mask for *frame_idx*. 1.0 = person pixel."""
        if frame_idx in self._masks:
            return self._masks[frame_idx].astype(np.float32)
        # Out of range — return empty mask.
        if self._masks:
            ref = next(iter(self._masks.values()))
            return np.zeros_like(ref, dtype=np.float32)
        return np.zeros((1, 1), dtype=np.float32)


class MaskSmoother:
    """Temporal mask post-processor — exact reproduction of tennis-virtual-ads.

    Two operations applied in order:
    1. Morphological close (fills arm-body and foot-court gaps)
    2. Single-frame dropout protection (union with previous frame)
    """

    def __init__(self, close_px: int = 7) -> None:
        self._close_px = close_px
        self._prev_raw: np.ndarray | None = None

    def update(self, raw_mask: np.ndarray) -> np.ndarray:
        """Smooth *raw_mask* and return the processed version."""
        mask = raw_mask.copy()

        # 1. Morphological close — fills small gaps in the silhouette
        # (e.g. between arm and body, or between shoe sole and court).
        if self._close_px > 0 and np.any(mask > 0):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self._close_px + 1, 2 * self._close_px + 1),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 2. Single-frame dropout protection — union with previous
        # frame's raw mask prevents flickering from single-frame
        # detection failures.
        if self._prev_raw is not None:
            mask = np.maximum(mask, self._prev_raw)
        self._prev_raw = raw_mask.copy()

        return mask
