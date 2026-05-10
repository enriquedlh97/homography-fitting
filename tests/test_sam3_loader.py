from __future__ import annotations

import os
import sys
import types
from typing import Any, cast

import pytest
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from banner_pipeline import device as device_mod


def _install_fake_sam3(monkeypatch: pytest.MonkeyPatch, builder_module: Any) -> None:
    sam3_pkg = cast(Any, types.ModuleType("sam3"))
    sam3_pkg.__path__ = []
    sam3_pkg.model_builder = builder_module
    monkeypatch.setitem(sys.modules, "sam3", sam3_pkg)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", builder_module)
    monkeypatch.setattr(device_mod, "_ensure_sam3_importable", lambda: None)
    monkeypatch.setattr(device_mod, "setup_torch_backend", lambda _device: None)
    monkeypatch.setattr(device_mod, "configure_sam3_attention_backend", lambda _device: "test")


def test_load_sam3_video_predictor_prefers_multiplex_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    sentinel = object()

    def build_sam3_multiplex_video_predictor():
        calls.append(("multiplex", {}))
        return sentinel

    def build_sam3_video_predictor(checkpoint: str, device: str):
        calls.append(("legacy", {"checkpoint": checkpoint, "device": device}))
        return object()

    builder_module = cast(Any, types.ModuleType("sam3.model_builder"))
    builder_module.build_sam3_multiplex_video_predictor = build_sam3_multiplex_video_predictor
    builder_module.build_sam3_video_predictor = build_sam3_video_predictor
    _install_fake_sam3(monkeypatch, builder_module)

    checkpoint = "/tmp/models/sam3.1_multiplex.pt"
    predictor = device_mod.load_sam3_video_predictor(checkpoint, torch.device("cpu"))

    assert predictor is sentinel
    assert calls == [("multiplex", {})]
    assert device_mod.os.environ["SAM31_CKPT_PATH"] == checkpoint


def test_load_sam3_video_predictor_supports_checkpoint_dir_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, str] = {}

    def build_sam3_video_predictor(checkpoint_dir: str, device: str):
        received["checkpoint_dir"] = checkpoint_dir
        received["device"] = device
        return "predictor"

    builder_module = cast(Any, types.ModuleType("sam3.model_builder"))
    builder_module.build_sam3_video_predictor = build_sam3_video_predictor
    _install_fake_sam3(monkeypatch, builder_module)

    predictor = device_mod.load_sam3_video_predictor(
        "/models/sam3.1/sam3.1_multiplex.pt",
        torch.device("cpu"),
    )

    assert predictor == "predictor"
    assert received == {
        "checkpoint_dir": "/models/sam3.1",
        "device": "cpu",
    }


def test_load_sam3_video_predictor_supports_checkpoint_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, str] = {}

    def build_sam3_video_predictor(checkpoint: str, device: str):
        received["checkpoint"] = checkpoint
        received["device"] = device
        return "predictor"

    builder_module = cast(Any, types.ModuleType("sam3.model_builder"))
    builder_module.build_sam3_video_predictor = build_sam3_video_predictor
    _install_fake_sam3(monkeypatch, builder_module)

    predictor = device_mod.load_sam3_video_predictor(
        "/models/sam3.1/sam3.1_multiplex.pt",
        torch.device("cpu"),
    )

    assert predictor == "predictor"
    assert received == {
        "checkpoint": "/models/sam3.1/sam3.1_multiplex.pt",
        "device": "cpu",
    }


def test_load_sam3_video_predictor_rejects_unsupported_builder_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build_sam3_video_predictor(model_name: str, device: str):
        return {"model_name": model_name, "device": device}

    builder_module = cast(Any, types.ModuleType("sam3.model_builder"))
    builder_module.build_sam3_video_predictor = build_sam3_video_predictor
    _install_fake_sam3(monkeypatch, builder_module)

    with pytest.raises(TypeError, match="Unsupported SAM3 builder signature"):
        device_mod.load_sam3_video_predictor(
            "/models/sam3.1/sam3.1_multiplex.pt",
            torch.device("cpu"),
        )


def test_build_video_segmenter_accepts_existing_sam3_checkpoint_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from banner_pipeline import pipeline as pipeline_mod

    class FakeSAM3VideoSegmenter:
        def __init__(self, checkpoint: str, device: str = "auto") -> None:
            self.checkpoint = checkpoint
            self.device = device

    monkeypatch.setattr(pipeline_mod, "SAM3VideoSegmenter", FakeSAM3VideoSegmenter)

    segmenter = pipeline_mod.build_video_segmenter(
        {
            "type": "sam3_video",
            "checkpoint": "/checkpoints/sam3.1_multiplex.pt",
            "device": "cuda",
        }
    )

    assert isinstance(segmenter, FakeSAM3VideoSegmenter)
    assert segmenter.checkpoint == "/checkpoints/sam3.1_multiplex.pt"
    assert segmenter.device == "cuda"
