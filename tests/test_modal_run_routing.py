from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "modal_run.py"


class _FakeImage:
    def __init__(self, label: str) -> None:
        self.label = label
        self.steps: list[tuple[str, object]] = []

    @classmethod
    def from_registry(cls, base: str, add_python: str | None = None) -> _FakeImage:
        image = cls(f"{base}:{add_python}")
        image.steps.append(("from_registry", (base, add_python)))
        return image

    def apt_install(self, *packages: str) -> _FakeImage:
        self.steps.append(("apt_install", packages))
        return self

    def run_commands(self, *commands: str) -> _FakeImage:
        self.steps.append(("run_commands", commands))
        return self

    def pip_install(self, *packages: str, **kwargs) -> _FakeImage:
        self.steps.append(("pip_install", (packages, kwargs)))
        return self

    def add_local_dir(self, *args, **kwargs) -> _FakeImage:
        self.steps.append(("add_local_dir", (args, kwargs)))
        return self


class _FakeApp:
    def __init__(self, name: str, image: _FakeImage | None = None) -> None:
        self.name = name
        self.image = image

    def function(self, **kwargs):
        def decorator(fn):
            fn._modal_function_kwargs = kwargs
            return fn

        return decorator

    def local_entrypoint(self):
        def decorator(fn):
            return fn

        return decorator


def _fake_modal_module() -> types.ModuleType:
    modal_mod = cast(Any, types.ModuleType("modal"))
    modal_mod.Image = _FakeImage
    modal_mod.App = _FakeApp
    modal_mod.Volume = types.SimpleNamespace(from_name=lambda *args, **kwargs: object())
    modal_mod.Secret = types.SimpleNamespace(from_name=lambda *args, **kwargs: object())
    return cast(types.ModuleType, modal_mod)


def _load_modal_run_module(monkeypatch: pytest.MonkeyPatch, gpu: str):
    module_name = f"test_modal_run_{gpu.lower().replace('-', '_')}"
    monkeypatch.setitem(sys.modules, "modal", _fake_modal_module())
    monkeypatch.setattr(sys, "argv", ["scripts/modal_run.py", "--gpu", gpu])

    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("gpu", "image_attr"),
    [
        ("T4", "t4_image"),
        ("A100", "fa2_image"),
        ("H100", "fa2_image"),
        ("H200", "fa2_image"),
        ("B200", "fa4_image"),
    ],
)
def test_modal_run_selects_image_for_gpu(
    monkeypatch: pytest.MonkeyPatch,
    gpu: str,
    image_attr: str,
) -> None:
    module = _load_modal_run_module(monkeypatch, gpu)

    assert module.image is getattr(module, image_attr)
    assert module._select_image_for_gpu(gpu) is getattr(module, image_attr)


def test_modal_run_b200_image_pins_flash_attn_4_beta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_modal_run_module(monkeypatch, "B200")

    assert module._select_image_for_gpu("B200") is module.fa4_image
    assert (
        "run_commands",
        ("python -m pip install --no-cache-dir 'flash-attn-4==4.0.0b8'",),
    ) in module.fa4_image.steps


def test_modal_run_rejects_sam3_on_t4(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_modal_run_module(monkeypatch, "T4")
    config_dict = {"pipeline": {"segmenter": {"type": "sam3_video"}}}

    with pytest.raises(SystemExit, match="SAM3 requires FlashAttention"):
        module._validate_gpu_config(config_dict, "T4")


def test_modal_run_accepts_sam2_on_t4(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_modal_run_module(monkeypatch, "T4")
    config_dict = {"pipeline": {"segmenter": {"type": "sam2_image"}}}

    assert module._validate_gpu_config(config_dict, "T4") is None


def test_modal_run_git_output_returns_trimmed_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_modal_run_module(monkeypatch, "A100")

    def _fake_run(*_args, **_kwargs):
        return SimpleNamespace(stdout="feat/court-geometry-stabilisation\n")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    assert module._git_output("branch", "--show-current") == "feat/court-geometry-stabilisation"


@pytest.mark.parametrize("gpu", ["A100", "H100", "H200"])
def test_modal_run_uses_fa2_image_for_supported_sam3_gpus(
    monkeypatch: pytest.MonkeyPatch,
    gpu: str,
) -> None:
    module = _load_modal_run_module(monkeypatch, gpu)

    assert module._select_image_for_gpu(gpu) is module.fa2_image
    assert (
        "run_commands",
        ("python -m pip install --no-cache-dir flash-attn --no-build-isolation",),
    ) in module.fa2_image.steps
