"""Torch device detection and SAM2/SAM3 model loading."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from banner_pipeline.sam3_attention import configure_sam3_attention_backend

# Apple MPS compatibility: fall back to CPU for unsupported ops.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def detect_device(override: str = "auto") -> torch.device:
    """Select the best available compute device.

    *override* can be ``"auto"`` (default), ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    if override != "auto":
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_torch_backend(device: torch.device) -> None:
    """Enable performance optimisations for *device* (autocast, TF32, etc.)."""
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        props = torch.cuda.get_device_properties(0)
        if props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def _ensure_sam2_importable() -> None:
    """Make sure the ``sam2`` package is importable.

    First tries a regular import.  If that fails, checks for a local ``sam2/``
    directory and temporarily adjusts ``sys.path``.
    """
    try:
        import sam2  # noqa: F401

        return
    except ImportError:
        pass

    # Look for a local clone of the sam2 repo.
    repo = os.path.join(os.getcwd(), "sam2")
    if os.path.isdir(repo) and repo not in sys.path:
        sys.path.insert(0, repo)

    try:
        import sam2  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SAM2 is not installed. Either:\n"
            "  1. pip install git+https://github.com/facebookresearch/sam2.git\n"
            "  2. Clone into ./sam2 and run: pip install -e ./sam2"
        ) from exc


def load_sam2_image_predictor(
    checkpoint: str,
    model_cfg: str,
    device: torch.device | None = None,
):
    """Build and return a ``SAM2ImagePredictor`` ready for inference."""
    _ensure_sam2_importable()
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device is None:
        device = detect_device()
    setup_torch_backend(device)

    model = build_sam2(model_cfg, checkpoint, device=device)
    return SAM2ImagePredictor(model)


def load_sam2_video_predictor(
    checkpoint: str,
    model_cfg: str,
    device: torch.device | None = None,
):
    """Build and return a SAM2 video predictor ready for inference."""
    _ensure_sam2_importable()
    from sam2.build_sam import build_sam2_video_predictor

    if device is None:
        device = detect_device()
    setup_torch_backend(device)

    return build_sam2_video_predictor(model_cfg, checkpoint, device=device)


def _ensure_sam3_importable() -> None:
    """Make sure the ``sam3`` package is importable.

    Mirrors the SAM2 helper: tries a regular import first, then checks for a
    local ``sam3/`` directory and adjusts ``sys.path`` if needed.
    """
    try:
        import sam3  # noqa: F401

        return
    except ModuleNotFoundError as exc:
        if "sam3" not in str(exc):
            # sam3 is installed but has a missing dependency — re-raise as-is
            raise

    # sam3 genuinely not found — try local directory fallback
    repo = os.path.join(os.getcwd(), "sam3")
    if os.path.isdir(repo) and repo not in sys.path:
        sys.path.insert(0, repo)

    try:
        import sam3  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SAM3 is not installed. Either:\n"
            "  1. pip install git+https://github.com/facebookresearch/sam3.git\n"
            "  2. Clone into ./sam3 and run: pip install -e ./sam3"
        ) from exc


def _get_sam3_video_builder(model_builder: object) -> tuple[Callable[..., Any], bool]:
    """Return the best available SAM3 video builder and whether it is multiplex."""
    multiplex_builder = getattr(model_builder, "build_sam3_multiplex_video_predictor", None)
    if callable(multiplex_builder):
        return multiplex_builder, True

    legacy_builder = getattr(model_builder, "build_sam3_video_predictor", None)
    if callable(legacy_builder):
        return legacy_builder, False

    raise ImportError(
        "sam3.model_builder does not expose a supported video builder. "
        "Expected build_sam3_multiplex_video_predictor or build_sam3_video_predictor."
    )


def _get_builder_signature(
    builder: Callable[..., Any],
) -> tuple[inspect.Signature, set[str], set[str]]:
    """Inspect *builder* and return its signature, kwarg params, and required params."""
    try:
        signature = inspect.signature(builder)
    except (TypeError, ValueError) as exc:
        name = getattr(builder, "__name__", repr(builder))
        raise TypeError(f"Could not inspect SAM3 builder signature for {name}.") from exc

    kwarg_params = {
        name
        for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    required_params = {
        name
        for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and param.default is inspect.Signature.empty
    }
    return signature, kwarg_params, required_params


def _raise_unsupported_sam3_builder(
    builder: Callable[..., Any],
    signature: inspect.Signature,
    message: str,
) -> None:
    name = getattr(builder, "__name__", repr(builder))
    raise TypeError(f"Unsupported SAM3 builder signature for {name}{signature}: {message}")


def load_sam3_video_predictor(
    checkpoint: str,
    device: torch.device | None = None,
):
    """Build and return a SAM 3 video predictor ready for inference.

    Upstream SAM3 has changed its builder API across releases:
    older builds expose ``build_sam3_video_predictor`` with a checkpoint
    argument, while newer multiplex builds expose
    ``build_sam3_multiplex_video_predictor`` and discover checkpoints via
    environment or alternate keyword names. This loader keeps the repo's
    ``checkpoint`` config stable and adapts to the installed SAM3 version.
    """
    _ensure_sam3_importable()
    model_builder = importlib.import_module("sam3.model_builder")
    builder, is_multiplex = _get_sam3_video_builder(model_builder)
    signature, kwarg_params, required_params = _get_builder_signature(builder)

    if device is None:
        device = detect_device()
    setup_torch_backend(device)
    configure_sam3_attention_backend(device)

    checkpoint_path = os.fspath(Path(checkpoint).expanduser())
    checkpoint_dir = os.path.dirname(checkpoint_path) or "."
    device_str = str(device)

    kwargs: dict[str, str] = {}
    if "checkpoint_dir" in kwarg_params:
        kwargs["checkpoint_dir"] = checkpoint_dir
    elif "checkpoint_path" in kwarg_params:
        kwargs["checkpoint_path"] = checkpoint_path
    elif "checkpoint" in kwarg_params:
        kwargs["checkpoint"] = checkpoint_path
    elif not is_multiplex:
        _raise_unsupported_sam3_builder(
            builder,
            signature,
            "expected one of checkpoint_dir, checkpoint_path, or checkpoint parameters.",
        )

    if "device" in kwarg_params:
        kwargs["device"] = device_str

    if is_multiplex:
        # Newer multiplex builders discover the checkpoint from the environment.
        os.environ["SAM31_CKPT_PATH"] = checkpoint_path

    missing_required = sorted(required_params - kwargs.keys())
    if missing_required:
        _raise_unsupported_sam3_builder(
            builder,
            signature,
            f"requires unsupported parameters: {', '.join(missing_required)}.",
        )

    return builder(**kwargs)
