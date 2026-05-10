"""GPU-family FlashAttention setup for SAM3."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Protocol, cast

import torch

_CONFIGURED_BACKEND: str | None = None
_CONFIGURED_GPU_FAMILY: str | None = None

_FlashAttnFunc = Callable[..., torch.Tensor]


class _FlashAttnModule(Protocol):
    flash_attn_func: _FlashAttnFunc


class _Sam3FlashModule(Protocol):
    flash_attn_func_op: _FlashAttnFunc
    flash_attn_func: _FlashAttnFunc


class _VitdetModule(Protocol):
    flash_attn_func: _FlashAttnFunc


def _current_cuda_gpu_name(device: torch.device) -> str:
    index = 0 if device.index is None else device.index
    return str(torch.cuda.get_device_name(index))


def _current_cuda_capability(device: torch.device) -> tuple[int, int]:
    index = 0 if device.index is None else device.index
    props = torch.cuda.get_device_properties(index)
    return int(props.major), int(props.minor)


def _gpu_family_for_sam3(
    gpu_name: str,
    capability: tuple[int, int] | None = None,
) -> str:
    upper = gpu_name.upper()
    if "B200" in upper:
        return "fa4"
    if "H100" in upper or "H200" in upper:
        return "fa2"
    if any(marker in upper for marker in ("L4", "A10G", "L40S", "A100")):
        return "fa2"
    if "T4" in upper:
        return "t4"

    if capability is None:
        raise RuntimeError(f"Unsupported GPU for SAM3 attention backend selection: {gpu_name}")

    major, minor = capability
    if major >= 10:
        return "fa4"
    if major >= 9:
        return "fa2"
    if major >= 8:
        return "fa2"
    if (major, minor) == (7, 5):
        return "t4"
    raise RuntimeError(
        f"Unsupported GPU for SAM3 attention backend selection: {gpu_name} (sm{major}{minor})"
    )


def _make_fa2_wrapper(
    flash_attn_func: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    def wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        causal = bool(kwargs.get("causal", False))
        softmax_scale = kwargs.get("softmax_scale")
        target_dtype = q.dtype

        if k.dtype != target_dtype:
            k = k.to(target_dtype)
        if v.dtype != target_dtype:
            v = v.to(target_dtype)

        return flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
        ).to(target_dtype)

    return wrapper


def _coerce_fa4_output_tensor(result: object) -> torch.Tensor:
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, tuple | list):
        if not result:
            raise RuntimeError(
                "FlashAttention-4 returned an empty tuple/list instead of an output tensor."
            )
        output = result[0]
        if isinstance(output, torch.Tensor):
            return output
        raise RuntimeError(
            "FlashAttention-4 returned a tuple/list whose first element is not a tensor: "
            f"{type(output).__name__}."
        )
    raise RuntimeError(
        f"FlashAttention-4 returned an unexpected result type: {type(result).__name__}."
    )


def _make_fa4_wrapper(
    flash_attn_func: Callable[..., object],
) -> Callable[..., torch.Tensor]:
    def wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        causal = bool(kwargs.get("causal", False))
        softmax_scale = kwargs.get("softmax_scale")
        target_dtype = q.dtype

        if k.dtype != target_dtype:
            k = k.to(target_dtype)
        if v.dtype != target_dtype:
            v = v.to(target_dtype)

        call_kwargs = {"causal": causal}
        if softmax_scale is not None:
            call_kwargs["softmax_scale"] = softmax_scale
        output = _coerce_fa4_output_tensor(flash_attn_func(q, k, v, **call_kwargs))
        return output.to(target_dtype)

    return wrapper


def _import_flash_attn_module(name: str) -> _FlashAttnModule:
    return cast(_FlashAttnModule, importlib.import_module(name))


def _import_sam3_flash_module(name: str) -> _Sam3FlashModule:
    return cast(_Sam3FlashModule, importlib.import_module(name))


def _import_vitdet_module() -> _VitdetModule:
    return cast(_VitdetModule, importlib.import_module("sam3.model.vitdet"))


def _patch_sam3_attention_modules(wrapper: Callable[..., torch.Tensor]) -> None:
    fa3_mod = _import_sam3_flash_module("sam3.perflib.fa3")
    fa3_mod.flash_attn_func_op = wrapper
    fa3_mod.flash_attn_func = wrapper

    vitdet_mod = _import_vitdet_module()
    vitdet_mod.flash_attn_func = wrapper

    try:
        fa2_mod = _import_sam3_flash_module("sam3.perflib.fa2")
    except ModuleNotFoundError:
        return

    fa2_mod.flash_attn_func_op = wrapper
    if hasattr(fa2_mod, "flash_attn_func"):
        fa2_mod.flash_attn_func = wrapper


def configure_sam3_attention_backend(device: torch.device | None = None) -> str:
    """Configure the SAM3 attention backend for the active CUDA GPU."""
    global _CONFIGURED_BACKEND, _CONFIGURED_GPU_FAMILY

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return "none"

    gpu_name = _current_cuda_gpu_name(device)
    capability = _current_cuda_capability(device)
    gpu_family = _gpu_family_for_sam3(gpu_name, capability)

    if gpu_family == "t4":
        raise RuntimeError(
            "SAM3 requires FlashAttention and is not supported on T4. "
            "Use SAM2 on T4 or switch SAM3 runs to L4/A10G/L40S/A100/H100/H200/B200."
        )

    if _CONFIGURED_BACKEND is not None and gpu_family == _CONFIGURED_GPU_FAMILY:
        return _CONFIGURED_BACKEND

    if gpu_family == "fa2":
        flash_attn_mod = _import_flash_attn_module("flash_attn")
        wrapper = _make_fa2_wrapper(flash_attn_mod.flash_attn_func)
        _patch_sam3_attention_modules(wrapper)
        backend = "fa2"
    else:
        flash_attn_cute_mod = _import_flash_attn_module("flash_attn.cute")
        wrapper = _make_fa4_wrapper(flash_attn_cute_mod.flash_attn_func)
        _patch_sam3_attention_modules(wrapper)
        backend = "fa4"

    _CONFIGURED_BACKEND = backend
    _CONFIGURED_GPU_FAMILY = gpu_family
    print(f"[SAM3] FlashAttention backend: {backend} on {gpu_name}", flush=True)
    return backend
