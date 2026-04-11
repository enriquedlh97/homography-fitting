from __future__ import annotations

import sys
import types
from typing import Any, cast

import pytest
import torch

from banner_pipeline import sam3_attention as attn_mod


def _reset_attention_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attn_mod, "_CONFIGURED_BACKEND", None)
    monkeypatch.setattr(attn_mod, "_CONFIGURED_GPU_FAMILY", None)


def _install_fake_sam3_attention_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any, Any]:
    sam3_pkg = cast(Any, types.ModuleType("sam3"))
    sam3_pkg.__path__ = []

    perflib_pkg = cast(Any, types.ModuleType("sam3.perflib"))
    perflib_pkg.__path__ = []
    model_pkg = cast(Any, types.ModuleType("sam3.model"))
    model_pkg.__path__ = []

    fa3_mod = cast(Any, types.ModuleType("sam3.perflib.fa3"))
    fa3_mod.flash_attn_func_op = lambda q, k, v, *args, **kwargs: q
    fa3_mod.flash_attn_func = lambda q, k, v, *args, **kwargs: q

    fa2_mod = cast(Any, types.ModuleType("sam3.perflib.fa2"))
    fa2_mod.flash_attn_func_op = lambda q, k, v, *args, **kwargs: q
    fa2_mod.flash_attn_func = lambda q, k, v, *args, **kwargs: q

    vitdet_mod = cast(Any, types.ModuleType("sam3.model.vitdet"))
    vitdet_mod.flash_attn_func = lambda q, k, v, *args, **kwargs: q

    sam3_pkg.perflib = perflib_pkg
    sam3_pkg.model = model_pkg
    perflib_pkg.fa3 = fa3_mod
    perflib_pkg.fa2 = fa2_mod
    model_pkg.vitdet = vitdet_mod

    monkeypatch.setitem(sys.modules, "sam3", sam3_pkg)
    monkeypatch.setitem(sys.modules, "sam3.perflib", perflib_pkg)
    monkeypatch.setitem(sys.modules, "sam3.perflib.fa3", fa3_mod)
    monkeypatch.setitem(sys.modules, "sam3.perflib.fa2", fa2_mod)
    monkeypatch.setitem(sys.modules, "sam3.model", model_pkg)
    monkeypatch.setitem(sys.modules, "sam3.model.vitdet", vitdet_mod)
    return fa3_mod, fa2_mod, vitdet_mod


def test_configure_sam3_attention_backend_selects_fa2_on_a100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_attention_state(monkeypatch)
    fa3_mod, fa2_mod, vitdet_mod = _install_fake_sam3_attention_modules(monkeypatch)
    recorded: dict[str, object] = {}

    flash_attn_mod = cast(Any, types.ModuleType("flash_attn"))

    def fake_flash_attn_func(q, k, v, dropout_p, softmax_scale, causal):
        recorded["dropout_p"] = dropout_p
        recorded["softmax_scale"] = softmax_scale
        recorded["causal"] = causal
        return q + k + v

    flash_attn_mod.flash_attn_func = fake_flash_attn_func
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn_mod)
    monkeypatch.setattr(attn_mod, "_current_cuda_gpu_name", lambda _device: "NVIDIA A100")
    monkeypatch.setattr(attn_mod, "_current_cuda_capability", lambda _device: (8, 0))

    backend = attn_mod.configure_sam3_attention_backend(torch.device("cuda"))

    q = torch.randn(2, 3, 4, 5, dtype=torch.float16)
    wrapper_out = vitdet_mod.flash_attn_func(q, q, q, causal=True, softmax_scale=0.125)

    assert backend == "fa2"
    assert wrapper_out.shape == q.shape
    assert wrapper_out.dtype == q.dtype
    assert recorded == {
        "dropout_p": 0.0,
        "softmax_scale": 0.125,
        "causal": True,
    }
    assert fa3_mod.flash_attn_func is vitdet_mod.flash_attn_func
    assert fa2_mod.flash_attn_func is vitdet_mod.flash_attn_func


@pytest.mark.parametrize("gpu_name", ["NVIDIA H100", "NVIDIA H200"])
def test_configure_sam3_attention_backend_selects_fa2_on_hopper(
    monkeypatch: pytest.MonkeyPatch,
    gpu_name: str,
) -> None:
    _reset_attention_state(monkeypatch)
    fa3_mod, fa2_mod, vitdet_mod = _install_fake_sam3_attention_modules(monkeypatch)
    recorded: dict[str, object] = {}

    flash_attn_mod = cast(Any, types.ModuleType("flash_attn"))

    def fake_flash_attn_func(q, k, v, dropout_p, softmax_scale, causal):
        recorded["dropout_p"] = dropout_p
        recorded["softmax_scale"] = softmax_scale
        recorded["causal"] = causal
        return q + k + v

    flash_attn_mod.flash_attn_func = fake_flash_attn_func
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn_mod)
    monkeypatch.setattr(attn_mod, "_current_cuda_gpu_name", lambda _device: gpu_name)
    monkeypatch.setattr(attn_mod, "_current_cuda_capability", lambda _device: (9, 0))

    backend = attn_mod.configure_sam3_attention_backend(torch.device("cuda"))

    q = torch.randn(2, 3, 4, 5, dtype=torch.float16)
    wrapper_out = vitdet_mod.flash_attn_func(q, q, q, causal=False, softmax_scale=0.25)

    assert backend == "fa2"
    assert wrapper_out.shape == q.shape
    assert wrapper_out.dtype == q.dtype
    assert recorded == {
        "dropout_p": 0.0,
        "softmax_scale": 0.25,
        "causal": False,
    }
    assert fa3_mod.flash_attn_func is vitdet_mod.flash_attn_func
    assert fa2_mod.flash_attn_func is vitdet_mod.flash_attn_func


def test_configure_sam3_attention_backend_selects_fa4_on_b200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_attention_state(monkeypatch)
    fa3_mod, fa2_mod, vitdet_mod = _install_fake_sam3_attention_modules(monkeypatch)
    recorded: dict[str, object] = {}

    flash_attn_pkg = cast(Any, types.ModuleType("flash_attn"))
    flash_attn_pkg.__path__ = []
    flash_attn_cute_mod = cast(Any, types.ModuleType("flash_attn.cute"))
    aux = torch.randn(2, 3, 4, dtype=torch.float32)

    def fake_flash_attn_func(q, k, v, causal, softmax_scale=None):
        recorded["causal"] = causal
        recorded["softmax_scale"] = softmax_scale
        return q + k + v, aux

    flash_attn_cute_mod.flash_attn_func = fake_flash_attn_func
    flash_attn_pkg.cute = flash_attn_cute_mod
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn_pkg)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", flash_attn_cute_mod)
    monkeypatch.setattr(attn_mod, "_current_cuda_gpu_name", lambda _device: "NVIDIA B200")
    monkeypatch.setattr(attn_mod, "_current_cuda_capability", lambda _device: (10, 0))

    backend = attn_mod.configure_sam3_attention_backend(torch.device("cuda"))

    q = torch.randn(2, 3, 4, 5, dtype=torch.bfloat16)
    wrapper_out = vitdet_mod.flash_attn_func(q, q, q, causal=False, softmax_scale=0.25)

    assert backend == "fa4"
    assert wrapper_out.shape == q.shape
    assert wrapper_out.dtype == q.dtype
    assert recorded == {"causal": False, "softmax_scale": 0.25}
    assert fa3_mod.flash_attn_func is vitdet_mod.flash_attn_func
    assert fa2_mod.flash_attn_func is vitdet_mod.flash_attn_func


def test_make_fa4_wrapper_accepts_tensor_only_return() -> None:
    recorded: dict[str, object] = {}

    def fake_flash_attn_func(q, k, v, causal, softmax_scale=None):
        recorded["causal"] = causal
        recorded["softmax_scale"] = softmax_scale
        return q + k + v

    wrapper = attn_mod._make_fa4_wrapper(fake_flash_attn_func)
    q = torch.randn(2, 3, 4, 5, dtype=torch.float16)

    wrapper_out = wrapper(q, q, q, causal=True, softmax_scale=0.5)

    assert wrapper_out.shape == q.shape
    assert wrapper_out.dtype == q.dtype
    assert recorded == {"causal": True, "softmax_scale": 0.5}


@pytest.mark.parametrize(
    ("bad_result", "expected_message"),
    [
        ((), "empty tuple/list"),
        ((None,), "first element is not a tensor"),
        ("not-a-tensor", "unexpected result type"),
    ],
)
def test_make_fa4_wrapper_rejects_invalid_return_shapes(
    bad_result: object,
    expected_message: str,
) -> None:
    wrapper = attn_mod._make_fa4_wrapper(lambda q, k, v, causal, softmax_scale=None: bad_result)
    q = torch.randn(2, 3, 4, 5, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match=expected_message):
        wrapper(q, q, q, causal=False)


def test_configure_sam3_attention_backend_is_idempotent_for_same_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_attention_state(monkeypatch)
    _, _, vitdet_mod = _install_fake_sam3_attention_modules(monkeypatch)

    flash_attn_mod = cast(Any, types.ModuleType("flash_attn"))
    flash_attn_mod.flash_attn_func = lambda q, k, v, dropout_p, softmax_scale, causal: q + k + v
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn_mod)
    monkeypatch.setattr(attn_mod, "_current_cuda_gpu_name", lambda _device: "NVIDIA A100")
    monkeypatch.setattr(attn_mod, "_current_cuda_capability", lambda _device: (8, 0))

    backend_1 = attn_mod.configure_sam3_attention_backend(torch.device("cuda"))
    wrapper_1 = vitdet_mod.flash_attn_func
    backend_2 = attn_mod.configure_sam3_attention_backend(torch.device("cuda"))

    assert backend_1 == "fa2"
    assert backend_2 == "fa2"
    assert wrapper_1 is vitdet_mod.flash_attn_func
