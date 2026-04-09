from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collect_prompts_mod = _load_script_module(
    ROOT / "scripts" / "collect_prompts.py",
    "test_collect_prompts_module",
)
matrix_mod = _load_script_module(
    ROOT / "scripts" / "run_matrix_parallel.py",
    "test_run_matrix_parallel_module",
)


def test_collect_prompts_uses_supported_gpu_hint_for_sam3() -> None:
    command = collect_prompts_mod._next_modal_command("configs/sam3_default.yaml", "sam3_video")

    assert "--gpu A100" in command
    assert "T4" not in command


def test_collect_prompts_keeps_t4_hint_for_sam2() -> None:
    command = collect_prompts_mod._next_modal_command("configs/default.yaml", "sam2_image")

    assert "--gpu T4" in command


def test_partition_supported_combinations_skips_sam3_t4(tmp_path: Path) -> None:
    sam3_cfg = tmp_path / "sam3.yaml"
    sam2_cfg = tmp_path / "sam2.yaml"
    sam3_cfg.write_text(
        yaml.safe_dump({"pipeline": {"segmenter": {"type": "sam3_video"}}}),
    )
    sam2_cfg.write_text(
        yaml.safe_dump({"pipeline": {"segmenter": {"type": "sam2_image"}}}),
    )

    runnable, skipped = matrix_mod._partition_supported_combinations(
        [str(sam3_cfg), str(sam2_cfg)],
        ["T4", "A100"],
    )

    assert runnable == [
        (str(sam3_cfg), "A100"),
        (str(sam2_cfg), "T4"),
        (str(sam2_cfg), "A100"),
    ]
    assert skipped == [
        (str(sam3_cfg), "T4", "SAM3 is unsupported on T4"),
    ]


@pytest.mark.parametrize("template_name", ["1prompt", "5prompts", "11prompts"])
def test_shipped_sam3_matrix_templates_preserve_prompt_shape(template_name: str) -> None:
    sam2_cfg = yaml.safe_load((ROOT / "configs" / "matrix" / f"{template_name}.yaml").read_text())
    sam3_cfg = yaml.safe_load(
        (ROOT / "configs" / "matrix" / f"sam3_{template_name}.yaml").read_text(),
    )

    assert sam3_cfg["pipeline"]["segmenter"] == {
        "type": "sam3_video",
        "checkpoint": "sam3/checkpoints/sam3.1_multiplex.pt",
    }
    assert sam3_cfg["input"]["prompts"] == sam2_cfg["input"]["prompts"]
