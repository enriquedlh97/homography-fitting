from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from banner_pipeline.segment.base import ObjectPrompt

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


def test_collect_prompts_persists_labels_and_frame_idx(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "sam3.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pipeline": {"segmenter": {"type": "sam3_video"}},
                "input": {"video": "data/tennis-clip.mp4"},
            },
            sort_keys=False,
        ),
    )
    monkeypatch.setattr(
        collect_prompts_mod,
        "load_frame",
        lambda _video_path, frame_idx=0: np.zeros((24, 32, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        collect_prompts_mod,
        "collect_clicks",
        lambda _frame, frame_idx=0: [
            ObjectPrompt(
                obj_id=1,
                points=np.array([[10.0, 20.0], [15.0, 25.0]], dtype=np.float32),
                labels=np.array([1, 0], dtype=np.int32),
                frame_idx=frame_idx,
            )
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_prompts.py",
            "--config",
            str(config_path),
            "--frame",
            "12",
        ],
    )

    collect_prompts_mod.main()

    saved = yaml.safe_load(config_path.read_text())
    assert saved["input"]["prompts"] == [
        {
            "obj_id": 1,
            "points": [[10, 20], [15, 25]],
            "labels": [1, 0],
            "frame_idx": 12,
        }
    ]


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
    assert len(sam3_cfg["input"]["prompts"]) == len(sam2_cfg["input"]["prompts"])
    assert all("labels" in prompt for prompt in sam3_cfg["input"]["prompts"])
    assert all(
        len(prompt["labels"]) == len(prompt["points"]) for prompt in sam3_cfg["input"]["prompts"]
    )
    assert any(0 in prompt["labels"] for prompt in sam3_cfg["input"]["prompts"])


def test_sam3_default_config_uses_inpaint_and_banner_only_prompts() -> None:
    sam3_cfg = yaml.safe_load((ROOT / "configs" / "sam3_default.yaml").read_text())
    fast_cfg = yaml.safe_load((ROOT / "configs" / "matrix" / "1prompt_fast.yaml").read_text())

    assert sam3_cfg["pipeline"]["compositor"]["type"] == "inpaint"
    assert fast_cfg["pipeline"]["compositor"]["type"] == "alpha"
    assert all(
        prompt.get("surface_type", "banner") == "banner" for prompt in sam3_cfg["input"]["prompts"]
    )
