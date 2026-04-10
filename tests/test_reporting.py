from __future__ import annotations

from banner_pipeline.reporting import build_metrics_report


def test_build_metrics_report_preserves_video_coverage_fields() -> None:
    report = build_metrics_report(
        [
            {
                "num_prompts": 3,
                "frames_with_masks": 12,
                "frames_with_quads": 9,
                "frames_composited": 7,
                "object_masks_total": 21,
                "first_frame_with_mask": 0,
                "last_frame_with_mask": 17,
                "max_consecutive_mask_gap": 4,
                "object_frame_coverage": {"1": {"frames_with_masks": 12, "coverage_ratio": 0.6}},
                "segment_total_s": 10.0,
                "run_total_s": 11.0,
                "sam3_reanchor_events": [{"obj_id": 1, "frame_idx": 8, "refresh_count": 1}],
            }
        ],
        benchmark_runs=1,
        gpu_name="NVIDIA A100",
        gpu_mem_gb=39.5,
        mode="video",
    )

    assert report["frames_with_masks"] == 12
    assert report["frames_with_quads"] == 9
    assert report["frames_composited"] == 7
    assert report["object_masks_total"] == 21
    assert report["first_frame_with_mask"] == 0
    assert report["last_frame_with_mask"] == 17
    assert report["max_consecutive_mask_gap"] == 4
    assert report["object_frame_coverage"] == {
        "1": {"frames_with_masks": 12, "coverage_ratio": 0.6}
    }
    assert report["sam3_reanchor_events"] == [{"obj_id": 1, "frame_idx": 8, "refresh_count": 1}]
