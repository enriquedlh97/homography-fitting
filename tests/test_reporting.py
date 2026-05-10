from __future__ import annotations

from banner_pipeline.reporting import build_metrics_report


def test_build_metrics_report_preserves_video_coverage_fields() -> None:
    report = build_metrics_report(
        [
            {
                "num_prompts": 3,
                "frames_with_masks": 12,
                "frames_with_valid_objects": 8,
                "frames_with_quads": 9,
                "frames_composited": 7,
                "object_masks_total": 21,
                "first_frame_with_mask": 0,
                "last_frame_with_mask": 17,
                "max_consecutive_mask_gap": 4,
                "object_frame_coverage": {"1": {"frames_with_masks": 12, "coverage_ratio": 0.6}},
                "object_valid_frame_coverage": {"1": {"frames_valid": 8, "coverage_ratio": 0.4}},
                "object_rejection_counts": {"1": 9},
                "object_rejection_reasons": {"1": {"quad_mask_iou_low": 9}},
                "segment_total_s": 10.0,
                "geometry_config_enabled": True,
                "geometry_runtime_enabled": True,
                "geometry_active_objects": [1],
                "object_geometry_model": {"1": "fronto_parallel_wall_banner"},
                "back_wall_runtime_model": {"1": "fronto_parallel_wall_banner"},
                "side_wall_runtime_model": {},
                "geometry_object_jitter_stats": {
                    "1": {
                        "count_samples": 11,
                        "median_corner_rms_px": 0.4,
                        "p95_corner_rms_px": 0.9,
                    }
                },
                "geometry_fit_method_counts": {"1": {"fronto_parallel_wall_banner": 12}},
                "geometry_total_s": 2.5,
                "court_width_candidate_count": 3.0,
                "court_depth_candidate_count": 4.0,
                "stabilization_config_enabled": True,
                "stabilization_runtime_enabled": True,
                "stabilization_total_s": 0.5,
                "stabilization_object_stats": {"1": {"frames_held": 2}},
                "compositor_config_enabled": True,
                "compositor_runtime_enabled": True,
                "compositor_total_s": 1.25,
                "compositor_object_model": {"1": "temporal_wall_plate"},
                "compositor_object_stats": {
                    "1": {
                        "plate_init_frame": 0,
                        "plate_reused_frames": 11,
                        "plate_reset_count": 1,
                        "delegated_inpaint_frames": 0,
                        "court_shading_updates": 0,
                    }
                },
                "git_branch": "feat/court-geometry-stabilisation",
                "git_commit_sha": "abc123",
                "git_dirty": True,
                "workspace_diff_sha256": "deadbeef",
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
    assert report["frames_with_valid_objects"] == 8
    assert report["frames_with_quads"] == 9
    assert report["frames_composited"] == 7
    assert report["object_masks_total"] == 21
    assert report["first_frame_with_mask"] == 0
    assert report["last_frame_with_mask"] == 17
    assert report["max_consecutive_mask_gap"] == 4
    assert report["object_frame_coverage"] == {
        "1": {"frames_with_masks": 12, "coverage_ratio": 0.6}
    }
    assert report["object_valid_frame_coverage"] == {
        "1": {"frames_valid": 8, "coverage_ratio": 0.4}
    }
    assert report["object_rejection_counts"] == {"1": 9}
    assert report["object_rejection_reasons"] == {"1": {"quad_mask_iou_low": 9}}
    assert report["sam3_reanchor_events"] == [{"obj_id": 1, "frame_idx": 8, "refresh_count": 1}]
    assert report["geometry_config_enabled"] is True
    assert report["geometry_runtime_enabled"] is True
    assert report["geometry_active_objects"] == [1]
    assert report["object_geometry_model"] == {"1": "fronto_parallel_wall_banner"}
    assert report["back_wall_runtime_model"] == {"1": "fronto_parallel_wall_banner"}
    assert report["side_wall_runtime_model"] == {}
    assert report["geometry_object_jitter_stats"] == {
        "1": {"count_samples": 11, "median_corner_rms_px": 0.4, "p95_corner_rms_px": 0.9}
    }
    assert report["geometry_fit_method_counts"] == {"1": {"fronto_parallel_wall_banner": 12}}
    assert report["geometry_total_s"] == 2.5
    assert report["court_width_candidate_count"] == 3.0
    assert report["court_depth_candidate_count"] == 4.0
    assert report["stabilization_config_enabled"] is True
    assert report["stabilization_runtime_enabled"] is True
    assert report["stabilization_total_s"] == 0.5
    assert report["stabilization_object_stats"] == {"1": {"frames_held": 2}}
    assert report["compositor_config_enabled"] is True
    assert report["compositor_runtime_enabled"] is True
    assert report["compositor_total_s"] == 1.25
    assert report["compositor_object_model"] == {"1": "temporal_wall_plate"}
    assert report["compositor_object_stats"] == {
        "1": {
            "plate_init_frame": 0,
            "plate_reused_frames": 11,
            "plate_reset_count": 1,
            "delegated_inpaint_frames": 0,
            "court_shading_updates": 0,
        }
    }
    assert report["git_branch"] == "feat/court-geometry-stabilisation"
    assert report["git_commit_sha"] == "abc123"
    assert report["git_dirty"] is True
    assert report["workspace_diff_sha256"] == "deadbeef"
