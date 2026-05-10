# Pretrained model weights

Place external model weights here.

## BallTrackerNet (court keypoint detector) — for `geometry.court_backend: ball_tracker_net_v1`

File: `tennis_court_detector.pt` (~80 MB, **not** tracked in git).

Download from:
  https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG

Or copy from the sibling repo if present:
  `cp ../tennis-virtual-ads/weights/tennis_court_detector.pt ./`

The estimator (`src/banner_pipeline/court_geometry_ball_tracker.py`) also
auto-discovers the weights from these locations (in order):

  1. `$BANNER_PIPELINE_BTN_WEIGHTS` (env var, absolute path)
  2. `<repo_root>/weights/tennis_court_detector.pt` (this directory)
  3. Any ancestor directory's `tennis-virtual-ads/weights/tennis_court_detector.pt`
  4. Any ancestor directory's `weights/tennis_court_detector.pt`

When running on Modal, `scripts/modal_run.py` mounts this directory to
`/root/weights` if it is non-empty.
