#!/usr/bin/env python3
"""Run video inpainting models to remove MELBOURNE text from court.

Generates a clean video (no text) that can be used as the base for
logo overlay. Three models available: ProPainter, DiffuEraser, VOID.

Usage:
    uv run modal run scripts/video_inpaint.py --model propainter
    uv run modal run scripts/video_inpaint.py --model diffueraser
    uv run modal run scripts/video_inpaint.py --model void
"""
from __future__ import annotations

import sys

import modal

app = modal.App("video-inpaint")

# --- Parse --model from CLI ---
_MODEL = "propainter"
for i, arg in enumerate(sys.argv):
    if arg == "--model" and i + 1 < len(sys.argv):
        _MODEL = sys.argv[i + 1]

# --- Base image with common deps ---
_base = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04", add_python="3.11"
    )
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0", "build-essential", "gcc", "g++")
    .pip_install(
        "torch>=2.0", "torchvision>=0.17", "opencv-python>=4.8",
        "numpy", "scipy", "Pillow", "imageio", "imageio-ffmpeg",
    )
)

# --- ProPainter image ---
propainter_image = (
    _base
    .run_commands(
        "git clone https://github.com/sczhou/ProPainter.git /opt/propainter",
        "cd /opt/propainter && pip install -r requirements.txt",
    )
)

from pathlib import Path as _Path
_REPO_SRC = str(_Path(__file__).resolve().parents[1] / "src")

# --- ProPainter + SAM2 image (for player-aware inpaint) ---
# Adds SAM2 so we can mask out the player region from the inpaint mask,
# preventing inpainting models from erasing the player and leaving smeary
# leg artifacts. mask_for_inpaint = MELBOURNE_quad AND NOT player_silhouette.
propainter_sam2_image = (
    propainter_image
    .pip_install("hydra-core", "iopath")
    .run_commands(
        "git clone https://github.com/facebookresearch/sam2.git /tmp/sam2",
        "cd /tmp/sam2 && pip install -e '.[all]'",
    )
    .add_local_dir(_REPO_SRC, remote_path="/root/src")
)

# Reuse the SAM2 checkpoints volume from the banner-pipeline app
sam_checkpoints_vol = modal.Volume.from_name(
    "banner-pipeline-checkpoints", create_if_missing=True
)

# --- DiffuEraser image ---
# Diffusion-based video inpainting (Jan 2025). Refines ProPainter priori
# via a fine-tuned Stable Diffusion v1.5 + BrushNet pipeline.
diffueraser_image = (
    _base
    .pip_install(
        "diffusers==0.29.2", "transformers", "accelerate", "safetensors",
        "huggingface-hub", "omegaconf", "av", "decord", "scikit-image",
        "peft",
    )
    .run_commands(
        "git clone https://github.com/lixiaowen-xw/DiffuEraser.git /opt/diffueraser",
        "cd /opt/diffueraser && pip install -r requirements.txt || true",
    )
)

# Volume to cache DiffuEraser weights (~7GB) across runs
diffueraser_weights_vol = modal.Volume.from_name(
    "diffueraser-weights", create_if_missing=True
)

# --- VOID image ---
void_image = (
    _base
    .pip_install(
        "diffusers", "transformers", "accelerate", "safetensors",
        "huggingface-hub", "einops", "decord",
    )
    .run_commands(
        "pip install cogvideox-fun || true",
    )
)


def _generate_text_mask(video_bytes: bytes, quad_corners: list) -> tuple[bytes, int, int]:
    """Generate per-frame binary masks for the MELBOURNE text region."""
    import os
    import tempfile

    import cv2
    import numpy as np

    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, "input.mp4")
    mask_dir = os.path.join(tmpdir, "masks")
    os.makedirs(mask_dir)

    with open(video_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(video_path)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = float(cap.get(cv2.CAP_PROP_FPS))

    # Build mask from quad corners (same for all frames since text is static)
    corners = np.array(quad_corners, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillPoly(mask, [corners], 255)
    # Dilate to ensure full text coverage
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.dilate(mask, kern, iterations=1)

    # Count actual frames by reading (frame_count can be inaccurate)
    actual_count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        actual_count += 1
    cap.release()

    # Save mask for each frame (use actual count)
    for i in range(actual_count):
        cv2.imwrite(os.path.join(mask_dir, f"{i:05d}.png"), mask)

    # Build mask video. DiffuEraser does strict FPS equality between mask
    # and input video, so we use cv2.VideoWriter at the same fps cv2 reports
    # for the input. (For DiffuEraser, the caller may also need to re-encode
    # the input video to ensure both report the same fps.)
    mask_video_path = os.path.join(tmpdir, "mask.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mask_video_path, fourcc, input_fps, (fw, fh), False)
    for _ in range(actual_count):
        writer.write(mask)
    writer.release()

    with open(mask_video_path, "rb") as f:
        mask_video_bytes = f.read()

    return mask_video_bytes, fw, fh


def _run_propainter_inner(video_bytes: bytes, mask_video_bytes: bytes, full_quality: bool) -> dict:
    """Shared ProPainter runner. full_quality=True keeps native res."""
    import os
    import subprocess
    import tempfile

    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}, full_quality={full_quality}")

    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, "input.mp4")
    mask_dir = os.path.join(tmpdir, "masks")
    os.makedirs(mask_dir)

    with open(video_path, "wb") as f:
        f.write(video_bytes)

    import cv2

    frame_dir = os.path.join(tmpdir, "frames")
    os.makedirs(frame_dir)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frame_dir, f"{i:05d}.png"), frame)
        i += 1
    cap.release()
    print(f"Extracted {i} video frames")

    mask_video_path = os.path.join(tmpdir, "mask.mp4")
    with open(mask_video_path, "wb") as f:
        f.write(mask_video_bytes)

    cap_mask = cv2.VideoCapture(mask_video_path)
    ret, mask_frame = cap_mask.read()
    cap_mask.release()
    if ret:
        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY) if len(mask_frame.shape) == 3 else mask_frame
    else:
        import numpy as np
        mask_gray = np.ones((1074, 1912), dtype=np.uint8) * 255

    for j in range(i):
        cv2.imwrite(os.path.join(mask_dir, f"{j:05d}.png"), mask_gray)
    print(f"Wrote {i} mask frames (matching video)")

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = [
        "python", "inference_propainter.py",
        "--video", frame_dir,
        "--mask", mask_dir,
        "--fp16",
        "--save_fps", "59",
    ]
    if full_quality:
        # B200 (192GB): full resolution, larger subvideo for quality.
        args += ["--subvideo_length", "40"]
    else:
        # H200 fast iteration: half-res inpaint, smaller chunks.
        args += [
            "--subvideo_length", "30",
            "--neighbor_length", "5",
            "--ref_stride", "20",
            "--resize_ratio", "0.5",
        ]
    result = subprocess.run(args, cwd="/opt/propainter", capture_output=True, text=True, env=env)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-1000:]}")

    # ProPainter writes 2 videos: inpaint_out.mp4 (actual result) and
    # masked_in.mp4 (green-tinted mask visualization). We want inpaint_out.
    output_dir = "/opt/propainter/results"
    inpaint_path = None
    all_mp4s = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".mp4"):
                full = os.path.join(root, f)
                all_mp4s.append(full)
                if f == "inpaint_out.mp4":
                    inpaint_path = full
    print(f"Found mp4s in results: {all_mp4s}")
    if inpaint_path is None and all_mp4s:
        # Fallback: any non-masked file
        for p in all_mp4s:
            if "masked" not in os.path.basename(p):
                inpaint_path = p
                break
    if inpaint_path is None:
        return {"error": "No inpaint_out.mp4 produced", "model": "propainter"}
    with open(inpaint_path, "rb") as fh:
        return {"output_bytes": fh.read(), "model": "propainter"}


@app.function(gpu="B200", image=propainter_image, timeout=3600)
def run_propainter(video_bytes: bytes, mask_video_bytes: bytes) -> dict:
    """ProPainter at full quality on B200 (192GB, native resolution)."""
    return _run_propainter_inner(video_bytes, mask_video_bytes, full_quality=True)


@app.function(gpu="H200", image=propainter_image, timeout=3600)
def run_propainter_fast(video_bytes: bytes, mask_video_bytes: bytes) -> dict:
    """ProPainter fast iteration on H200 (resize_ratio=0.5)."""
    return _run_propainter_inner(video_bytes, mask_video_bytes, full_quality=False)


def _download_sam2_checkpoint(filename: str, dest: str) -> None:
    """Download SAM2 checkpoint from Meta CDN if not cached."""
    import os
    import urllib.request
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{filename}"
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)


def _generate_player_aware_masks(
    video_path: str,
    mask_dir: str,
    quad_corners: list,
    player_dilate_px: int = 18,
    temporal_window: int = 0,
) -> int:
    """Run SAM2 to detect the player per frame; write inpaint masks
    (MELBOURNE_quad AND NOT player_dilated) as PNGs in mask_dir.

    Returns the number of mask frames written.
    """
    import os
    import sys
    import shutil
    import tempfile

    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, "/root/src")
    from banner_pipeline.masking import SAM2VideoPersonMasker

    # Ensure SAM2 checkpoint exists in the volume
    ckpt_dst = "/sam_checkpoints/sam2.1_hiera_large.pt"
    _download_sam2_checkpoint("sam2.1_hiera_large.pt", ckpt_dst)
    sam_checkpoints_vol.commit()

    # Extract video frames to a tmp directory (SAM2VideoPersonMasker needs this)
    cap = cv2.VideoCapture(video_path)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = 0
    frame_tmp = tempfile.mkdtemp(prefix="sam2_frames_")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # SAM2 video predictor expects JPEGs named by frame number
        cv2.imwrite(os.path.join(frame_tmp, f"{n_frames:05d}.jpg"), frame)
        n_frames += 1
    cap.release()
    print(f"[player-aware] Extracted {n_frames} frames to {frame_tmp}")

    frame_names = sorted(os.listdir(frame_tmp))

    # SAM2 video propagation (uses Mask R-CNN to detect persons on frame 0)
    masker = SAM2VideoPersonMasker(
        frame_dir=frame_tmp,
        frame_names=frame_names,
        checkpoint=ckpt_dst,
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        confidence_threshold=0.5,
        prompt_frame_idx=0,
    )

    # Build static MELBOURNE quad
    quad = np.array(quad_corners, dtype=np.int32).reshape((-1, 1, 2))
    quad_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillPoly(quad_mask, [quad], 255)
    kern_quad = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    quad_mask = cv2.dilate(quad_mask, kern_quad, iterations=1)

    # Per-frame: inpaint mask = quad AND NOT (player_mask dilated)
    if player_dilate_px > 0:
        kern_p = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * player_dilate_px + 1, 2 * player_dilate_px + 1),
        )
    else:
        kern_p = None

    # Pre-compute dilated binary masks for all frames (so we can union them
    # over a temporal window to cover motion-blur trails).
    binary_masks: list[np.ndarray] = []
    for i in range(n_frames):
        player_m = masker.mask(i)
        player_b = (player_m > 0.5).astype(np.uint8) * 255
        if kern_p is not None and np.any(player_b > 0):
            player_b = cv2.dilate(player_b, kern_p, iterations=1)
        binary_masks.append(player_b)

    for i in range(n_frames):
        if temporal_window > 0:
            lo = max(0, i - temporal_window)
            hi = min(n_frames, i + temporal_window + 1)
            player_b = binary_masks[lo]
            for j in range(lo + 1, hi):
                player_b = np.maximum(player_b, binary_masks[j])
        else:
            player_b = binary_masks[i]
        # Subtract player path from quad: pixels in quad AND not player anywhere in window
        inpaint_m = np.where((quad_mask > 0) & (player_b == 0), 255, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(mask_dir, f"{i:05d}.png"), inpaint_m)

    shutil.rmtree(frame_tmp, ignore_errors=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return n_frames


@app.function(
    gpu="H200",
    image=propainter_sam2_image,
    timeout=3600,
    volumes={"/sam_checkpoints": sam_checkpoints_vol},
)
def generate_player_aware_mask_video(video_bytes: bytes) -> bytes:
    """Generate a player-aware MELBOURNE inpaint mask video via SAM2.

    Returns mask video bytes (for use with run_propainter / run_diffueraser).
    Mask = MELBOURNE_quad AND NOT (player_silhouette dilated 18px).
    """
    import os
    import shutil
    import subprocess
    import tempfile

    import cv2

    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, "input.mp4")
    mask_dir = os.path.join(tmpdir, "masks")
    os.makedirs(mask_dir)
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    quad_corners = [[649, 840], [1268, 840], [1268, 1038], [649, 1038]]
    n_frames = _generate_player_aware_masks(
        video_path, mask_dir, quad_corners, player_dilate_px=35
    )

    # Get input fps
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    # Encode mask frames as video
    mask_video_path = os.path.join(tmpdir, "mask.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", f"{fps:.6f}",
            "-i", os.path.join(mask_dir, "%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-r", f"{fps:.6f}",
            mask_video_path,
        ],
        check=True,
        capture_output=True,
    )
    print(f"Encoded {n_frames} mask frames at {fps} fps")

    with open(mask_video_path, "rb") as f:
        mask_bytes = f.read()
    shutil.rmtree(tmpdir, ignore_errors=True)
    return mask_bytes


@app.function(
    gpu="H200",
    image=propainter_sam2_image,
    timeout=3600,
    volumes={"/sam_checkpoints": sam_checkpoints_vol},
)
def run_propainter_player_aware(
    video_bytes: bytes, full_quality: bool = False
) -> dict:
    """ProPainter with PLAYER-AWARE inpaint mask.

    The mask passed to ProPainter excludes the player silhouette, so the
    model only inpaints the court behind/around the player. The player
    itself is preserved (untouched). Result: clean video with MELBOURNE
    text removed AND player intact, no smeary leg artifacts.
    """
    import os
    import shutil
    import subprocess
    import tempfile

    import cv2
    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}, full_quality={full_quality}")

    tmpdir = tempfile.mkdtemp()
    video_path = os.path.join(tmpdir, "input.mp4")
    mask_dir = os.path.join(tmpdir, "masks")
    os.makedirs(mask_dir)
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    # MELBOURNE quad (matches scripts/video_inpaint.py default)
    quad_corners = [
        [649, 840], [1268, 840], [1268, 1038], [649, 1038]
    ]

    # Generate per-frame player-aware inpaint masks.
    # 35px dilate + temporal_window=4: union of player_dilated_35px across
    # frames [N-4, N+4]. This covers the motion-blur TRAIL (player's past
    # positions) which extends way beyond per-frame 35px dilation.
    n_frames = _generate_player_aware_masks(
        video_path, mask_dir, quad_corners,
        player_dilate_px=35, temporal_window=4,
    )
    print(f"[player-aware] Generated {n_frames} player-aware masks")

    # Extract video frames as PNG (ProPainter prefers a directory)
    frame_dir = os.path.join(tmpdir, "frames")
    os.makedirs(frame_dir)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frame_dir, f"{i:05d}.png"), frame)
        i += 1
    cap.release()
    print(f"[player-aware] Extracted {i} video frames")

    if i != n_frames:
        print(f"WARNING: video frames ({i}) != mask frames ({n_frames})")

    # Run ProPainter
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = [
        "python", "inference_propainter.py",
        "--video", frame_dir,
        "--mask", mask_dir,
        "--fp16",
        "--save_fps", "59",
    ]
    if full_quality:
        args += ["--subvideo_length", "40"]
    else:
        args += [
            "--subvideo_length", "30",
            "--neighbor_length", "5",
            "--ref_stride", "20",
            "--resize_ratio", "0.5",
        ]
    result = subprocess.run(
        args, cwd="/opt/propainter", capture_output=True, text=True, env=env
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-1000:]}")

    # Find inpaint_out.mp4
    output_dir = "/opt/propainter/results"
    inpaint_path = None
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f == "inpaint_out.mp4":
                inpaint_path = os.path.join(root, f)
                break
    if inpaint_path is None:
        return {"error": "No inpaint_out.mp4 produced", "model": "propainter_player_aware"}
    with open(inpaint_path, "rb") as fh:
        return {"output_bytes": fh.read(), "model": "propainter_player_aware"}


@app.function(
    gpu="H200",
    image=diffueraser_image,
    timeout=7200,
    volumes={"/diffueraser_weights": diffueraser_weights_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_diffueraser(video_bytes: bytes, mask_video_bytes: bytes) -> dict:
    """DiffuEraser video inpainting (diffusion-based, refines ProPainter priori)."""
    import os
    import subprocess
    import tempfile
    import urllib.request

    import torch
    from huggingface_hub import snapshot_download

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    weights_root = "/diffueraser_weights"
    os.makedirs(weights_root, exist_ok=True)

    # Download HuggingFace weights once into the cache volume
    hf_repos = [
        ("lixiaowen/diffuEraser", "diffuEraser"),
        ("stable-diffusion-v1-5/stable-diffusion-v1-5", "stable-diffusion-v1-5"),
        ("stabilityai/sd-vae-ft-mse", "sd-vae-ft-mse"),
        ("wangfuyun/PCM_Weights", "PCM_Weights"),
    ]
    for repo_id, subdir in hf_repos:
        target = os.path.join(weights_root, subdir)
        # Check for any non-trivial content (snapshot_download writes a sentinel)
        if os.path.exists(target) and len(os.listdir(target)) > 1:
            print(f"  cached: {subdir}")
            continue
        print(f"  downloading {repo_id} -> {target}")
        snapshot_download(
            repo_id=repo_id, local_dir=target, token=os.environ.get("HF_TOKEN")
        )

    # ProPainter weights from GitHub releases
    propainter_dir = os.path.join(weights_root, "propainter")
    os.makedirs(propainter_dir, exist_ok=True)
    pp_files = ["ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth"]
    for fname in pp_files:
        dest = os.path.join(propainter_dir, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            continue
        url = f"https://github.com/sczhou/ProPainter/releases/download/v0.1.0/{fname}"
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, dest)

    diffueraser_weights_vol.commit()

    # Symlink cache into /opt/diffueraser/weights so the script finds them
    weights_link = "/opt/diffueraser/weights"
    if os.path.exists(weights_link) and not os.path.islink(weights_link):
        import shutil
        shutil.rmtree(weights_link)
    if not os.path.exists(weights_link):
        os.symlink(weights_root, weights_link)

    # Write input video bytes (mask we'll regenerate ourselves)
    tmpdir = tempfile.mkdtemp()
    raw_video_path = os.path.join(tmpdir, "raw_input.mp4")
    video_path = os.path.join(tmpdir, "input.mp4")
    mask_path = os.path.join(tmpdir, "mask.mp4")
    with open(raw_video_path, "wb") as f:
        f.write(video_bytes)
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # DiffuEraser does STRICT fps equality between video and mask. To
    # guarantee match, we re-encode BOTH at integer 60 fps CFR using ffmpeg.
    import cv2
    TARGET_FPS = 60
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", raw_video_path,
            "-r", str(TARGET_FPS), "-vsync", "cfr",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
            video_path,
        ],
        check=True, capture_output=True,
    )
    cap_v = cv2.VideoCapture(video_path)
    video_fps = float(cap_v.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap_v.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap_v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_v.release()
    print(f"Re-encoded input: {video_fps} fps, {n_frames} frames, {fw}x{fh}")

    # Build mask video at IDENTICAL 60 fps CFR using ffmpeg from PNG frames.
    import numpy as np
    mask_dir = os.path.join(tmpdir, "mask_frames")
    os.makedirs(mask_dir, exist_ok=True)
    quad = np.array(
        [[649, 840], [1268, 840], [1268, 1038], [649, 1038]], dtype=np.int32
    )
    mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillPoly(mask, [quad], 255)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.dilate(mask, kern, iterations=1)
    for i in range(n_frames):
        cv2.imwrite(os.path.join(mask_dir, f"{i:05d}.png"), mask)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(TARGET_FPS),
            "-i", os.path.join(mask_dir, "%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-r", str(TARGET_FPS),
            mask_path,
        ],
        check=True, capture_output=True,
    )
    cap_m = cv2.VideoCapture(mask_path)
    mask_fps_actual = float(cap_m.get(cv2.CAP_PROP_FPS))
    cap_m.release()
    print(f"Mask: {mask_fps_actual} fps")

    # Run DiffuEraser
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    result = subprocess.run(
        [
            "python", "run_diffueraser.py",
            "--input_video", video_path,
            "--input_mask", mask_path,
            "--video_length", str(n_frames),
            "--max_img_size", "960",
            "--save_path", output_dir,
        ],
        cwd="/opt/diffueraser",
        capture_output=True,
        text=True,
        env=env,
    )
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}")

    # Find DiffuEraser output (refined diffusion result)
    output_video = os.path.join(output_dir, "diffueraser_result.mp4")
    if os.path.exists(output_video):
        with open(output_video, "rb") as f:
            return {"output_bytes": f.read(), "model": "diffueraser"}
    # Fallback to priori
    priori = os.path.join(output_dir, "priori.mp4")
    if os.path.exists(priori):
        with open(priori, "rb") as f:
            return {"output_bytes": f.read(), "model": "diffueraser_priori"}
    return {"error": "No output produced", "model": "diffueraser"}


@app.local_entrypoint()
def main(model: str = "propainter"):
    import os

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Read video
    video_path = os.path.join(repo, "data", "melbourne-walking-over-logo.mov")
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    print(f"Video: {len(video_bytes) // 1024}KB")

    # MELBOURNE text quad corners (from config bbox [729, 920, 1188, 958] + expansion)
    # TL, TR, BR, BL
    quad_corners = [
        [649, 840], [1268, 840], [1268, 1038], [649, 1038]
    ]

    # Generate mask
    mask_video_bytes, fw, fh = _generate_text_mask(video_bytes, quad_corners)
    print(f"Mask: {len(mask_video_bytes) // 1024}KB, {fw}x{fh}")

    if model == "propainter":
        print("Running ProPainter (B200, full quality)...")
        result = run_propainter.remote(video_bytes, mask_video_bytes)
    elif model == "propainter_fast":
        print("Running ProPainter (H200, fast iteration)...")
        result = run_propainter_fast.remote(video_bytes, mask_video_bytes)
    elif model == "propainter_player_aware":
        print("Running ProPainter PLAYER-AWARE (H200)...")
        result = run_propainter_player_aware.remote(video_bytes, False)
    elif model == "propainter_player_aware_full":
        print("Running ProPainter PLAYER-AWARE (B200, full quality)...")
        result = run_propainter_player_aware.remote(video_bytes, True)
    elif model == "diffueraser":
        print("Running DiffuEraser (H200, diffusion-based)...")
        result = run_diffueraser.remote(video_bytes, mask_video_bytes)
    elif model == "diffueraser_player_aware":
        print("Step 1: Generating player-aware mask via SAM2...")
        pa_mask_bytes = generate_player_aware_mask_video.remote(video_bytes)
        print(f"Player-aware mask: {len(pa_mask_bytes) // 1024}KB")
        print("Step 2: Running DiffuEraser with player-aware mask...")
        result = run_diffueraser.remote(video_bytes, pa_mask_bytes)
    else:
        print(f"Unknown model: {model}")
        return

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    # Save output
    out_dir = os.path.join(repo, "data", f"clean_court_{model}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "clean.mp4")
    with open(out_path, "wb") as f:
        f.write(result["output_bytes"])
    print(f"Saved: {out_path} ({len(result['output_bytes']) // 1024}KB)")
