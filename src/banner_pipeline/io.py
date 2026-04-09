"""Frame extraction and video I/O utilities."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def load_frame(path: str, frame_idx: int = 0) -> np.ndarray:
    """Load a single frame from an image file or video.

    Tries ``cv2.imread`` first; falls back to ffmpeg extraction for videos.
    """
    if frame_idx == 0:
        img = cv2.imread(path)
        if img is not None:
            return img

    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        cmd = [
            "ffmpeg",
            "-i",
            path,
            "-vf",
            f"select=eq(n\\,{frame_idx})",
            "-vframes",
            "1",
            "-q:v",
            "2",
            tmp,
            "-y",
            "-loglevel",
            "error",
        ]
        subprocess.run(cmd, check=True)
        frame = cv2.imread(tmp)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    if frame is None:
        raise RuntimeError(f"Could not load frame {frame_idx} from: {path}")
    return frame


def extract_all_frames(video_path: str, out_dir: str) -> list[str]:
    """Extract every frame as a numbered JPEG into *out_dir*.

    Returns a sorted list of filenames (not full paths).
    """
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-q:v",
        "2",
        "-start_number",
        "0",
        os.path.join(out_dir, "%05d.jpg"),
        "-y",
        "-loglevel",
        "error",
    ]
    subprocess.run(cmd, check=True)

    frame_names = sorted(
        [p for p in os.listdir(out_dir) if Path(p).suffix.lower() in {".jpg", ".jpeg"}],
        key=lambda p: int(Path(p).stem),
    )
    if not frame_names:
        raise RuntimeError(f"No frames extracted from: {video_path}")
    return frame_names


class StreamingVideoWriter:
    """Single-pass H.264 writer that pipes raw BGR frames to ffmpeg via stdin.

    Avoids the legacy two-step write (OpenCV mp4v → ffmpeg re-encode) and
    avoids holding the entire frame buffer in memory. Use as a context
    manager or call ``write()`` per frame and ``close()`` at the end.
    """

    def __init__(self, output_path: str, width: int, height: int, fps: float = 30.0) -> None:
        self.output_path = output_path
        os.makedirs(str(Path(output_path).parent), exist_ok=True)
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{width}x{height}",
                "-pix_fmt",
                "bgr24",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-loglevel",
                "error",
                output_path,
            ],
            stdin=subprocess.PIPE,
        )
        # subprocess.PIPE guarantees stdin is set, but mypy can't see that.
        assert self._proc.stdin is not None
        self._stdin = self._proc.stdin
        self._closed = False

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("StreamingVideoWriter is already closed")
        # tobytes() requires C-contiguous; cv2 outputs are usually contiguous
        # but we ensure it for in-place mutated slices.
        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr)
        self._stdin.write(frame_bgr.tobytes())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stdin.close()
            rc = self._proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg exited with code {rc}")
        except Exception:
            self._proc.kill()
            raise

    def __enter__(self) -> StreamingVideoWriter:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def write_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: float = 30.0,
) -> None:
    """Write a list of BGR frames as an H.264 MP4 video.

    Single-pass: pipes raw BGR frames to ffmpeg's libx264 encoder via stdin.
    For streaming use cases, prefer :class:`StreamingVideoWriter`.
    """
    if not frames:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    with StreamingVideoWriter(output_path, w, h, fps) as writer:
        for frame in frames:
            writer.write(frame)


def get_video_fps(video_path: str, default: float = 30.0) -> float:
    """Read the FPS from a video file via OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or np.isnan(fps) or fps < 1e-3:
        return default
    return fps
