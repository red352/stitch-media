"""Pytest fixtures for generating synthetic audio and video test media using FFmpeg."""

import subprocess
import pytest
from pathlib import Path

from stitch_media.utils.ffmpeg_runner import find_ffmpeg


@pytest.fixture(scope="session")
def test_media_dir(tmp_path_factory) -> Path:
    """Session directory for test media."""
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="session")
def synthetic_video_pair(test_media_dir) -> tuple[Path, Path, float]:
    """
    Generate two video clips with a known 2.0-second overlap.
    Clip 1: 0s to 6s (6s total)
    Clip 2: 4s to 10s (6s total)
    Shared overlapping window: 4s to 6s (2.0s overlap)
    """
    ffmpeg = find_ffmpeg()
    master_path = test_media_dir / "master.mp4"
    clip1_path = test_media_dir / "clip1.mp4"
    clip2_path = test_media_dir / "clip2.mp4"

    # 1. Generate 10-second master video with varying chirp audio and test visual pattern
    cmd_master = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=30",
        "-f", "lavfi", "-i", "aevalsrc=sin(2*PI*(200+50*t)*t):d=10:s=44100",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(master_path),
    ]
    subprocess.run(cmd_master, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # 2. Slice Clip 1 (0s .. 6s)
    cmd_c1 = [
        ffmpeg, "-y",
        "-ss", "0", "-t", "6",
        "-i", str(master_path),
        "-c:v", "libx264", "-c:a", "aac",
        str(clip1_path),
    ]
    subprocess.run(cmd_c1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # 3. Slice Clip 2 (4s .. 10s) -> overlaps clip1 by 2s
    cmd_c2 = [
        ffmpeg, "-y",
        "-ss", "4", "-t", "6",
        "-i", str(master_path),
        "-c:v", "libx264", "-c:a", "aac",
        str(clip2_path),
    ]
    subprocess.run(cmd_c2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    expected_overlap = 2.0
    return clip1_path, clip2_path, expected_overlap
