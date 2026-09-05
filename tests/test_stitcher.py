"""Unit tests for the MediaStitcher engine."""

from pathlib import Path
import pytest

from stitch_media.core.stitcher import MediaStitcher, StitchConfig, GapStrategy, StreamCopyMode
from stitch_media.core.probe import probe_media
from stitch_media.utils.ffmpeg_runner import find_ffmpeg
import subprocess


def test_stitch_dry_run(synthetic_video_pair, tmp_path):
    """Test stitching dry-run generates accurate manifest without encoding output file."""
    clip1, clip2, expected_overlap = synthetic_video_pair
    out_file = tmp_path / "dry_out.mp4"

    config = StitchConfig(
        output_path=out_file,
        gap_strategy=GapStrategy.PAD,
        dry_run=True,
    )
    stitcher = MediaStitcher(config)
    manifest = stitcher.stitch([clip2, clip1])  # Shuffled input

    # In dry-run, output video file should not exist, but manifest is generated
    assert not out_file.exists()
    assert len(manifest.segments) == 2
    # Combined duration = 6.0s + (6.0s - 2.0s) = ~10.0s
    assert abs(manifest.total_duration - 10.0) < 0.3


def test_stitch_render_and_manifest(synthetic_video_pair, tmp_path):
    """Test actual video rendering, overlap elimination, and manifest creation."""
    clip1, clip2, expected_overlap = synthetic_video_pair
    out_file = tmp_path / "stitched.mp4"
    manifest_file = tmp_path / "stitched.manifest.json"

    config = StitchConfig(
        output_path=out_file,
        manifest_path=manifest_file,
        gap_strategy=GapStrategy.PAD,
        hardware_accel=False,  # Use standard libx264 for deterministic testing
        dry_run=False,
    )
    stitcher = MediaStitcher(config)
    manifest = stitcher.stitch([clip1, clip2])

    assert out_file.exists()
    assert manifest_file.exists()

    # Verify rendered video properties
    props = probe_media(out_file)
    assert props.has_video is True
    assert props.has_audio is True
    # 6s + (6s - 2s overlap) = ~10.0s
    assert abs(props.duration - 10.0) < 0.5


def test_stitch_stream_copy_contiguous_clips(test_media_dir, tmp_path):
    """Test fast stream copy on contiguous identical-codec segments."""
    ffmpeg = find_ffmpeg()
    master_path = test_media_dir / "master.mp4"
    seg1 = tmp_path / "video_00_00_00_000_00_00_03_000_seg1.mp4"
    seg2 = tmp_path / "video_00_00_03_000_00_00_07_000_seg2.mp4"

    # Slice seg1: 0 to 3s
    subprocess.run([
        ffmpeg, "-y", "-ss", "0", "-t", "3",
        "-i", str(master_path), "-c", "copy", str(seg1)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # Slice seg2: 3 to 7s
    subprocess.run([
        ffmpeg, "-y", "-ss", "3", "-t", "4",
        "-i", str(master_path), "-c", "copy", str(seg2)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    out_file = tmp_path / "stream_copy_out.mp4"
    config = StitchConfig(
        output_path=out_file,
        stream_copy=StreamCopyMode.AUTO,
    )
    stitcher = MediaStitcher(config)
    manifest = stitcher.stitch([seg2, seg1])  # Shuffled input should be sorted by timecode

    assert out_file.exists()
    props = probe_media(out_file)
    assert abs(props.duration - 7.0) < 0.5
    assert len(manifest.segments) == 2
    assert manifest.segments[0].source_filename == seg1.name
    assert manifest.segments[1].source_filename == seg2.name
