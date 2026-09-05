"""Unit tests for the MediaStitcher engine."""

from pathlib import Path
import pytest

from stitch_media.core.stitcher import MediaStitcher, StitchConfig, GapStrategy
from stitch_media.core.probe import probe_media


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
