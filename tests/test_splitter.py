"""Unit tests for the MediaSplitter engine."""

from pathlib import Path
import pytest

from stitch_media.core.stitcher import MediaStitcher, StitchConfig, GapStrategy
from stitch_media.core.splitter import MediaSplitter, SplitConfig, SplitMode
from stitch_media.core.probe import probe_media


def test_split_by_manifest_reconstruction(synthetic_video_pair, tmp_path):
    """Test full cycle: stitch 2 clips with overlap -> split by manifest -> restore clips."""
    clip1, clip2, expected_overlap = synthetic_video_pair
    stitched_file = tmp_path / "cycle_stitched.mp4"
    manifest_file = tmp_path / "cycle_stitched.manifest.json"

    # 1. Stitch
    stitch_cfg = StitchConfig(
        output_path=stitched_file,
        manifest_path=manifest_file,
        gap_strategy=GapStrategy.PAD,
        hardware_accel=False,
    )
    MediaStitcher(stitch_cfg).stitch([clip1, clip2])

    # 2. Reverse split using manifest
    split_dir = tmp_path / "split_restored"
    split_cfg = SplitConfig(
        input_path=stitched_file,
        output_dir=split_dir,
        mode=SplitMode.MANIFEST,
        manifest_path=manifest_file,
        restore_overlap=True,
    )
    splitter = MediaSplitter(split_cfg)
    restored_files = splitter.split()

    assert len(restored_files) == 2
    for f in restored_files:
        assert f.exists()
        props = probe_media(f)
        # Original clips were 6.0s each
        assert abs(props.duration - 6.0) < 0.6


def test_split_by_duration_with_overlap(synthetic_video_pair, tmp_path):
    """Test autonomous split by fixed duration with overlap window."""
    clip1, _, _ = synthetic_video_pair  # 6.0s clip
    split_dir = tmp_path / "chunks"

    split_cfg = SplitConfig(
        input_path=clip1,
        output_dir=split_dir,
        mode=SplitMode.DURATION,
        chunk_duration=3.0,
        overlap_duration=1.0,
    )
    splitter = MediaSplitter(split_cfg)
    chunks = splitter.split()

    # In a 6s clip with 3s chunks stepping by 2s:
    # chunk 1: 0..3s
    # chunk 2: 2..5s
    # chunk 3: 4..6s
    assert len(chunks) == 3
    for c in chunks:
        assert c.exists()
        p = probe_media(c)
        assert p.duration > 1.5
