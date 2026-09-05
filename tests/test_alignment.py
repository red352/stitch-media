"""Unit tests for multimodal audio and visual alignment engines."""

import numpy as np
import pytest

from stitch_media.core.audio_aligner import (
    extract_pcm_audio,
    align_audio_pair,
)
from stitch_media.core.visual_aligner import (
    extract_sampled_frames,
    align_visual_pair,
)
from stitch_media.core.order_detector import analyze_and_order_clips


def test_audio_alignment_synthetic_pair(synthetic_video_pair):
    """Test audio alignment accurately detects the 2.0s overlap between clip1 and clip2."""
    clip1, clip2, expected_overlap = synthetic_video_pair

    pcm1, sr1 = extract_pcm_audio(clip1, sample_rate=8000)
    pcm2, sr2 = extract_pcm_audio(clip2, sample_rate=8000)

    assert len(pcm1) > 0
    assert len(pcm2) > 0
    assert sr1 == sr2 == 8000

    # Align clip1 and clip2
    res = align_audio_pair(pcm1, pcm2, sample_rate=8000)

    assert res.first_clip_is_a is True
    assert res.confidence > 0.4
    # Expected overlap is 2.0s, allow small boundary tolerance
    assert abs(res.overlap_seconds - expected_overlap) < 0.2


def test_audio_alignment_reversed_order(synthetic_video_pair):
    """Test audio alignment detects reversed order when clip2 is passed before clip1."""
    clip1, clip2, expected_overlap = synthetic_video_pair

    pcm1, _ = extract_pcm_audio(clip1, sample_rate=8000)
    pcm2, _ = extract_pcm_audio(clip2, sample_rate=8000)

    res_reversed = align_audio_pair(pcm2, pcm1, sample_rate=8000)

    # Since pcm2 comes after pcm1 chronologically, first_clip_is_a should be False
    assert res_reversed.first_clip_is_a is False
    assert res_reversed.confidence > 0.4
    assert abs(res_reversed.overlap_seconds - expected_overlap) < 0.2


def test_visual_alignment_synthetic_pair(synthetic_video_pair):
    """Test visual frame matching for video clips."""
    clip1, clip2, expected_overlap = synthetic_video_pair

    hashes1, t1 = extract_sampled_frames(clip1, fps=2.0)
    hashes2, t2 = extract_sampled_frames(clip2, fps=2.0)

    assert len(hashes1) > 0
    assert len(hashes2) > 0

    vres = align_visual_pair(hashes1, t1, hashes2, t2)
    assert vres.first_clip_is_a is True
    assert vres.confidence > 0.5


def test_order_detector_reorders_shuffled_clips(synthetic_video_pair):
    """Test that analyze_and_order_clips automatically sorts [clip2, clip1] into [clip1, clip2]."""
    clip1, clip2, expected_overlap = synthetic_video_pair

    # Pass in shuffled order
    shuffled = [clip2, clip1]
    plan = analyze_and_order_clips(shuffled)

    assert len(plan.ordered_clips) == 2
    assert plan.ordered_clips[0] == clip1
    assert plan.ordered_clips[1] == clip2
    assert len(plan.steps) == 1
    assert abs(plan.steps[0].overlap_seconds - expected_overlap) < 0.2
