"""Visual perceptual hashing and frame-based alignment for video clips."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
import imagehash

from stitch_media.utils.ffmpeg_runner import find_ffmpeg, FFmpegError


@dataclass
class VisualAlignmentResult:
    """Result of visual frame matching between two video clips."""
    offset_seconds: float
    overlap_seconds: float
    confidence: float
    first_clip_is_a: bool


def extract_sampled_frames(
    filepath: Path | str,
    fps: float = 2.0,
    frame_size: Tuple[int, int] = (64, 64),
) -> Tuple[List[imagehash.ImageHash], List[float]]:
    """
    Extract perceptual hashes (dHash) and timestamps for frames sampled at `fps`.
    """
    ffmpeg = find_ffmpeg()
    w, h = frame_size
    cmd = [
        ffmpeg,
        "-v", "quiet",
        "-i", str(filepath),
        "-vf", f"fps={fps},scale={w}:{h}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]

    frame_bytes_len = w * h * 3
    hashes: List[imagehash.ImageHash] = []
    timestamps: List[float] = []

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_idx = 0
        while True:
            raw = proc.stdout.read(frame_bytes_len)
            if not raw or len(raw) < frame_bytes_len:
                break
            img = Image.frombytes("RGB", (w, h), raw)
            # Use dhash (difference hash) which is robust to lighting and contrast shifts
            h_val = imagehash.dhash(img)
            hashes.append(h_val)
            timestamps.append(frame_idx / fps)
            frame_idx += 1

        proc.communicate()
        return hashes, timestamps
    except Exception as e:
        raise FFmpegError(f"Failed to extract sampled frames from '{filepath}': {e}")


def align_visual_pair(
    hashes_a: List[imagehash.ImageHash],
    timestamps_a: List[float],
    hashes_b: List[imagehash.ImageHash],
    timestamps_b: List[float],
    max_hamming_distance: int = 5,
    min_matching_frames: int = 2,
) -> VisualAlignmentResult:
    """
    Find visual overlap between two video clips by matching perceptual hash sequences.
    """
    n_a = len(hashes_a)
    n_b = len(hashes_b)

    if n_a == 0 or n_b == 0:
        return VisualAlignmentResult(0.0, 0.0, 0.0, True)

    dur_a = timestamps_a[-1] if timestamps_a else 0.0
    dur_b = timestamps_b[-1] if timestamps_b else 0.0

    # Test Direction 1: A precedes B (Tail of A matches Head of B)
    best_a_first_score = 0.0
    best_a_first_overlap = 0.0
    best_a_first_offset = 0.0

    # Search possible offsets where tail of A aligns with head of B
    for offset_idx in range(n_a):
        # A starts at 0, B starts at offset_idx in timestamps_a
        matched = 0
        total_compared = 0
        total_dist = 0

        for b_idx in range(n_b):
            a_idx = offset_idx + b_idx
            if a_idx >= n_a:
                break
            dist = hashes_a[a_idx] - hashes_b[b_idx]
            total_compared += 1
            total_dist += dist
            if dist <= max_hamming_distance:
                matched += 1

        if total_compared >= min_matching_frames:
            match_rate = matched / total_compared
            avg_dist = total_dist / total_compared
            # Score combines match rate and low average Hamming distance (0..64)
            score = match_rate * max(0.0, 1.0 - (avg_dist / 16.0))
            if score > best_a_first_score and match_rate >= 0.7:
                best_a_first_score = score
                best_a_first_offset = timestamps_a[offset_idx]
                best_a_first_overlap = max(0.0, dur_a - best_a_first_offset)

    # Test Direction 2: B precedes A (Tail of B matches Head of A)
    best_b_first_score = 0.0
    best_b_first_overlap = 0.0
    best_b_first_offset = 0.0

    for offset_idx in range(n_b):
        matched = 0
        total_compared = 0
        total_dist = 0

        for a_idx in range(n_a):
            b_idx = offset_idx + a_idx
            if b_idx >= n_b:
                break
            dist = hashes_b[b_idx] - hashes_a[a_idx]
            total_compared += 1
            total_dist += dist
            if dist <= max_hamming_distance:
                matched += 1

        if total_compared >= min_matching_frames:
            match_rate = matched / total_compared
            avg_dist = total_dist / total_compared
            score = match_rate * max(0.0, 1.0 - (avg_dist / 16.0))
            if score > best_b_first_score and match_rate >= 0.7:
                best_b_first_score = score
                best_b_first_offset = timestamps_b[offset_idx]
                best_b_first_overlap = max(0.0, dur_b - best_b_first_offset)

    if best_a_first_score >= best_b_first_score:
        return VisualAlignmentResult(
            offset_seconds=best_a_first_offset,
            overlap_seconds=best_a_first_overlap,
            confidence=best_a_first_score,
            first_clip_is_a=True,
        )
    else:
        return VisualAlignmentResult(
            offset_seconds=-best_b_first_offset,
            overlap_seconds=best_b_first_overlap,
            confidence=best_b_first_score,
            first_clip_is_a=False,
        )
