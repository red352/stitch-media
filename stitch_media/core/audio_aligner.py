"""Audio extraction and FFT-based cross-correlation alignment."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Optional, Tuple
import numpy as np
from scipy import signal

from stitch_media.utils.ffmpeg_runner import find_ffmpeg, FFmpegError


@dataclass
class AudioAlignmentResult:
    """Result of cross-correlating two audio signals."""
    offset_seconds: float       # t_B - t_A: positive if B started after A, negative if A started after B
    overlap_seconds: float      # Detected overlapping duration (0.0 if separated or gap)
    confidence: float           # Correlation coefficient between 0.0 and 1.0
    first_clip_is_a: bool       # True if A precedes B chronologically


def extract_pcm_audio(filepath: Path | str, sample_rate: int = 8000) -> Tuple[np.ndarray, int]:
    """
    Extract single-channel mono PCM float32 audio from a media file using FFmpeg.
    Returns (audio_array, sample_rate).
    """
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg,
        "-v", "quiet",
        "-i", str(filepath),
        "-vn",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-",
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        data = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
        if len(data) == 0:
            return np.array([], dtype=np.float32), sample_rate

        # Normalize to [-1.0, 1.0]
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
        return data, sample_rate
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Failed to extract audio from '{filepath}': {e.stderr}")


def align_audio_pair(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sample_rate: int = 8000,
    min_overlap_sec: float = 0.2,
    max_search_overlap_sec: Optional[float] = None,
) -> AudioAlignmentResult:
    """
    Align two audio tracks using normalized FFT cross-correlation.
    Detects if A overlaps B, or B overlaps A, and determines chronological ordering.
    """
    len_a = len(audio_a)
    len_b = len(audio_b)

    if len_a == 0 or len_b == 0:
        return AudioAlignmentResult(0.0, 0.0, 0.0, True)

    dur_a = len_a / sample_rate
    dur_b = len_b / sample_rate

    # Check signal activity (if essentially silent, correlation will be invalid)
    std_a = float(np.std(audio_a))
    std_b = float(np.std(audio_b))
    if std_a < 1e-4 or std_b < 1e-4:
        return AudioAlignmentResult(0.0, 0.0, 0.0, True)

    # Standardize signals
    sig_a = (audio_a - np.mean(audio_a)) / (std_a * np.sqrt(len_a))
    sig_b = (audio_b - np.mean(audio_b)) / (std_b * np.sqrt(len_b))

    # Fast Fourier Transform Cross-Correlation
    # cross_corr[k] corresponds to lag = k - (len_b - 1)
    # lag > 0: B is shifted forward in time (B starts after A)
    # lag < 0: A is shifted forward in time (A starts after B)
    corr = signal.fftconvolve(sig_a, sig_b[::-1], mode="full")
    lags = np.arange(-len_b + 1, len_a)

    # Restrict search range if max_search_overlap_sec is specified
    # An overlap between tail of A and head of B means lag is near len_a - overlap_len
    # An overlap between tail of B and head of A means lag is near -(len_b - overlap_len)
    min_overlap_samples = int(min_overlap_sec * sample_rate)

    # Filter out valid search indices where overlap is at least min_overlap_samples
    # Case 1: A precedes B (lag > 0): overlap is len_a - lag
    # Case 2: B precedes A (lag < 0): overlap is len_b - (-lag) = len_b + lag
    valid_mask = np.zeros_like(corr, dtype=bool)

    # A precedes B
    mask_a_first = (lags >= 0) & ((len_a - lags) >= min_overlap_samples)
    # B precedes A
    mask_b_first = (lags < 0) & ((len_b + lags) >= min_overlap_samples)
    valid_mask = mask_a_first | mask_b_first

    if not np.any(valid_mask):
        return AudioAlignmentResult(0.0, 0.0, 0.0, True)

    corr_filtered = np.where(valid_mask, corr, -np.inf)
    peak_idx = int(np.argmax(corr_filtered))
    peak_val = float(corr[peak_idx])
    best_lag = lags[peak_idx]

    # Parabolic sub-sample peak refinement
    precise_lag = float(best_lag)
    if 0 < peak_idx < len(corr) - 1:
        y0, y1, y2 = corr[peak_idx - 1], corr[peak_idx], corr[peak_idx + 1]
        denom = 2 * (2 * y1 - y0 - y2)
        if denom > 1e-9:
            delta = (y2 - y0) / denom
            if abs(delta) < 1.0:
                precise_lag += delta

    offset_seconds = precise_lag / sample_rate

    # Calculate actual overlap duration
    if offset_seconds >= 0:
        first_clip_is_a = True
        overlap_seconds = max(0.0, dur_a - offset_seconds)
    else:
        first_clip_is_a = False
        overlap_seconds = max(0.0, dur_b - (-offset_seconds))

    # Normalize peak value to local Pearson correlation on overlapping window
    overlap_samples = int(overlap_seconds * sample_rate)
    if overlap_samples > 0:
        local_scale = np.sqrt(len_a * len_b) / overlap_samples
        confidence = float(np.clip(peak_val * local_scale, 0.0, 1.0))
    else:
        confidence = 0.0

    return AudioAlignmentResult(
        offset_seconds=offset_seconds,
        overlap_seconds=overlap_seconds,
        confidence=confidence,
        first_clip_is_a=first_clip_is_a,
    )
