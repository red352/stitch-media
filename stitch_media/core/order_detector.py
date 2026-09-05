"""Multimodal order detection and alignment planning for media sequences."""

from dataclasses import dataclass, field
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from stitch_media.core.probe import probe_media, MediaProperties
from stitch_media.core.audio_aligner import extract_pcm_audio, align_audio_pair, AudioAlignmentResult
from stitch_media.core.visual_aligner import extract_sampled_frames, align_visual_pair, VisualAlignmentResult
from stitch_media.utils.logger import info, warning


@dataclass
class ClipAlignmentStep:
    """Alignment detail between clip[i] and clip[i+1]."""
    clip_a: Path
    clip_b: Path
    overlap_seconds: float
    gap_seconds: float
    confidence: float
    method: str  # 'audio', 'visual', 'metadata_fallback'


@dataclass
class AlignmentPlan:
    """Overall plan describing chronological sequence and transition metrics."""
    ordered_clips: List[Path]
    clip_properties: Dict[Path, MediaProperties]
    steps: List[ClipAlignmentStep] = field(default_factory=list)


def analyze_and_order_clips(
    clip_paths: List[Path | str],
    force_order: bool = False,
    min_confidence: float = 0.35,
) -> AlignmentPlan:
    """
    Analyze all clips, determine their chronological sequence using multimodal alignment,
    and compute precise overlaps/gaps between consecutive clips.
    """
    paths = [Path(p).resolve() for p in clip_paths]
    n = len(paths)
    if n == 0:
        raise ValueError("No input media files provided.")

    props: Dict[Path, MediaProperties] = {}
    for p in paths:
        props[p] = probe_media(p)

    if n == 1:
        return AlignmentPlan(ordered_clips=paths, clip_properties=props, steps=[])

    # Extract audio signals for all clips that contain audio
    audio_cache: Dict[Path, np.ndarray] = {}
    for p in paths:
        if props[p].has_audio:
            try:
                pcm, _ = extract_pcm_audio(p, sample_rate=8000)
                audio_cache[p] = pcm
            except Exception as e:
                warning(f"Audio extraction failed for {p.name}: {e}")
                audio_cache[p] = np.array([], dtype=np.float32)
        else:
            audio_cache[p] = np.array([], dtype=np.float32)

    # Visual hashes cache (populated lazily if needed)
    visual_cache: Dict[Path, Tuple[list, list]] = {}

    def get_visual(p: Path):
        if p not in visual_cache:
            if props[p].has_video:
                hashes, tstamps = extract_sampled_frames(p, fps=2.0)
                visual_cache[p] = (hashes, tstamps)
            else:
                visual_cache[p] = ([], [])
        return visual_cache[p]

    def compute_pairwise(p1: Path, p2: Path) -> Tuple[bool, float, float, str]:
        """
        Returns (p1_precedes_p2, overlap_sec, confidence, method).
        """
        # Try Audio first
        aud1 = audio_cache.get(p1, np.array([]))
        aud2 = audio_cache.get(p2, np.array([]))
        if len(aud1) > 0 and len(aud2) > 0:
            res = align_audio_pair(aud1, aud2, sample_rate=8000)
            if res.confidence >= min_confidence and res.overlap_seconds > 0.05:
                return (res.first_clip_is_a, res.overlap_seconds, res.confidence, "audio")

        # Fallback to Visual matching
        if props[p1].has_video and props[p2].has_video:
            h1, t1 = get_visual(p1)
            h2, t2 = get_visual(p2)
            if len(h1) > 0 and len(h2) > 0:
                vres = align_visual_pair(h1, t1, h2, t2)
                if vres.confidence >= min_confidence and vres.overlap_seconds > 0.1:
                    return (vres.first_clip_is_a, vres.overlap_seconds, vres.confidence, "visual")

        return (True, 0.0, 0.0, "none")

    # If force_order is specified, preserve user's provided list order
    if force_order:
        ordered = list(paths)
    else:
        # Pairwise precedence scoring
        # score_matrix[i][j] > 0 indicates paths[i] precedes paths[j] with that score
        score_matrix = np.zeros((n, n), dtype=np.float32)
        overlap_matrix = np.zeros((n, n), dtype=np.float32)
        method_matrix: Dict[Tuple[int, int], str] = {}

        for i, j in itertools.combinations(range(n), 2):
            p_i, p_j = paths[i], paths[j]
            i_first, overlap, conf, method = compute_pairwise(p_i, p_j)
            if conf >= min_confidence and overlap > 0:
                if i_first:
                    score_matrix[i, j] = conf * (1.0 + overlap)
                    overlap_matrix[i, j] = overlap
                    method_matrix[(i, j)] = method
                else:
                    score_matrix[j, i] = conf * (1.0 + overlap)
                    overlap_matrix[j, i] = overlap
                    method_matrix[(j, i)] = method

        # Find the best Hamiltonian path (permutation of indices) that maximizes score
        best_perm = None
        best_score = -1.0

        for perm in itertools.permutations(range(n)):
            perm_score = 0.0
            for k in range(n - 1):
                u, v = perm[k], perm[k + 1]
                perm_score += score_matrix[u, v]
            if perm_score > best_score:
                best_score = perm_score
                best_perm = perm

        if best_score > 0 and best_perm is not None:
            ordered = [paths[idx] for idx in best_perm]
        else:
            # Fallback to user sequence if no strong overlap chain was detected
            ordered = list(paths)

    # Now calculate step-by-step transition metrics for the final sequence
    steps: List[ClipAlignmentStep] = []
    for k in range(len(ordered) - 1):
        ca, cb = ordered[k], ordered[k + 1]
        a_first, overlap, conf, method = compute_pairwise(ca, cb)

        if not a_first and overlap > 0.05:
            # Note: in rare cases where sequence was forced or estimated with low confidence
            overlap = 0.0

        gap = 0.0  # Gaps between unlinked files default to 0.0 unless configured
        steps.append(ClipAlignmentStep(
            clip_a=ca,
            clip_b=cb,
            overlap_seconds=overlap if a_first else 0.0,
            gap_seconds=gap,
            confidence=conf,
            method=method if (a_first and overlap > 0) else "fallback",
        ))

    return AlignmentPlan(
        ordered_clips=ordered,
        clip_properties=props,
        steps=steps,
    )
