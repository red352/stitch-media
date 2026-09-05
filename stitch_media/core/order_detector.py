"""Multimodal order detection and alignment planning with boundary window optimization."""

from dataclasses import dataclass, field
import itertools
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from stitch_media.core.probe import probe_media, MediaProperties
from stitch_media.core.audio_aligner import extract_pcm_audio, align_audio_pair
from stitch_media.core.visual_aligner import extract_sampled_frames, align_visual_pair
from stitch_media.utils.logger import info, warning


@dataclass
class ClipAlignmentStep:
    """Alignment detail between clip[i] and clip[i+1]."""
    clip_a: Path
    clip_b: Path
    overlap_seconds: float
    gap_seconds: float
    confidence: float
    method: str  # 'audio', 'visual', 'filename_timecode', 'fallback'


@dataclass
class AlignmentPlan:
    """Overall plan describing chronological sequence and transition metrics."""
    ordered_clips: List[Path]
    clip_properties: Dict[Path, MediaProperties]
    steps: List[ClipAlignmentStep] = field(default_factory=list)


def parse_filename_timecodes(name: str) -> Optional[Tuple[float, float]]:
    """
    Parse start and end timestamps from standardized filenames like:
    国语中字_无水印_可投屏_00_00_00_000_01_02_08_492_seg1.mp4 -> (0.0, 3728.492)
    """
    match = re.search(
        r"(\d{2})_(\d{2})_(\d{2})_(\d{3})_(\d{2})_(\d{2})_(\d{2})_(\d{3})",
        name
    )
    if match:
        h1, m1, s1, ms1, h2, m2, s2, ms2 = [int(x) for x in match.groups()]
        t1 = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
        t2 = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
        return t1, t2
    return None


def analyze_and_order_clips(
    clip_paths: List[Path | str],
    force_order: bool = False,
    min_confidence: float = 0.35,
    boundary_window_sec: float = 120.0,
) -> AlignmentPlan:
    """
    Analyze all clips, determine chronological sequence, and compute transition metrics.
    Automatically optimizes long videos (> 180s) using boundary sliding windows and filename heuristics.
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

    # 1. Check if all filenames have valid continuous timecodes
    timecodes: Dict[Path, Optional[Tuple[float, float]]] = {}
    has_all_timecodes = True
    for p in paths:
        tc = parse_filename_timecodes(p.name)
        timecodes[p] = tc
        if tc is None:
            has_all_timecodes = False

    if has_all_timecodes and not force_order:
        # Sort by start timestamp
        sorted_by_tc = sorted(paths, key=lambda p: timecodes[p][0])
        steps = []
        is_contiguous_chain = True

        for k in range(len(sorted_by_tc) - 1):
            ca, cb = sorted_by_tc[k], sorted_by_tc[k + 1]
            end_a = timecodes[ca][1]
            start_b = timecodes[cb][0]

            # If end_a matches start_b within 0.1s
            if abs(end_a - start_b) <= 0.1:
                steps.append(ClipAlignmentStep(
                    clip_a=ca,
                    clip_b=cb,
                    overlap_seconds=0.0,
                    gap_seconds=0.0,
                    confidence=1.0,
                    method="filename_timecode",
                ))
            elif start_b < end_a:
                # Overlap indicated by timecodes
                overlap = end_a - start_b
                steps.append(ClipAlignmentStep(
                    clip_a=ca,
                    clip_b=cb,
                    overlap_seconds=overlap,
                    gap_seconds=0.0,
                    confidence=0.99,
                    method="filename_timecode",
                ))
            else:
                # Gap indicated by timecodes
                gap = start_b - end_a
                steps.append(ClipAlignmentStep(
                    clip_a=ca,
                    clip_b=cb,
                    overlap_seconds=0.0,
                    gap_seconds=gap,
                    confidence=0.99,
                    method="filename_timecode",
                ))

        info("Detected continuous timeline sequence directly from filename timecodes.")
        return AlignmentPlan(ordered_clips=sorted_by_tc, clip_properties=props, steps=steps)

    # 2. Content-based alignment
    # If media is long (> 180s), extract boundary windows instead of full duration
    def get_audio_segment(p: Path, is_tail: bool) -> np.ndarray:
        if not props[p].has_audio:
            return np.array([], dtype=np.float32)

        dur = props[p].duration
        if dur > boundary_window_sec * 1.5:
            if is_tail:
                start_sec = max(0.0, dur - boundary_window_sec)
                dur_sec = boundary_window_sec
            else:
                start_sec = 0.0
                dur_sec = boundary_window_sec
            pcm, _ = extract_pcm_audio(p, sample_rate=8000, start_sec=start_sec, duration_sec=dur_sec)
            return pcm
        else:
            pcm, _ = extract_pcm_audio(p, sample_rate=8000)
            return pcm

    # Cache boundary audio
    head_audio_cache: Dict[Path, np.ndarray] = {}
    tail_audio_cache: Dict[Path, np.ndarray] = {}

    for p in paths:
        if props[p].has_audio:
            tail_audio_cache[p] = get_audio_segment(p, is_tail=True)
            head_audio_cache[p] = get_audio_segment(p, is_tail=False)

    def compute_pairwise(p1: Path, p2: Path) -> Tuple[bool, float, float, str]:
        """Check if p1 precedes p2 (tail of p1 overlaps head of p2)."""
        aud1_tail = tail_audio_cache.get(p1, np.array([]))
        aud2_head = head_audio_cache.get(p2, np.array([]))

        if len(aud1_tail) > 0 and len(aud2_head) > 0:
            res = align_audio_pair(aud1_tail, aud2_head, sample_rate=8000)
            if res.confidence >= min_confidence and res.overlap_seconds > 0.05 and res.first_clip_is_a:
                return (True, res.overlap_seconds, res.confidence, "audio")

        # Also test reverse: tail of p2 overlaps head of p1
        aud2_tail = tail_audio_cache.get(p2, np.array([]))
        aud1_head = head_audio_cache.get(p1, np.array([]))

        if len(aud2_tail) > 0 and len(aud1_head) > 0:
            res_rev = align_audio_pair(aud2_tail, aud1_head, sample_rate=8000)
            if res_rev.confidence >= min_confidence and res_rev.overlap_seconds > 0.05 and res_rev.first_clip_is_a:
                return (False, res_rev.overlap_seconds, res_rev.confidence, "audio")

        # Fallback to visual matching on short/silent files
        if props[p1].has_video and props[p2].has_video and max(props[p1].duration, props[p2].duration) < 300:
            h1, t1 = extract_sampled_frames(p1, fps=2.0)
            h2, t2 = extract_sampled_frames(p2, fps=2.0)
            if len(h1) > 0 and len(h2) > 0:
                vres = align_visual_pair(h1, t1, h2, t2)
                if vres.confidence >= min_confidence and vres.overlap_seconds > 0.1:
                    return (vres.first_clip_is_a, vres.overlap_seconds, vres.confidence, "visual")

        return (True, 0.0, 0.0, "none")

    if force_order:
        ordered = list(paths)
    else:
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
            ordered = list(paths)

    steps: List[ClipAlignmentStep] = []
    for k in range(len(ordered) - 1):
        ca, cb = ordered[k], ordered[k + 1]
        a_first, overlap, conf, method = compute_pairwise(ca, cb)

        steps.append(ClipAlignmentStep(
            clip_a=ca,
            clip_b=cb,
            overlap_seconds=overlap if a_first else 0.0,
            gap_seconds=0.0,
            confidence=conf,
            method=method if (a_first and overlap > 0) else "fallback",
        ))

    return AlignmentPlan(
        ordered_clips=ordered,
        clip_properties=props,
        steps=steps,
    )
