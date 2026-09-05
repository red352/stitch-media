"""Media splitting engine: Manifest reverse reconstruction and autonomous split."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
import subprocess
from typing import Callable, List, Optional

from stitch_media.core.manifest import StitchManifest, SegmentInfo
from stitch_media.core.probe import probe_media, MediaProperties
from stitch_media.utils.ffmpeg_runner import find_ffmpeg, run_command, FFmpegError
from stitch_media.utils.logger import info, success, warning


class SplitMode(str, Enum):
    MANIFEST = "manifest"
    SCENE = "scene"
    SILENCE = "silence"
    DURATION = "duration"


@dataclass
class SplitConfig:
    """Configuration for media splitting."""
    input_path: Path
    output_dir: Path
    mode: SplitMode = SplitMode.MANIFEST
    manifest_path: Optional[Path] = None
    restore_overlap: bool = True       # In manifest mode, reconstruct the original overlap from previous segment
    chunk_duration: float = 60.0       # Used for duration-based split
    overlap_duration: float = 0.0      # Overlap window between chunks in duration mode
    scene_threshold: float = 0.35      # 0.0 - 1.0 scene change sensitivity
    silence_noise_db: float = -30.0    # dB threshold for silence detection
    silence_min_dur: float = 0.5       # Minimum duration of silence in seconds
    progress_callback: Optional[Callable[[float], None]] = None


class MediaSplitter:
    """Splits media via reverse manifest or autonomous scene/silence/duration detection."""

    def __init__(self, config: SplitConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def split(self) -> List[Path]:
        """Execute splitting based on configured mode."""
        if self.config.mode == SplitMode.MANIFEST:
            return self._split_by_manifest()
        elif self.config.mode == SplitMode.SCENE:
            return self._split_by_scene()
        elif self.config.mode == SplitMode.SILENCE:
            return self._split_by_silence()
        elif self.config.mode == SplitMode.DURATION:
            return self._split_by_duration()
        else:
            raise ValueError(f"Unknown split mode: {self.config.mode}")

    def _split_by_manifest(self) -> List[Path]:
        """Reverse reconstruction: restore original constituent clips from manifest."""
        manifest_file = self.config.manifest_path
        if not manifest_file or not manifest_file.exists():
            # Try finding a sibling .manifest.json
            candidate = self.config.input_path.with_suffix(".manifest.json")
            if candidate.exists():
                manifest_file = candidate
            else:
                raise FileNotFoundError(f"Manifest file not found for: {self.config.input_path}")

        manifest = StitchManifest.from_json(manifest_file)
        created_files: List[Path] = []
        ffmpeg = find_ffmpeg()

        total_segs = len(manifest.segments)
        for i, seg in enumerate(manifest.segments):
            # Calculate source extraction window
            if self.config.restore_overlap:
                start_sec = max(0.0, seg.output_start - seg.trim_start)
                end_sec = seg.output_end
            else:
                start_sec = seg.output_start
                end_sec = seg.output_end

            duration_sec = max(0.01, end_sec - start_sec)
            out_filename = f"part_{seg.segment_id + 1:03d}_{seg.source_filename}"
            out_file = self.config.output_dir / out_filename

            cmd = [
                ffmpeg, "-y",
                "-ss", f"{start_sec:.3f}",
                "-i", str(self.config.input_path),
                "-t", f"{duration_sec:.3f}",
                "-c", "copy",  # Fast stream copy
                str(out_file),
            ]

            try:
                run_command(cmd, desc=f"Extracting {out_filename}")
            except FFmpegError:
                # If stream copy fails on keyframe boundary, re-encode cleanly
                cmd_reencode = [
                    ffmpeg, "-y",
                    "-ss", f"{start_sec:.3f}",
                    "-i", str(self.config.input_path),
                    "-t", f"{duration_sec:.3f}",
                    "-c:v", "libx264", "-crf", "22", "-preset", "fast",
                    "-c:a", "aac", "-b:a", "192k",
                    str(out_file),
                ]
                run_command(cmd_reencode, desc=f"Re-encoding {out_filename}")

            created_files.append(out_file)
            if self.config.progress_callback:
                self.config.progress_callback((i + 1) / total_segs)

        return created_files

    def _split_by_scene(self) -> List[Path]:
        """Detect scene cut boundaries and split video into shots."""
        ffmpeg = find_ffmpeg()
        prop = probe_media(self.config.input_path)
        if not prop.has_video:
            raise FFmpegError("Scene cut detection requires a video stream.")

        # Run scene detection filter
        thresh = self.config.scene_threshold
        cmd = [
            ffmpeg,
            "-i", str(self.config.input_path),
            "-vf", f"select='gt(scene,{thresh:.2f})',metadata=print:file=-",
            "-f", "null",
            "-",
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
        output = proc.stdout + "\n" + proc.stderr

        # Extract pts_time
        cut_points = [0.0]
        pts_regex = re.compile(r"pts_time:([0-9]+\.?[0-9]*)")
        for match in pts_regex.finditer(output):
            t = float(match.group(1))
            if t - cut_points[-1] >= 1.0:  # Avoid micro-cuts less than 1 sec apart
                cut_points.append(t)

        if prop.duration > cut_points[-1] + 0.5:
            cut_points.append(prop.duration)

        return self._cut_intervals(cut_points, prefix="scene")

    def _split_by_silence(self) -> List[Path]:
        """Detect silence intervals and split audio/video into active segments."""
        ffmpeg = find_ffmpeg()
        prop = probe_media(self.config.input_path)
        if not prop.has_audio:
            raise FFmpegError("Silence detection requires an audio stream.")

        noise_db = self.config.silence_noise_db
        min_dur = self.config.silence_min_dur
        cmd = [
            ffmpeg,
            "-i", str(self.config.input_path),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
            "-f", "null",
            "-",
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
        output = proc.stderr

        # Parse silence_start and silence_end
        silence_starts = [float(m.group(1)) for m in re.finditer(r"silence_start:\s*([0-9]+\.?[0-9]*)", output)]
        silence_ends = [float(m.group(1)) for m in re.finditer(r"silence_end:\s*([0-9]+\.?[0-9]*)", output)]

        # Determine cut midpoints in silences
        cut_points = [0.0]
        for s_start, s_end in zip(silence_starts, silence_ends):
            mid = (s_start + s_end) / 2.0
            if mid - cut_points[-1] >= 1.0:
                cut_points.append(mid)

        if prop.duration > cut_points[-1] + 0.5:
            cut_points.append(prop.duration)

        return self._cut_intervals(cut_points, prefix="speech")

    def _split_by_duration(self) -> List[Path]:
        """Split video into fixed duration chunks, optionally with overlap."""
        prop = probe_media(self.config.input_path)
        total_dur = prop.duration
        chunk_dur = self.config.chunk_duration
        overlap = self.config.overlap_duration

        if chunk_dur <= 0:
            raise ValueError("Chunk duration must be greater than 0.")
        if overlap >= chunk_dur:
            raise ValueError("Overlap duration must be strictly less than chunk duration.")

        step = chunk_dur - overlap
        intervals: List[tuple[float, float]] = []
        curr = 0.0

        while curr < total_dur:
            end = min(total_dur, curr + chunk_dur)
            intervals.append((curr, end))
            if end >= total_dur:
                break
            curr += step

        ffmpeg = find_ffmpeg()
        ext = self.config.input_path.suffix or ".mp4"
        created_files: List[Path] = []

        for i, (t_start, t_end) in enumerate(intervals, 1):
            dur = t_end - t_start
            out_file = self.config.output_dir / f"chunk_{i:03d}{ext}"

            cmd = [
                ffmpeg, "-y",
                "-ss", f"{t_start:.3f}",
                "-i", str(self.config.input_path),
                "-t", f"{dur:.3f}",
                "-c:v", "libx264", "-crf", "22", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                str(out_file),
            ]
            run_command(cmd, desc=f"Creating {out_file.name}")
            created_files.append(out_file)

        return created_files

    def _cut_intervals(self, cut_points: List[float], prefix: str = "clip") -> List[Path]:
        """Helper to cut media between a sequence of timestamp cut points."""
        ffmpeg = find_ffmpeg()
        ext = self.config.input_path.suffix or ".mp4"
        created_files: List[Path] = []

        for i in range(len(cut_points) - 1):
            t_start = cut_points[i]
            t_end = cut_points[i + 1]
            dur = t_end - t_start
            if dur < 0.2:
                continue

            out_file = self.config.output_dir / f"{prefix}_{i + 1:03d}{ext}"
            cmd = [
                ffmpeg, "-y",
                "-ss", f"{t_start:.3f}",
                "-i", str(self.config.input_path),
                "-t", f"{dur:.3f}",
                "-c:v", "libx264", "-crf", "22", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                str(out_file),
            ]
            run_command(cmd, desc=f"Extracting {out_file.name}")
            created_files.append(out_file)

        return created_files
