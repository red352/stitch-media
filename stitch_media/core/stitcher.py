"""Media stitching engine with overlap micro-trimming and gap compensation."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional
import shutil

from stitch_media.core.manifest import StitchManifest, SegmentInfo
from stitch_media.core.order_detector import AlignmentPlan, analyze_and_order_clips
from stitch_media.core.probe import MediaProperties
from stitch_media.utils.ffmpeg_runner import (
    find_ffmpeg,
    run_command,
    detect_best_video_encoder,
    FFmpegError,
)
from stitch_media.utils.logger import info, success, warning


class GapStrategy(str, Enum):
    PAD = "pad"         # Insert black frames and silence to preserve real-time timeline
    SMOOTH = "smooth"   # Seamless join with audio micro-fade and warning
    CUT = "cut"         # Hard join directly at cut point


@dataclass
class StitchConfig:
    """Configuration for stitching media files."""
    output_path: Path
    gap_strategy: GapStrategy = GapStrategy.PAD
    crossfade_duration: float = 0.05  # 50ms audio micro-crossfade to prevent clicks
    generate_manifest: bool = True
    manifest_path: Optional[Path] = None
    hardware_accel: bool = True
    crf: int = 22
    dry_run: bool = False
    progress_callback: Optional[Callable[[float], None]] = None


class MediaStitcher:
    """Orchestrates alignment analysis, filter graph construction, and rendering."""

    def __init__(self, config: StitchConfig):
        self.config = config

    def stitch(self, input_paths: List[Path | str], force_order: bool = False) -> StitchManifest:
        """
        Execute full stitching pipeline:
        1. Multimodal alignment & ordering
        2. Overlap & gap timeline computation
        3. Build and execute FFmpeg filtergraph
        4. Emit manifest.json
        """
        plan: AlignmentPlan = analyze_and_order_clips(input_paths, force_order=force_order)
        ordered_clips = plan.ordered_clips
        n = len(ordered_clips)

        if n == 0:
            raise ValueError("No input clips to stitch.")

        # Compute trim points and timeline offsets
        segments: List[SegmentInfo] = []
        current_timeline_time = 0.0

        for i, clip_path in enumerate(ordered_clips):
            prop = plan.clip_properties[clip_path]
            orig_dur = prop.duration

            # Trim leading overlap if this is not the first clip
            leading_overlap = 0.0
            if i > 0 and i - 1 < len(plan.steps):
                prev_step = plan.steps[i - 1]
                leading_overlap = prev_step.overlap_seconds

            trim_start = min(leading_overlap, orig_dur)
            trim_end = orig_dur
            trimmed_dur = max(0.0, trim_end - trim_start)

            overlap_with_next = 0.0
            gap_to_next = 0.0
            if i < len(plan.steps):
                overlap_with_next = plan.steps[i].overlap_seconds
                gap_to_next = plan.steps[i].gap_seconds

            seg_out_start = current_timeline_time
            seg_out_end = seg_out_start + trimmed_dur

            segments.append(SegmentInfo(
                segment_id=i,
                source_path=str(clip_path),
                source_filename=clip_path.name,
                original_duration=orig_dur,
                trim_start=trim_start,
                trim_end=trim_end,
                trimmed_duration=trimmed_dur,
                output_start=seg_out_start,
                output_end=seg_out_end,
                overlap_with_next=overlap_with_next,
                gap_to_next=gap_to_next,
            ))

            current_timeline_time = seg_out_end

            # Account for gap if pad strategy is enabled
            if gap_to_next > 0 and self.config.gap_strategy == GapStrategy.PAD:
                current_timeline_time += gap_to_next

        manifest = StitchManifest(
            output_filename=self.config.output_path.name,
            total_duration=current_timeline_time,
            gap_strategy=self.config.gap_strategy.value,
            segments=segments,
        )

        manifest_file = self.config.manifest_path or self.config.output_path.with_suffix(".manifest.json")

        if self.config.dry_run:
            if self.config.generate_manifest:
                manifest.to_json(manifest_file)
            return manifest

        # Execute FFmpeg rendering
        self._render_stitch(plan, segments, current_timeline_time)

        if self.config.generate_manifest:
            manifest.to_json(manifest_file)
            success(f"Stitch manifest saved to: {manifest_file}")

        return manifest

    def _render_stitch(
        self,
        plan: AlignmentPlan,
        segments: List[SegmentInfo],
        total_duration: float,
    ) -> None:
        """Construct FFmpeg filtergraph and render output."""
        ffmpeg = find_ffmpeg()
        ordered = plan.ordered_clips
        n = len(ordered)

        # Probe master dimensions and framerate from the first video clip
        ref_w = 1920
        ref_h = 1080
        ref_fps = 30.0
        has_any_video = False
        has_any_audio = False

        for p in ordered:
            prop = plan.clip_properties[p]
            if prop.has_video and not has_any_video:
                ref_w = prop.width or 1920
                ref_h = prop.height or 1080
                ref_fps = prop.fps or 30.0
                has_any_video = True
            if prop.has_audio:
                has_any_audio = True

        # Build FFmpeg command inputs
        cmd = [ffmpeg, "-y"]
        for p in ordered:
            cmd.extend(["-i", str(p)])

        # Construct Filter Complex
        filter_complex_parts = []
        concat_v_inputs = []
        concat_a_inputs = []

        for i, seg in enumerate(segments):
            p = ordered[i]
            prop = plan.clip_properties[p]
            t_in = seg.trim_start
            t_out = seg.trim_end

            # Video filter branch
            if has_any_video:
                if prop.has_video:
                    v_filter = (
                        f"[{i}:v]trim=start={t_in:.3f}:end={t_out:.3f},"
                        f"setpts=PTS-STARTPTS,"
                        f"scale={ref_w}:{ref_h}:force_original_aspect_ratio=decrease,"
                        f"pad={ref_w}:{ref_h}:(ow-iw)/2:(oh-ih)/2,"
                        f"fps={ref_fps:.2f},format=yuv420p[v{i}]"
                    )
                else:
                    # Synthetic black video for audio-only clip in a video stitch
                    v_filter = (
                        f"color=c=black:s={ref_w}x{ref_h}:r={ref_fps:.2f}:d={seg.trimmed_duration:.3f},"
                        f"format=yuv420p[v{i}]"
                    )
                filter_complex_parts.append(v_filter)
                concat_v_inputs.append(f"[v{i}]")

            # Audio filter branch
            if has_any_audio:
                if prop.has_audio:
                    fade_dur = self.config.crossfade_duration
                    # Audio trim with subtle micro fade-in and fade-out to prevent clicks
                    dur = seg.trimmed_duration
                    a_fade = ""
                    if dur > 2 * fade_dur:
                        a_fade = f",afade=t=in:st=0:d={fade_dur:.3f},afade=t=out:st={dur - fade_dur:.3f}:d={fade_dur:.3f}"

                    a_filter = (
                        f"[{i}:a]atrim=start={t_in:.3f}:end={t_out:.3f},"
                        f"asetpts=PTS-STARTPTS{a_fade},"
                        f"aformat=sample_rates=48000:channel_layouts=stereo[a{i}]"
                    )
                else:
                    # Synthetic silence
                    a_filter = (
                        f"anullsrc=r=48000:cl=stereo:d={seg.trimmed_duration:.3f}[a{i}]"
                    )
                filter_complex_parts.append(a_filter)
                concat_a_inputs.append(f"[a{i}]")

            # Handle Gap Padding between segments if configured
            if seg.gap_to_next > 0 and self.config.gap_strategy == GapStrategy.PAD:
                gap_dur = seg.gap_to_next
                gap_v_tag = f"[gap_v_{i}]"
                gap_a_tag = f"[gap_a_{i}]"

                if has_any_video:
                    gap_v = (
                        f"color=c=black:s={ref_w}x{ref_h}:r={ref_fps:.2f}:d={gap_dur:.3f},"
                        f"format=yuv420p{gap_v_tag}"
                    )
                    filter_complex_parts.append(gap_v)
                    concat_v_inputs.append(gap_v_tag)

                if has_any_audio:
                    gap_a = f"anullsrc=r=48000:cl=stereo:d={gap_dur:.3f}{gap_a_tag}"
                    filter_complex_parts.append(gap_a)
                    concat_a_inputs.append(gap_a_tag)

        # Final concat filter
        num_concat = len(concat_v_inputs) if has_any_video else len(concat_a_inputs)
        concat_str = ""
        for k in range(num_concat):
            if has_any_video:
                concat_str += concat_v_inputs[k]
            if has_any_audio:
                concat_str += concat_a_inputs[k]

        v_flag = 1 if has_any_video else 0
        a_flag = 1 if has_any_audio else 0
        concat_str += f"concat=n={num_concat}:v={v_flag}:a={a_flag}"
        if has_any_video:
            concat_str += "[outv]"
        if has_any_audio:
            concat_str += "[outa]"
        filter_complex_parts.append(concat_str)

        cmd.extend(["-filter_complex", ";".join(filter_complex_parts)])

        # Map streams
        if has_any_video:
            cmd.extend(["-map", "[outv]"])
            encoder = detect_best_video_encoder(self.config.hardware_accel)
            cmd.extend(["-c:v", encoder])
            if encoder == "libx264":
                cmd.extend(["-crf", str(self.config.crf), "-preset", "medium"])
            elif encoder == "h264_nvenc":
                cmd.extend(["-cq", str(self.config.crf), "-preset", "p4"])
        else:
            cmd.extend(["-vn"])

        if has_any_audio:
            cmd.extend(["-map", "[outa]"])
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            cmd.extend(["-an"])

        # Create target parent directory if needed
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.append(str(self.config.output_path))

        run_command(
            cmd,
            desc="Stitching media",
            progress_callback=self.config.progress_callback,
            total_duration=total_duration,
        )
