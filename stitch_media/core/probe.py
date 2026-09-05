"""Media probing and stream inspection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from stitch_media.utils.ffmpeg_runner import probe_file_json, FFmpegError


@dataclass
class MediaProperties:
    """Attributes and stream info of a media file."""
    filepath: Path
    duration: float
    has_video: bool
    has_audio: bool
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    bitrate: Optional[int] = None


def probe_media(filepath: Path | str) -> MediaProperties:
    """
    Extract technical properties from a video or audio file.
    Raises FFmpegError if probe fails or no valid stream found.
    """
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Media file not found: {path}")

    data = probe_file_json(str(path))
    fmt = data.get("format", {})
    streams = data.get("streams", [])

    duration_str = fmt.get("duration")
    duration = float(duration_str) if duration_str else 0.0

    has_video = False
    has_audio = False
    width = None
    height = None
    fps = None
    sample_rate = None
    channels = None
    video_codec = None
    audio_codec = None

    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "video" and not has_video:
            # Skip image attachments (e.g. album art)
            if stream.get("disposition", {}).get("attached_pic", 0) == 1:
                continue
            has_video = True
            video_codec = stream.get("codec_name")
            width = stream.get("width")
            height = stream.get("height")

            # Parse frame rate
            r_frame_rate = stream.get("r_frame_rate", "0/0")
            if "/" in r_frame_rate:
                num, den = r_frame_rate.split("/")
                if float(den) > 0:
                    fps = float(num) / float(den)

            if duration == 0.0 and stream.get("duration"):
                duration = float(stream["duration"])

        elif codec_type == "audio" and not has_audio:
            has_audio = True
            audio_codec = stream.get("codec_name")
            if stream.get("sample_rate"):
                sample_rate = int(stream["sample_rate"])
            if stream.get("channels"):
                channels = int(stream["channels"])
            if duration == 0.0 and stream.get("duration"):
                duration = float(stream["duration"])

    if not has_video and not has_audio:
        raise FFmpegError(f"No audio or video streams detected in: {path}")

    bitrate = int(fmt["bit_rate"]) if fmt.get("bit_rate") else None

    return MediaProperties(
        filepath=path,
        duration=duration,
        has_video=has_video,
        has_audio=has_audio,
        width=width,
        height=height,
        fps=fps,
        sample_rate=sample_rate,
        channels=channels,
        video_codec=video_codec,
        audio_codec=audio_codec,
        bitrate=bitrate,
    )
