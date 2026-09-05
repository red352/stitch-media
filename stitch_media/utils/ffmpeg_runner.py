"""FFmpeg and FFprobe execution utilities with hardware acceleration detection."""

import json
import os
import re
import shutil
import subprocess
from typing import Callable, Dict, List, Optional, Tuple


class FFmpegError(Exception):
    """Raised when an FFmpeg or FFprobe command fails."""
    pass


def find_ffmpeg() -> str:
    """Locate the ffmpeg binary."""
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegError("FFmpeg binary not found on PATH. Please install FFmpeg.")
    return path


def find_ffprobe() -> str:
    """Locate the ffprobe binary."""
    path = shutil.which("ffprobe")
    if not path:
        raise FFmpegError("FFprobe binary not found on PATH. Please install FFmpeg (including ffprobe).")
    return path


def run_command(
    cmd: List[str],
    desc: Optional[str] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    total_duration: Optional[float] = None,
) -> Tuple[int, str, str]:
    """
    Run an FFmpeg command with optional progress monitoring.
    Returns (returncode, stdout, stderr).
    """
    # Ensure subprocess doesn't pop up a console window on Windows
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0x08000000

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        creationflags=creationflags,
        encoding="utf-8",
        errors="replace",
    )

    stdout_acc = []
    stderr_acc = []

    # Monitor stdout or stderr for progress
    time_regex = re.compile(r"time=(\d+):(\d+):(\d+\.?\d*)")

    for line in iter(process.stderr.readline, ""):
        stderr_acc.append(line)
        if progress_callback and total_duration and total_duration > 0:
            match = time_regex.search(line)
            if match:
                hours, minutes, seconds = match.groups()
                current_time = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                pct = min(1.0, current_time / total_duration)
                progress_callback(pct)

    stdout_rest, stderr_rest = process.communicate()
    if stdout_rest:
        stdout_acc.append(stdout_rest)
    if stderr_rest:
        stderr_acc.append(stderr_rest)

    stdout_str = "".join(stdout_acc)
    stderr_str = "".join(stderr_acc)

    if process.returncode != 0:
        error_sample = "\n".join(stderr_str.strip().splitlines()[-15:])
        raise FFmpegError(f"Command failed (exit code {process.returncode}): {' '.join(cmd)}\n{error_sample}")

    return process.returncode, stdout_str, stderr_str


def probe_file_json(filepath: str) -> dict:
    """Run ffprobe on a file and return parsed JSON."""
    ffprobe = find_ffprobe()
    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        filepath,
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0x08000000

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return json.loads(proc.stdout)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Failed to probe file '{filepath}': {e.stderr}")
    except json.JSONDecodeError as e:
        raise FFmpegError(f"Failed to parse ffprobe output for '{filepath}': {e}")


_HWACCEL_CACHE: Optional[str] = None


def detect_best_video_encoder(prefer_hw: bool = True) -> str:
    """
    Detect the best available video encoder.
    Checks for NVENC (NVIDIA), QSV (Intel), VideoToolbox (Apple), or AMF (AMD).
    Falls back to software libx264.
    """
    global _HWACCEL_CACHE
    if not prefer_hw:
        return "libx264"

    if _HWACCEL_CACHE is not None:
        return _HWACCEL_CACHE

    ffmpeg = find_ffmpeg()
    cmd = [ffmpeg, "-v", "quiet", "-encoders"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        encoders_text = proc.stdout

        candidates = [
            ("h264_nvenc", "h264_nvenc"),
            ("h264_videotoolbox", "h264_videotoolbox"),
            ("h264_qsv", "h264_qsv"),
            ("h264_amf", "h264_amf"),
        ]

        for enc_name, actual in candidates:
            if enc_name in encoders_text:
                # Test if encoder actually works on current system (e.g. GPU drivers present)
                test_cmd = [
                    ffmpeg,
                    "-v", "error",
                    "-f", "lavfi",
                    "-i", "testsrc=duration=0.1:size=64x64:rate=30",
                    "-c:v", actual,
                    "-f", "null",
                    "-",
                ]
                test_proc = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if test_proc.returncode == 0:
                    _HWACCEL_CACHE = actual
                    return actual

        _HWACCEL_CACHE = "libx264"
        return "libx264"
    except Exception:
        _HWACCEL_CACHE = "libx264"
        return "libx264"
