"""Manifest data models for recording and reversing media stitches."""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import List, Optional


@dataclass
class SegmentInfo:
    """Detailed metadata for a single stitched segment."""
    segment_id: int
    source_path: str
    source_filename: str
    original_duration: float
    trim_start: float
    trim_end: float
    trimmed_duration: float
    output_start: float
    output_end: float
    overlap_with_next: float = 0.0
    gap_to_next: float = 0.0


@dataclass
class StitchManifest:
    """Full manifest representing a completed or planned stitch."""
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    output_filename: str = ""
    total_duration: float = 0.0
    gap_strategy: str = "pad"
    segments: List[SegmentInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert manifest to Python dict."""
        return asdict(self)

    def to_json(self, path: Path | str) -> None:
        """Serialize manifest to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "StitchManifest":
        """Recreate StitchManifest from a dict."""
        segments_raw = data.get("segments", [])
        segments = [
            SegmentInfo(
                segment_id=s["segment_id"],
                source_path=s["source_path"],
                source_filename=s["source_filename"],
                original_duration=float(s["original_duration"]),
                trim_start=float(s["trim_start"]),
                trim_end=float(s["trim_end"]),
                trimmed_duration=float(s["trimmed_duration"]),
                output_start=float(s["output_start"]),
                output_end=float(s["output_end"]),
                overlap_with_next=float(s.get("overlap_with_next", 0.0)),
                gap_to_next=float(s.get("gap_to_next", 0.0)),
            )
            for s in segments_raw
        ]
        return cls(
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", ""),
            output_filename=data.get("output_filename", ""),
            total_duration=float(data.get("total_duration", 0.0)),
            gap_strategy=data.get("gap_strategy", "pad"),
            segments=segments,
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "StitchManifest":
        """Load StitchManifest from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
