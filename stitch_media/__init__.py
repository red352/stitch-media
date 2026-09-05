"""Stitch-Media: Intelligent media stitch & split toolkit with multimodal adaptive alignment."""

__version__ = "0.1.0"
__author__ = "red352"

from stitch_media.core.manifest import StitchManifest, SegmentInfo
from stitch_media.core.stitcher import MediaStitcher, StitchConfig, GapStrategy, StreamCopyMode
from stitch_media.core.splitter import MediaSplitter, SplitMode

__all__ = [
    "__version__",
    "StitchManifest",
    "SegmentInfo",
    "MediaStitcher",
    "StitchConfig",
    "GapStrategy",
    "StreamCopyMode",
    "MediaSplitter",
    "SplitMode",
]
