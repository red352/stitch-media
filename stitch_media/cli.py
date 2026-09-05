"""Stitch-Media Command Line Interface powered by Typer and Rich."""

import sys
from pathlib import Path
from typing import List, Optional
import typer
from rich.panel import Panel
from rich.table import Table

# Ensure UTF-8 output where possible on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from stitch_media import __version__
from stitch_media.core.manifest import StitchManifest
from stitch_media.core.order_detector import analyze_and_order_clips
from stitch_media.core.stitcher import MediaStitcher, StitchConfig, GapStrategy, StreamCopyMode
from stitch_media.core.splitter import MediaSplitter, SplitConfig, SplitMode
from stitch_media.utils.ffmpeg_runner import (
    find_ffmpeg,
    find_ffprobe,
    detect_best_video_encoder,
    FFmpegError,
)
from stitch_media.utils.logger import (
    console,
    info,
    success,
    warning,
    error,
    create_progress,
    print_manifest_summary,
)

app = typer.Typer(
    name="stitch-media",
    help="Intelligent media stitch & split toolkit with multimodal adaptive alignment.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command(name="join")
def join_command(
    inputs: List[Path] = typer.Argument(..., help="List of input media files to stitch"),
    output: Path = typer.Option(..., "-o", "--output", help="Output file path (e.g. out.mp4)"),
    gap_strategy: GapStrategy = typer.Option(
        GapStrategy.PAD,
        "--gap-strategy",
        "-g",
        help="Strategy for handling missing gaps: 'pad' (black+silence), 'smooth' (crossfade), 'cut' (hard cut)",
    ),
    force_order: bool = typer.Option(
        False,
        "--force-order",
        help="Preserve provided file order without auto-detecting chronological sequence",
    ),
    stream_copy: StreamCopyMode = typer.Option(
        StreamCopyMode.AUTO,
        "--stream-copy",
        "-c",
        help="Stream copy mode: 'auto' (fast concat if codecs match & no overlap), 'always' (force copy), 'never' (force re-encode)",
    ),
    boundary_window: float = typer.Option(
        120.0,
        "--boundary-window",
        help="Boundary sliding window duration in seconds for fast audio alignment on large files",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Analyze alignment, overlaps, and sequence without rendering final video",
    ),
    manifest: bool = typer.Option(
        True,
        "--manifest/--no-manifest",
        help="Export stitch manifest JSON file",
    ),
    manifest_path: Optional[Path] = typer.Option(
        None,
        "--manifest-path",
        help="Custom path for manifest JSON file (defaults to <output>.manifest.json)",
    ),
    hwaccel: bool = typer.Option(
        True,
        "--hwaccel/--no-hwaccel",
        help="Enable automatic GPU hardware acceleration if available (NVENC/QSV/VideoToolbox)",
    ),
    crf: int = typer.Option(
        22,
        "--crf",
        help="Constant Rate Factor for H.264 encoding quality (lower is higher quality, default 22)",
    ),
) -> None:
    """
    Stitch multiple media files together with automatic order detection,
    seamless overlap micro-trimming, and gap compensation.
    """
    if len(inputs) < 2 and not dry_run:
        error("At least two input media files are required to stitch.")
        raise typer.Exit(code=1)

    for p in inputs:
        if not p.exists():
            error(f"Input file not found: {p}")
            raise typer.Exit(code=1)

    console.print(Panel.fit(
        f"[bold cyan]Stitch-Media Join[/bold cyan]\n"
        f"Input clips: {len(inputs)} files\n"
        f"Output target: {output}\n"
        f"Gap strategy: {gap_strategy.value}\n"
        f"Stream copy: {stream_copy.value}\n"
        f"Boundary window: {boundary_window}s\n"
        f"Dry run: {dry_run}",
        border_style="cyan"
    ))

    info("Analyzing media clips and detecting chronological sequence...")

    with create_progress() as progress:
        task = progress.add_task("[cyan]Rendering video stitch...", total=100)

        def progress_update(pct: float):
            progress.update(task, completed=int(pct * 100))

        config = StitchConfig(
            output_path=output,
            gap_strategy=gap_strategy,
            stream_copy=stream_copy,
            boundary_window_sec=boundary_window,
            generate_manifest=manifest,
            manifest_path=manifest_path,
            hardware_accel=hwaccel,
            crf=crf,
            dry_run=dry_run,
            progress_callback=progress_update,
        )

        stitcher = MediaStitcher(config)
        try:
            res_manifest = stitcher.stitch(inputs, force_order=force_order)
            progress.update(task, completed=100)
        except Exception as e:
            error(f"Stitch operation failed: {e}")
            raise typer.Exit(code=1)

    print_manifest_summary(res_manifest.to_dict())

    if dry_run:
        info("[yellow]Dry-run completed. No media file was encoded.[/yellow]")
    else:
        success(f"Successfully stitched into [bold green]{output}[/bold green] ({res_manifest.total_duration:.2f}s)")


@app.command(name="split")
def split_command(
    input_file: Path = typer.Argument(..., help="Source media file to split"),
    output_dir: Path = typer.Option(Path("./split_output"), "-o", "--output-dir", help="Directory to save split clips"),
    mode: SplitMode = typer.Option(
        SplitMode.MANIFEST,
        "--mode",
        "-m",
        help="Split mode: 'manifest' (reverse restore), 'scene' (visual cuts), 'silence' (pauses), 'duration' (fixed chunks)",
    ),
    manifest: Optional[Path] = typer.Option(
        None,
        "--manifest",
        help="Path to manifest.json file (required for 'manifest' mode if not sibling of input)",
    ),
    restore_overlap: bool = typer.Option(
        True,
        "--restore-overlap/--no-restore-overlap",
        help="In manifest mode, restore the original overlapping portion of each clip",
    ),
    duration: float = typer.Option(
        60.0,
        "--duration",
        "-d",
        help="Chunk duration in seconds (for 'duration' mode)",
    ),
    overlap: float = typer.Option(
        0.0,
        "--overlap",
        help="Overlap window in seconds between consecutive chunks (for 'duration' mode)",
    ),
    scene_threshold: float = typer.Option(
        0.35,
        "--scene-threshold",
        help="Scene change sensitivity threshold [0.0 - 1.0] (for 'scene' mode)",
    ),
) -> None:
    """
    Split a media file back into constituent segments using manifest metadata,
    or autonomously using scene detection, silence detection, or fixed intervals.
    """
    if not input_file.exists():
        error(f"Input file not found: {input_file}")
        raise typer.Exit(code=1)

    console.print(Panel.fit(
        f"[bold magenta]Stitch-Media Split[/bold magenta]\n"
        f"Input file: {input_file}\n"
        f"Destination directory: {output_dir}\n"
        f"Split mode: {mode.value}",
        border_style="magenta"
    ))

    config = SplitConfig(
        input_path=input_file,
        output_dir=output_dir,
        mode=mode,
        manifest_path=manifest,
        restore_overlap=restore_overlap,
        chunk_duration=duration,
        overlap_duration=overlap,
        scene_threshold=scene_threshold,
    )

    splitter = MediaSplitter(config)
    try:
        results = splitter.split()
        success(f"Split complete! Generated {len(results)} clips in: [bold green]{output_dir}[/bold green]")
        for f in results:
            console.print(f"  * {f.name}")
    except Exception as e:
        error(f"Split operation failed: {e}")
        raise typer.Exit(code=1)


@app.command(name="inspect")
def inspect_command(
    inputs: List[Path] = typer.Argument(..., help="List of media files to analyze"),
    boundary_window: float = typer.Option(
        120.0,
        "--boundary-window",
        help="Boundary sliding window duration in seconds for fast audio alignment on large files",
    ),
) -> None:
    """
    Inspect a set of media files, reporting detected order, overlaps, gaps, and audio/video confidence.
    """
    for p in inputs:
        if not p.exists():
            error(f"File not found: {p}")
            raise typer.Exit(code=1)

    info(f"Analyzing {len(inputs)} media files...")
    plan = analyze_and_order_clips(inputs, boundary_window_sec=boundary_window)

    table = Table(title="Media Sequence & Alignment Analysis", show_header=True, header_style="bold blue")
    table.add_column("Order", justify="center", width=6)
    table.add_column("Filename", style="cyan")
    table.add_column("Duration", justify="right")
    table.add_column("Transition Method", justify="center")
    table.add_column("Overlap with Next", justify="right", style="green")
    table.add_column("Confidence", justify="right")

    for i, p in enumerate(plan.ordered_clips):
        prop = plan.clip_properties[p]
        dur = f"{prop.duration:.2f}s"
        method = "-"
        overlap = "-"
        conf = "-"

        if i < len(plan.steps):
            step = plan.steps[i]
            method = step.method
            overlap = f"{step.overlap_seconds:.2f}s" if step.overlap_seconds > 0 else "-"
            conf = f"{step.confidence * 100:.1f}%" if step.confidence > 0 else "-"

        table.add_row(str(i + 1), p.name, dur, method, overlap, conf)

    console.print(table)


@app.command(name="version")
def version_command() -> None:
    """Display version and detected FFmpeg environment capabilities."""
    try:
        ffmpeg = find_ffmpeg()
        ffprobe = find_ffprobe()
        encoder = detect_best_video_encoder(True)
        status_text = (
            f"[bold green]stitch-media[/bold green] v{__version__}\n"
            f"FFmpeg binary: {ffmpeg}\n"
            f"FFprobe binary: {ffprobe}\n"
            f"Optimal H.264 Encoder: [bold yellow]{encoder}[/bold yellow]"
        )
    except Exception as e:
        status_text = f"stitch-media v{__version__}\n[red]FFmpeg error: {e}[/red]"

    console.print(Panel.fit(status_text, border_style="cyan"))


if __name__ == "__main__":
    app()
