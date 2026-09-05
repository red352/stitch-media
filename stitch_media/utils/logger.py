"""Console output and logging utilities using Rich with cross-platform encoding safety."""

import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)

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

console = Console(legacy_windows=False)
err_console = Console(stderr=True, legacy_windows=False)


def info(msg: str) -> None:
    """Print an informational message."""
    console.print(f"[bold cyan][*][/bold cyan] {msg}")


def success(msg: str) -> None:
    """Print a success message."""
    console.print(f"[bold green][+][/bold green] {msg}")


def warning(msg: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow][!][/bold yellow] {msg}")


def error(msg: str) -> None:
    """Print an error message."""
    err_console.print(f"[bold red][-][/bold red] {msg}")


def create_progress() -> Progress:
    """Create a standardized rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def print_manifest_summary(manifest_dict: dict) -> None:
    """Print formatted summary table of a stitch manifest."""
    table = Table(title="Stitch Manifest Summary", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Source Clip", style="cyan")
    table.add_column("Orig Dur", justify="right")
    table.add_column("Trim Range", justify="center")
    table.add_column("Cut Dur", justify="right")
    table.add_column("Overlap w/ Next", justify="right", style="green")
    table.add_column("Gap to Next", justify="right", style="yellow")

    segments = manifest_dict.get("segments", [])
    for idx, seg in enumerate(segments, 1):
        source = seg.get("source_filename", "")
        orig_dur = f"{seg.get('original_duration', 0.0):.2f}s"
        trim = f"{seg.get('trim_start', 0.0):.2f}s - {seg.get('trim_end', 0.0):.2f}s"
        cut_dur = f"{seg.get('trimmed_duration', 0.0):.2f}s"

        overlap = f"{seg.get('overlap_with_next', 0.0):.2f}s" if seg.get("overlap_with_next", 0.0) > 0 else "-"
        gap = f"{seg.get('gap_to_next', 0.0):.2f}s" if seg.get("gap_to_next", 0.0) > 0 else "-"

        table.add_row(str(idx), source, orig_dur, trim, cut_dur, overlap, gap)

    console.print(table)
    total_dur = manifest_dict.get("total_duration", 0.0)
    console.print(Panel(
        f"[bold]Total Output Duration:[/bold] {total_dur:.2f}s | "
        f"[bold]Segments Joined:[/bold] {len(segments)} | "
        f"[bold]Gap Strategy:[/bold] {manifest_dict.get('gap_strategy', 'pad')}",
        border_style="green"
    ))
