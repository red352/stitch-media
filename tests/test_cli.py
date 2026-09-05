"""End-to-end CLI command tests using Typer CliRunner."""

from typer.testing import CliRunner
from stitch_media.cli import app

runner = CliRunner()


def test_cli_version():
    """Test 'stitch-media version' command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "stitch-media v" in result.stdout
    assert "FFmpeg binary" in result.stdout


def test_cli_inspect(synthetic_video_pair):
    """Test 'stitch-media inspect' command."""
    clip1, clip2, _ = synthetic_video_pair
    result = runner.invoke(app, ["inspect", str(clip1), str(clip2)])
    assert result.exit_code == 0
    assert "Media Sequence & Alignment Analysis" in result.stdout
    assert clip1.name in result.stdout
    assert clip2.name in result.stdout


def test_cli_join_dry_run(synthetic_video_pair, tmp_path):
    """Test 'stitch-media join --dry-run' command."""
    clip1, clip2, _ = synthetic_video_pair
    out_file = tmp_path / "cli_dry.mp4"

    result = runner.invoke(app, [
        "join",
        str(clip1),
        str(clip2),
        "-o", str(out_file),
        "--dry-run",
    ])
    assert result.exit_code == 0
    assert "Dry-run completed" in result.stdout
    assert not out_file.exists()


def test_cli_join_stream_copy_options(synthetic_video_pair, tmp_path):
    """Test 'stitch-media join' with custom stream-copy and boundary-window options."""
    clip1, clip2, _ = synthetic_video_pair
    out_file = tmp_path / "cli_opt.mp4"

    result = runner.invoke(app, [
        "join",
        str(clip1),
        str(clip2),
        "-o", str(out_file),
        "--stream-copy", "auto",
        "--boundary-window", "60",
        "--dry-run",
    ])
    assert result.exit_code == 0
    assert "Stream copy: auto" in result.stdout
    assert "Boundary window: 60" in result.stdout
