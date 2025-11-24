"""CLI interface for extract-slides."""

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import click

from slides_extractor.extract_slides import __version__
from slides_extractor.extract_slides.video_analyzer import AnalysisConfig, analyze_video
from slides_extractor.extract_slides.video_output import (
    generate_summary,
    save_analysis_results,
)


T = TypeVar("T", bound=Callable[..., Any])


@dataclass
class VideoAnalysisConfig:
    """User-provided configuration for video static segment analysis."""

    input_file: str
    output_dir: str
    grid_cols: int = 4
    grid_rows: int = 4
    cell_hash_threshold: int = 5
    min_static_cell_ratio: float = 0.8
    min_static_frames: int = 3
    verbose: bool = False

    def to_analysis_config(self) -> AnalysisConfig:
        """Convert CLI config to the analyzer's expected mapping."""

        return {
            "grid_cols": self.grid_cols,
            "grid_rows": self.grid_rows,
            "cell_hash_threshold": self.cell_hash_threshold,
            "min_static_cell_ratio": self.min_static_cell_ratio,
            "min_static_frames": self.min_static_frames,
        }


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Extract slides from presentations.

    A command-line tool for extracting slides from various presentation formats.
    """
    pass


@main.command()
def info() -> None:
    """Display information about the tool."""
    click.echo(f"extract-slides version {__version__}")
    click.echo("A CLI tool for extracting slides from presentations")


def _video_analysis_options(command: T) -> T:
    """Apply shared options for video analysis commands."""

    shared_options = [
        click.option(
            "--input",
            "input_file",
            required=True,
            type=click.Path(exists=True),
            help="Path to the input video file",
        ),
        click.option(
            "--output-dir",
            required=True,
            type=click.Path(),
            help="Directory where images and markdown report are written",
        ),
        click.option(
            "--grid-cols",
            type=int,
            default=4,
            help="Number of cells along the horizontal axis",
        ),
        click.option(
            "--grid-rows",
            type=int,
            default=4,
            help="Number of cells along the vertical axis",
        ),
        click.option(
            "--cell-hash-threshold",
            type=int,
            default=5,
            help="Maximum hash distance for two cells to be considered unchanged",
        ),
        click.option(
            "--min-static-cell-ratio",
            type=float,
            default=0.8,
            help="Minimum ratio of unchanged cells (0-1) for a frame to be static",
        ),
        click.option(
            "--min-static-frames",
            type=int,
            default=3,
            help="Minimum number of consecutive static frames required",
        ),
        click.option(
            "--verbose",
            "-v",
            is_flag=True,
            help="Enable verbose output",
        ),
    ]

    for option in reversed(shared_options):
        command = option(command)
    return command


def _run_video_analysis(config: VideoAnalysisConfig) -> None:
    """Shared implementation for video analysis commands."""

    if config.verbose:
        click.echo(f"Analyzing video: {config.input_file}")
        click.echo(f"Output directory: {config.output_dir}")
        click.echo(f"Grid: {config.grid_cols}x{config.grid_rows}")
        click.echo(f"Cell hash threshold: {config.cell_hash_threshold}")
        click.echo(f"Min static cell ratio: {config.min_static_cell_ratio}")
        click.echo(f"Min static frames: {config.min_static_frames}")

    if config.grid_cols <= 0 or config.grid_rows <= 0:
        click.echo("✗ Error: Grid dimensions must be positive", err=True)
        raise click.Abort()

    if not 0 <= config.min_static_cell_ratio <= 1:
        click.echo("✗ Error: min-static-cell-ratio must be between 0 and 1", err=True)
        raise click.Abort()

    if config.min_static_frames < 1:
        click.echo("✗ Error: min-static-frames must be at least 1", err=True)
        raise click.Abort()

    try:
        analysis_config: AnalysisConfig = config.to_analysis_config()

        if config.verbose:
            click.echo("Extracting and analyzing frames...")

        result = analyze_video(config.input_file, analysis_config)

        if config.verbose:
            click.echo("Saving results...")

        save_analysis_results(result, config.output_dir)

        click.echo("✓ Analysis complete!")
        click.echo("")
        summary = generate_summary(result)
        click.echo(summary)
        click.echo("")
        click.echo(f"Results saved to: {config.output_dir}")
        click.echo(f"  - Images: {config.output_dir}/static/")
        click.echo(f"  - Report: {config.output_dir}/report.md")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        if config.verbose:
            raise
        raise click.Abort() from e


@main.command(name="analyze-video")
@_video_analysis_options
def analyze_video_command(**kwargs: Any) -> None:
    """Analyze video to detect static segments using grid-based image hashing.

    This command extracts frames at 1 fps, compares them using a grid-based
    perceptual hash approach, and identifies static segments and moving sections.
    """
    _run_video_analysis(VideoAnalysisConfig(**kwargs))


@click.command()
@_video_analysis_options
def video_main(**kwargs: Any) -> None:
    """Analyze video to detect static segments using grid-based image hashing.

    This is the standalone entry point for the video_static_segments command.
    """
    _run_video_analysis(VideoAnalysisConfig(**kwargs))


if __name__ == "__main__":
    main()
