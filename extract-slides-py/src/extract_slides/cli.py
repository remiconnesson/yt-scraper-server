"""CLI interface for extract-slides."""

import click

from extract_slides import __version__
from extract_slides.extractor import extract_slides
from extract_slides.video_analyzer import AnalysisConfig, analyze_video
from extract_slides.video_output import generate_summary, save_analysis_results


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Extract slides from presentations.

    A command-line tool for extracting slides from various presentation formats.
    """
    pass


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="output",
    help="Output directory for extracted slides",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["png", "jpg", "pdf"], case_sensitive=False),
    default="png",
    help="Output format for slides",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def extract(input_file: str, output_dir: str, format: str, verbose: bool) -> None:
    """Extract slides from a presentation file.

    INPUT_FILE: Path to the presentation file to extract slides from
    """
    if verbose:
        click.echo(f"Extracting slides from: {input_file}")
        click.echo(f"Output directory: {output_dir}")
        click.echo(f"Output format: {format}")

    try:
        result = extract_slides(input_file, output_dir, format, verbose)
        click.echo(f"✓ Successfully extracted {result['count']} slides to {output_dir}")
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort() from e


@main.command()
def info() -> None:
    """Display information about the tool."""
    click.echo(f"extract-slides version {__version__}")
    click.echo("A CLI tool for extracting slides from presentations")


def _run_video_analysis(
    input_file: str,
    output_dir: str,
    grid_cols: int,
    grid_rows: int,
    cell_hash_threshold: int,
    min_static_cell_ratio: float,
    min_static_frames: int,
    verbose: bool,
) -> None:
    """Shared implementation for video analysis commands.

    Args:
        input_file: Path to the input video file
        output_dir: Directory where images and markdown report are written
        grid_cols: Number of cells along the horizontal axis
        grid_rows: Number of cells along the vertical axis
        cell_hash_threshold: Maximum hash distance for cells to be considered unchanged
        min_static_cell_ratio: Minimum ratio of unchanged cells for a frame to be static
        min_static_frames: Minimum number of consecutive static frames required
        verbose: Enable verbose output

    Raises:
        click.Abort: If validation fails or an error occurs during analysis
    """
    if verbose:
        click.echo(f"Analyzing video: {input_file}")
        click.echo(f"Output directory: {output_dir}")
        click.echo(f"Grid: {grid_cols}x{grid_rows}")
        click.echo(f"Cell hash threshold: {cell_hash_threshold}")
        click.echo(f"Min static cell ratio: {min_static_cell_ratio}")
        click.echo(f"Min static frames: {min_static_frames}")

    # Validate parameters
    if grid_cols <= 0 or grid_rows <= 0:
        click.echo("✗ Error: Grid dimensions must be positive", err=True)
        raise click.Abort()

    if not 0 <= min_static_cell_ratio <= 1:
        click.echo("✗ Error: min-static-cell-ratio must be between 0 and 1", err=True)
        raise click.Abort()

    if min_static_frames < 1:
        click.echo("✗ Error: min-static-frames must be at least 1", err=True)
        raise click.Abort()

    try:
        # Create config
        config: AnalysisConfig = {
            "grid_cols": grid_cols,
            "grid_rows": grid_rows,
            "cell_hash_threshold": cell_hash_threshold,
            "min_static_cell_ratio": min_static_cell_ratio,
            "min_static_frames": min_static_frames,
        }

        # Analyze video
        if verbose:
            click.echo("Extracting and analyzing frames...")

        result = analyze_video(input_file, config)

        if verbose:
            click.echo("Saving results...")

        # Save results
        save_analysis_results(result, output_dir)

        # Print summary
        click.echo("✓ Analysis complete!")
        click.echo("")
        summary = generate_summary(result)
        click.echo(summary)
        click.echo("")
        click.echo(f"Results saved to: {output_dir}")
        click.echo(f"  - Images: {output_dir}/static/")
        click.echo(f"  - Report: {output_dir}/report.md")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        if verbose:
            raise
        raise click.Abort() from e


@main.command(name="analyze-video")
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to the input video file",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory where images and markdown report are written",
)
@click.option(
    "--grid-cols",
    type=int,
    default=4,
    help="Number of cells along the horizontal axis",
)
@click.option(
    "--grid-rows",
    type=int,
    default=4,
    help="Number of cells along the vertical axis",
)
@click.option(
    "--cell-hash-threshold",
    type=int,
    default=5,
    help="Maximum hash distance for two cells to be considered unchanged",
)
@click.option(
    "--min-static-cell-ratio",
    type=float,
    default=0.8,
    help="Minimum ratio of unchanged cells (0-1) for a frame to be static",
)
@click.option(
    "--min-static-frames",
    type=int,
    default=3,
    help="Minimum number of consecutive static frames required",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def analyze_video_command(
    input_file: str,
    output_dir: str,
    grid_cols: int,
    grid_rows: int,
    cell_hash_threshold: int,
    min_static_cell_ratio: float,
    min_static_frames: int,
    verbose: bool,
) -> None:
    """Analyze video to detect static segments using grid-based image hashing.

    This command extracts frames at 1 fps, compares them using a grid-based
    perceptual hash approach, and identifies static segments and moving sections.
    """
    _run_video_analysis(
        input_file,
        output_dir,
        grid_cols,
        grid_rows,
        cell_hash_threshold,
        min_static_cell_ratio,
        min_static_frames,
        verbose,
    )


@click.command()
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to the input video file",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory where images and markdown report are written",
)
@click.option(
    "--grid-cols",
    type=int,
    default=4,
    help="Number of cells along the horizontal axis",
)
@click.option(
    "--grid-rows",
    type=int,
    default=4,
    help="Number of cells along the vertical axis",
)
@click.option(
    "--cell-hash-threshold",
    type=int,
    default=5,
    help="Maximum hash distance for two cells to be considered unchanged",
)
@click.option(
    "--min-static-cell-ratio",
    type=float,
    default=0.8,
    help="Minimum ratio of unchanged cells (0-1) for a frame to be static",
)
@click.option(
    "--min-static-frames",
    type=int,
    default=3,
    help="Minimum number of consecutive static frames required",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def video_main(
    input_file: str,
    output_dir: str,
    grid_cols: int,
    grid_rows: int,
    cell_hash_threshold: int,
    min_static_cell_ratio: float,
    min_static_frames: int,
    verbose: bool,
) -> None:
    """Analyze video to detect static segments using grid-based image hashing.

    This is the standalone entry point for the video_static_segments command.
    """
    _run_video_analysis(
        input_file,
        output_dir,
        grid_cols,
        grid_rows,
        cell_hash_threshold,
        min_static_cell_ratio,
        min_static_frames,
        verbose,
    )


if __name__ == "__main__":
    main()
