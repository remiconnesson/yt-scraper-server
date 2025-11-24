import click
import sys
import logging
from slides_extractor.video_jobs import process_video_task

# Configure basic logging for CLI
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@click.group()
def cli():
    """Slides Extractor CLI"""
    pass


@cli.command()
@click.argument("video_id")
@click.option("--output-dir", help="Directory to save output to (implies --no-s3)")
@click.option(
    "--no-s3",
    is_flag=True,
    help="Disable S3 upload. Defaults to './output' if no --output-dir provided",
)
def process(video_id: str, output_dir: str, no_s3: bool):
    """Process a YouTube video by ID."""

    # Handle output logic
    local_dir = None
    if output_dir:
        local_dir = output_dir
    elif no_s3:
        local_dir = "./output"

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    click.echo(click.style(f"\nProcessing Video: {video_id}", fg="green", bold=True))
    click.echo(f"URL: {video_url}")
    if local_dir:
        click.echo(f"Output: Local ({local_dir})")
    else:
        click.echo("Output: S3 Upload")
    click.echo("=" * 50)

    try:
        process_video_task(video_url, video_id, local_output_dir=local_dir)
        click.echo(click.style("\nSuccess! Pipeline completed.", fg="green"))
    except Exception as e:
        click.echo(click.style(f"\nError: {e}", fg="red"), err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
