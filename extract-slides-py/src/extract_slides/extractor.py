"""Core extraction logic for extract-slides."""

from pathlib import Path
from typing import TypedDict


class ExtractionResult(TypedDict):
    """Result of slide extraction."""

    count: int
    output_dir: str
    format: str


def extract_slides(
    input_file: str,
    output_dir: str,
    output_format: str,
    verbose: bool = False,
) -> ExtractionResult:
    """Extract slides from a presentation file.

    Args:
        input_file: Path to the input presentation file
        output_dir: Directory to save extracted slides
        output_format: Format for output files (png, jpg, pdf)
        verbose: Enable verbose output

    Returns:
        ExtractionResult containing extraction details

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If output format is not supported
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    supported_formats = {"png", "jpg", "pdf"}
    if output_format.lower() not in supported_formats:
        raise ValueError(
            f"Unsupported format: {output_format}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Processing file: {input_path}")
        print(f"Output directory: {output_path}")
        print(f"Output format: {output_format}")

    # TODO: Implement actual slide extraction logic
    # For now, this is a placeholder that demonstrates the structure
    slide_count = 0

    return ExtractionResult(
        count=slide_count,
        output_dir=str(output_path),
        format=output_format,
    )
