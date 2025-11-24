"""Output generation for video analysis results."""

from pathlib import Path

import cv2

from extract_slides.video_analyzer import AnalysisResult, Segment


def format_timestamp(seconds: float) -> str:
    """Format timestamp in seconds to HH:MM:SS format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string in HH:MM:SS format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamp_for_filename(seconds: float) -> str:
    """Format timestamp for use in filenames (using hyphens instead of colons).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string in HH-MM-SS format
    """
    return format_timestamp(seconds).replace(":", "-")


def save_analysis_results(result: AnalysisResult, output_dir: str) -> None:
    """Save analysis results to output directory.

    Creates:
    - static/ directory with representative images for each static segment
    - report.md markdown file with analysis summary

    Args:
        result: AnalysisResult from video analysis
        output_dir: Directory to save output files

    Raises:
        OSError: If directories cannot be created or files cannot be written
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    static_dir = output_path / "static"
    static_dir.mkdir(exist_ok=True)

    # Save static segment images
    image_paths: dict[int, str] = {}
    for idx, segment in enumerate(result.static_segments, start=1):
        if segment.representative_frame is None:
            continue

        # Format filename
        start_formatted = format_timestamp_for_filename(segment.start_time)
        end_formatted = format_timestamp_for_filename(segment.end_time)
        filename = f"segment_{idx:03d}_{start_formatted}_to_{end_formatted}.png"

        # Convert RGB back to BGR for OpenCV
        bgr_frame = cv2.cvtColor(segment.representative_frame, cv2.COLOR_RGB2BGR)

        image_path = static_dir / filename
        success = cv2.imwrite(str(image_path), bgr_frame)
        if not success:
            raise OSError(f"Failed to write image file: {image_path}")

        # Store relative path for markdown
        image_paths[idx] = f"static/{filename}"

    # Generate markdown report
    report_path = output_path / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Video Analysis Report\n\n")

        # Combine segments with their types for chronological ordering
        timeline_items: list[tuple[float, Segment]] = [
            (seg.start_time, seg) for seg in result.segments
        ]

        # Sort by start time
        timeline_items.sort(key=lambda x: x[0])

        # Write sections in chronological order
        static_counter = 0
        moving_counter = 0

        for _start_time, segment in timeline_items:
            if segment.type == "static":
                static_counter += 1
                start_str = format_timestamp(segment.start_time)
                end_str = format_timestamp(segment.end_time)

                f.write(
                    f"## Static segment {static_counter} ({start_str} → {end_str})\n"
                )
                f.write(f"Duration: {segment.duration:.1f} seconds\n\n")

                if static_counter in image_paths:
                    f.write(f"![Static frame]({image_paths[static_counter]})\n\n")

            elif segment.type == "moving":
                moving_counter += 1
                start_str = format_timestamp(segment.start_time)
                end_str = format_timestamp(segment.end_time)

                f.write(f"## Moving section {moving_counter}\n")
                f.write(f"Time range: {start_str} → {end_str}\n\n")


def generate_summary(result: AnalysisResult) -> str:
    """Generate a text summary of the analysis results.

    Args:
        result: AnalysisResult from video analysis

    Returns:
        Human-readable summary string
    """
    summary_lines = [
        f"Video duration: {format_timestamp(result.video_duration)}",
        f"Total frames analyzed: {result.total_frames}",
        f"Static segments detected: {len(result.static_segments)}",
        f"Moving sections detected: {len(result.moving_segments)}",
    ]

    if result.static_segments:
        total_static_time = sum(seg.duration for seg in result.static_segments)
        summary_lines.append(
            f"Total static time: {format_timestamp(total_static_time)} "
            f"({total_static_time / result.video_duration * 100:.1f}%)"
        )

    return "\n".join(summary_lines)
