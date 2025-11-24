"""Tests for video analysis functionality."""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from slides_extractor.extract_slides.cli import main, video_main
from slides_extractor.extract_slides.video_analyzer import (
    FrameData,
    FrameStreamer,
    Segment,
    SegmentDetector,
    _compute_frame_hash,
)
from slides_extractor.extract_slides.video_output import (
    format_timestamp,
    format_timestamp_for_filename,
)


def test_format_timestamp() -> None:
    """Test timestamp formatting to HH:MM:SS."""
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(65) == "00:01:05"
    assert format_timestamp(3661) == "01:01:01"
    assert format_timestamp(3723.5) == "01:02:03"


def test_format_timestamp_for_filename() -> None:
    """Test timestamp formatting for filenames (with hyphens)."""
    assert format_timestamp_for_filename(0) == "00-00-00"
    assert format_timestamp_for_filename(65) == "00-01-05"
    assert format_timestamp_for_filename(3661) == "01-01-01"


def test_compute_frame_hash() -> None:
    """Test frame hash computation."""
    # Create a simple test image (100x100 pixels, 3 channels, RGB) with seeded RNG
    rng = np.random.default_rng(42)
    test_image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)

    # Test with 2x2 grid
    hashes = _compute_frame_hash((test_image, 2, 2))

    # Should return 4 hashes (2x2 grid)
    assert len(hashes) == 4


def test_compute_frame_hash_invalid_grid() -> None:
    """Test that oversized grid dimensions raise ValueError."""
    rng = np.random.default_rng(42)
    test_image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)

    # Grid larger than frame should raise ValueError
    with pytest.raises(
        ValueError, match="Grid dimensions .* cannot exceed frame dimensions"
    ):
        _compute_frame_hash((test_image, 200, 200))

    # Grid with one dimension larger should also raise
    with pytest.raises(
        ValueError, match="Grid dimensions .* cannot exceed frame dimensions"
    ):
        _compute_frame_hash((test_image, 150, 2))


def test_segment_properties() -> None:
    """Test Segment dataclass properties."""
    segment = Segment(
        type="static", start_time=1.0, end_time=5.0, frames=[0, 1, 2, 3, 4]
    )

    assert segment.duration == 4.0
    assert segment.frame_count == 5


def test_segment_detector_single_static() -> None:
    """Test segment detector with identical frames (single static segment)."""
    # Create detector
    detector = SegmentDetector(threshold=5, static_ratio=0.8, min_frames=3)

    # Create 5 identical frames with seeded RNG
    rng = np.random.default_rng(42)
    test_image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    hashes = _compute_frame_hash((test_image, 2, 2))

    frames = [
        FrameData(index=i, timestamp=float(i), image=test_image, hashes=hashes)
        for i in range(5)
    ]

    # Process frames
    def frame_stream():  # type: ignore[no-untyped-def]
        for frame in frames:
            yield frame, len(frames)

    # Consume the generator
    for _ in detector.analyze(frame_stream()):
        pass

    # Should detect one static segment
    assert len(detector.segments) == 1
    assert detector.segments[0].type == "static"
    assert detector.segments[0].frames == [0, 1, 2, 3, 4]


def test_segment_detector_all_different() -> None:
    """Test segment detector with all different frames (moving segment)."""
    # Create detector with very strict settings
    detector = SegmentDetector(threshold=0, static_ratio=0.99, min_frames=3)

    # Create 5 completely different frames with seeded RNG
    rng = np.random.default_rng(42)
    frames = [
        FrameData(
            index=i,
            timestamp=float(i),
            image=rng.integers(0, 255, (100, 100, 3), dtype=np.uint8),
            hashes=_compute_frame_hash(
                (rng.integers(0, 255, (100, 100, 3), dtype=np.uint8), 2, 2)
            ),
        )
        for i in range(5)
    ]

    # Process frames
    def frame_stream():  # type: ignore[no-untyped-def]
        for frame in frames:
            yield frame, len(frames)

    # Consume the generator
    for _ in detector.analyze(frame_stream()):
        pass

    # Should have segments, but they should not all be static
    # First frame starts as static, then transitions to moving
    moving_segments = [s for s in detector.segments if s.type == "moving"]

    # With very different frames and strict settings, should mostly be moving
    assert len(moving_segments) >= 1


def test_segment_detector_mixed() -> None:
    """Test segment detector with mixed static and moving frames."""
    detector = SegmentDetector(threshold=5, static_ratio=0.8, min_frames=2)

    # Create frames: 3 static, 3 different, 3 static
    rng = np.random.default_rng(42)
    static_image1 = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    static_image2 = rng.integers(100, 200, (100, 100, 3), dtype=np.uint8)

    hashes1 = _compute_frame_hash((static_image1, 2, 2))
    hashes2 = _compute_frame_hash((static_image2, 2, 2))

    frames = []
    # First 3 frames: static
    for i in range(3):
        frames.append(
            FrameData(index=i, timestamp=float(i), image=static_image1, hashes=hashes1)
        )
    # Next 3 frames: different (moving)
    for i in range(3, 6):
        img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        frames.append(
            FrameData(
                index=i,
                timestamp=float(i),
                image=img,
                hashes=_compute_frame_hash((img, 2, 2)),
            )
        )
    # Last 3 frames: static again
    for i in range(6, 9):
        frames.append(
            FrameData(index=i, timestamp=float(i), image=static_image2, hashes=hashes2)
        )

    # Process frames
    def frame_stream():  # type: ignore[no-untyped-def]
        for frame in frames:
            yield frame, len(frames)

    # Consume the generator
    for _ in detector.analyze(frame_stream()):
        pass

    # Should detect at least 2 static segments
    static_segments = [s for s in detector.segments if s.type == "static"]
    assert len(static_segments) >= 2

    # Verify frame indices don't have duplicates across segments
    all_frames = []
    for seg in detector.segments:
        all_frames.extend(seg.frames)
    assert len(all_frames) == len(set(all_frames)), (
        "Frame indices should not be duplicated"
    )

    # Verify first static segment contains expected frames
    first_static = static_segments[0]
    assert first_static.frames == [0, 1, 2], (
        f"Expected [0, 1, 2], got {first_static.frames}"
    )

    # Verify last static segment contains expected frames
    last_static = static_segments[-1]
    assert last_static.frames == [6, 7, 8], (
        f"Expected [6, 7, 8], got {last_static.frames}"
    )


def test_frame_streamer_file_not_found() -> None:
    """Test FrameStreamer with non-existent file."""
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        FrameStreamer("nonexistent.mp4")


def test_analyze_video_command_help() -> None:
    """Test that analyze-video command shows help."""
    runner = CliRunner()
    result = runner.invoke(main, ["analyze-video", "--help"])
    assert result.exit_code == 0
    assert "grid-based image hashing" in result.output


def test_video_main_help() -> None:
    """Test that video_static_segments command shows help."""
    runner = CliRunner()
    result = runner.invoke(video_main, ["--help"])
    assert result.exit_code == 0
    assert "grid-based image hashing" in result.output


def test_video_main_validation_grid_cols() -> None:
    """Test video_main validates grid columns."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create a dummy file
        Path("test.mp4").touch()

        result = runner.invoke(
            video_main,
            ["--input", "test.mp4", "--output-dir", "output", "--grid-cols", "0"],
        )
        assert result.exit_code == 1
        assert "Grid dimensions must be positive" in result.output


def test_video_main_validation_static_cell_ratio() -> None:
    """Test video_main validates min-static-cell-ratio."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create a dummy file
        Path("test.mp4").touch()

        result = runner.invoke(
            video_main,
            [
                "--input",
                "test.mp4",
                "--output-dir",
                "output",
                "--min-static-cell-ratio",
                "1.5",
            ],
        )
        assert result.exit_code == 1
        assert "min-static-cell-ratio must be between 0 and 1" in result.output


def test_video_main_validation_min_static_frames() -> None:
    """Test video_main validates min-static-frames."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create a dummy file
        Path("test.mp4").touch()

        result = runner.invoke(
            video_main,
            [
                "--input",
                "test.mp4",
                "--output-dir",
                "output",
                "--min-static-frames",
                "0",
            ],
        )
        assert result.exit_code == 1
        assert "min-static-frames must be at least 1" in result.output
