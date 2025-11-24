"""Tests for video analysis functionality."""

import imagehash
import numpy as np
import pytest
from PIL import Image

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


def create_synthetic_stream(
    sequence: list[tuple[str, int]],
    rng_seed: int = 42,
    grid_size: tuple[int, int] = (2, 2),
) -> list[FrameData]:
    """Create frames that represent alternating slides and motion.

    Args:
        sequence: Ordered list of (content_type, frame_count) pairs, where
            ``content_type`` can be ``"motion"`` or a slide label such as ``"slide_A"``.
        rng_seed: Seed to make generated pixels deterministic.
        grid_size: Hash grid dimensions passed to ``_compute_frame_hash``.
    """

    rng = np.random.default_rng(rng_seed)
    slide_cache: dict[str, tuple[np.ndarray, list]] = {}
    frames: list[FrameData] = []
    frame_index = 0

    for content_type, frame_count in sequence:
        for _ in range(frame_count):
            if content_type.startswith("slide_"):
                if content_type not in slide_cache:
                    slide_image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
                    slide_cache[content_type] = (
                        slide_image,
                        _compute_frame_hash((slide_image, grid_size[0], grid_size[1])),
                    )
                slide_image, slide_hashes = slide_cache[content_type]
            elif content_type == "motion":
                slide_image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
                slide_hashes = _compute_frame_hash(
                    (slide_image, grid_size[0], grid_size[1])
                )
            else:
                raise ValueError(f"Unknown content type: {content_type}")

            frames.append(
                FrameData(
                    index=frame_index,
                    timestamp=float(frame_index),
                    image=slide_image,
                    hashes=slide_hashes,
                )
            )
            frame_index += 1

    return frames


def stream_helper(frames: list[FrameData]):  # type: ignore[no-untyped-def]
    """Yield frames with a consistent total count for the detector."""

    total = len(frames)
    for frame in frames:
        yield frame, total


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
    random_noise_image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)

    # Test with 2x2 grid
    hashes = _compute_frame_hash((random_noise_image, 2, 2))

    # Should return 4 hashes (2x2 grid)
    assert len(hashes) == 4


def test_compute_frame_hash_invalid_grid() -> None:
    """Test that oversized grid dimensions raise ValueError."""
    rng = np.random.default_rng(42)
    test_image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)

    # Grid larger than frame should raise ValueError
    with pytest.raises(
        ValueError,
        match="Grid dimensions .* cannot exceed center-cropped frame dimensions",
    ):
        _compute_frame_hash((test_image, 200, 200))

    # Grid with one dimension larger should also raise
    with pytest.raises(
        ValueError,
        match="Grid dimensions .* cannot exceed center-cropped frame dimensions",
    ):
        _compute_frame_hash((test_image, 150, 2))


def test_compute_frame_hash_uses_center_crop() -> None:
    """Hash computation should ignore high-variance borders via center crop."""

    # Create a frame with bright borders but a dark center
    bordered_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    bordered_frame[:, :2] = 255
    bordered_frame[:, -2:] = 255

    expected_crop = Image.fromarray(bordered_frame).crop((2, 2, 8, 8))
    expected_hash = imagehash.phash(expected_crop)

    hashes = _compute_frame_hash((bordered_frame, 1, 1))

    assert len(hashes) == 1
    assert hashes[0] == expected_hash


def test_segment_properties() -> None:
    """Test Segment dataclass properties."""
    segment = Segment(
        type="static", start_time=1.0, end_time=5.0, frames=[0, 1, 2, 3, 4]
    )

    assert segment.duration == 4.0
    assert segment.frame_count == 5


def test_single_static_slide_is_detected_as_one_segment() -> None:
    """Verify a paused slide is grouped into a single static segment."""

    # GIVEN: Five identical frames representing Slide A
    detector = SegmentDetector(threshold=5, static_ratio=0.8, min_frames=3)
    slide_frames = create_synthetic_stream([("slide_A", 5)])

    # WHEN: The detector analyzes the frames
    for _ in detector.analyze(stream_helper(slide_frames)):
        pass

    # THEN: All frames belong to one static segment
    assert len(detector.segments) == 1
    assert detector.segments[0].type == "static"
    assert detector.segments[0].frames == [0, 1, 2, 3, 4]


def test_continuous_motion_is_classified_as_moving_segment() -> None:
    """Treat a sequence with no repeated frames as continuous motion."""

    # GIVEN: Five frames of motion with no repeating slide
    detector = SegmentDetector(threshold=0, static_ratio=0.99, min_frames=3)
    motion_frames = create_synthetic_stream([("motion", 5)])

    # WHEN: The detector processes the stream
    for _ in detector.analyze(stream_helper(motion_frames)):
        pass

    # THEN: The detector starts with a single static frame and classifies the rest as motion
    assert len(detector.segments) == 2
    assert detector.segments[0].type == "static"
    assert detector.segments[0].frames == [0]
    assert detector.segments[1].type == "moving"
    assert detector.segments[1].frames == [1, 2, 3, 4]


def test_detects_transition_from_slide_to_motion_to_new_slide() -> None:
    """Detect static-movement-static pattern without losing frames."""

    # GIVEN: Slide A (3 frames), motion (3 frames), Slide B (3 frames)
    detector = SegmentDetector(threshold=5, static_ratio=0.8, min_frames=2)
    frames = create_synthetic_stream([("slide_A", 3), ("motion", 3), ("slide_B", 3)])

    # WHEN: The detector analyzes the stream
    for _ in detector.analyze(stream_helper(frames)):
        pass

    # THEN: Static → Moving → Static is recorded without duplicated frames
    assert len(detector.segments) == 3

    static_segments = [s for s in detector.segments if s.type == "static"]
    assert len(static_segments) == 2

    assert detector.segments[0].type == "static"
    assert detector.segments[0].frames == [0, 1, 2]

    assert detector.segments[1].type == "moving"
    assert detector.segments[1].frames == [3, 4, 5]

    assert detector.segments[2].type == "static"
    assert detector.segments[2].frames == [6, 7, 8]


def test_frame_streamer_file_not_found() -> None:
    """Test FrameStreamer with non-existent file."""
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        FrameStreamer("nonexistent.mp4")
