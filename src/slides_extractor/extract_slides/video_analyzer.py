"""Video analysis for detecting static segments using streaming and parallel processing.

This module provides a memory-efficient streaming approach to video analysis,
using parallel hash computation and a sophisticated state machine for segment detection.
"""

import concurrent.futures
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypedDict, Union

import imagehash
import numpy as np
from PIL import Image


class AnalysisConfig(TypedDict):
    """Configuration for video analysis."""

    grid_cols: int
    grid_rows: int
    cell_hash_threshold: int
    min_static_cell_ratio: float
    min_static_frames: int


@dataclass
class FrameData:
    """Holds all necessary data for a single frame.

    Attributes:
        index: Frame index in the sequence of extracted frames
        timestamp: Timestamp in seconds from video start
        image: Frame image as numpy array (RGB format)
        hashes: List of perceptual hashes for grid cells (flattened)
    """

    index: int
    timestamp: float
    image: np.ndarray
    hashes: list[imagehash.ImageHash]


@dataclass
class Segment:
    """Represents a detected video segment (static or moving).

    Attributes:
        type: Segment type ('static' or 'moving')
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        frames: List of frame indices in this segment
        representative_frame: Representative image for static segments (None for moving)
        last_frame: The final image of the static segment (used for build-up detection)
    """

    type: str  # 'static' or 'moving'
    start_time: float = 0.0
    end_time: float = 0.0
    frames: list[int] = field(default_factory=list)
    representative_frame: Optional[np.ndarray] = None
    last_frame: Optional[np.ndarray] = None

    @property
    def duration(self) -> float:
        """Calculate segment duration in seconds."""
        return self.end_time - self.start_time

    @property
    def frame_count(self) -> int:
        """Get number of frames in this segment."""
        return len(self.frames)


CENTER_CROP_RATIO = 0.6


def _compute_frame_hash(args: tuple[np.ndarray, int, int]) -> list[imagehash.ImageHash]:
    """Worker function for parallel hash computation.

    This function must be at module level for multiprocessing serialization.

    Args:
        args: Tuple of (frame_array, grid_cols, grid_rows)

    Returns:
        Flattened list of perceptual hashes for each grid cell

    Raises:
        ValueError: If grid dimensions are larger than frame dimensions
    """
    frame_arr, grid_x, grid_y = args

    # Convert to PIL once for efficiency
    pil_image = Image.fromarray(frame_arr)

    # Compute a center crop to reduce border influence
    crop_w = max(1, int(pil_image.width * CENTER_CROP_RATIO))
    crop_h = max(1, int(pil_image.height * CENTER_CROP_RATIO))
    left = (pil_image.width - crop_w) // 2
    top = (pil_image.height - crop_h) // 2
    cropped = pil_image.crop((left, top, left + crop_w, top + crop_h))
    w, h = cropped.size

    # Validate grid size against frame dimensions
    if grid_x > w or grid_y > h:
        raise ValueError(
            f"Grid dimensions ({grid_x}x{grid_y}) cannot exceed "
            f"center-cropped frame dimensions ({w}x{h})"
        )

    cw, ch = w // grid_x, h // grid_y

    # Ensure non-zero cell sizes
    if cw == 0 or ch == 0:
        raise ValueError(
            f"Grid dimensions ({grid_x}x{grid_y}) result in zero-sized cells "
            f"for center-cropped frame size ({w}x{h})"
        )

    hashes: list[imagehash.ImageHash] = []
    for r in range(grid_y):
        for c in range(grid_x):
            # Fast crop using box coordinates
            box = (c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)
            hashes.append(imagehash.phash(cropped.crop(box)))
    return hashes


class FrameStreamer:
    """Handles video reading, resizing, and parallel hashing.

    This class provides a streaming interface to video frames, processing them
    in batches with parallel hash computation for efficiency.

    Attributes:
        path: Path to the video file
        grid_x: Number of grid columns for hash computation
        grid_y: Number of grid rows for hash computation
        fps: Frames per second to extract from video
        max_width: Maximum width for frame resizing (maintains aspect ratio)
    """

    def __init__(
        self,
        video_path: str,
        grid_x: int = 4,
        grid_y: int = 4,
        fps: float = 1.0,
        max_width: int = 1280,
    ) -> None:
        """Initialize the frame streamer.

        Args:
            video_path: Path to the input video file
            grid_x: Number of columns in the grid (default: 4)
            grid_y: Number of rows in the grid (default: 4)
            fps: Frames per second to extract (default: 1.0)
            max_width: Maximum frame width in pixels (default: 1280)

        Raises:
            FileNotFoundError: If video file doesn't exist
        """
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.path = video_path
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.fps = fps
        self.max_width = max_width

    def stream(
        self, batch_size: int = 10
    ) -> Generator[tuple[FrameData, int], None, None]:
        """Stream frames with parallel hash computation.

        Yields frames as they are processed, enabling real-time progress updates
        and memory-efficient processing of large videos.

        Args:
            batch_size: Number of frames to process in parallel (default: 10)

        Yields:
            Tuple of (FrameData object, total expected frame count)

        Raises:
            ValueError: If video cannot be opened or has invalid properties
        """
        import cv2

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        # Handle videos with missing or invalid FPS metadata
        if video_fps <= 0:
            raise ValueError(
                f"Video has invalid FPS ({video_fps}). Cannot determine frame extraction rate."
            )

        duration = total_frames / video_fps
        expected_count = int(duration * self.fps)

        extract_interval = 1.0 / self.fps
        next_time = 0.0
        frame_idx = 0

        batch_frames: list[dict[str, Union[int, float, np.ndarray]]] = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                if curr_time >= next_time:
                    # Resize if needed
                    h, w = frame.shape[:2]  # type: ignore[union-attr]
                    if w > self.max_width:
                        scale = self.max_width / w
                        frame = cv2.resize(frame, (self.max_width, int(h * scale)))

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Add to batch
                    batch_frames.append(
                        {"index": frame_idx, "time": curr_time, "img": frame}
                    )

                    # Process batch if full
                    if len(batch_frames) >= batch_size:
                        yield from self._process_batch(
                            executor, batch_frames, expected_count
                        )
                        batch_frames = []

                    frame_idx += 1
                    next_time += extract_interval

            # Process remaining frames
            if batch_frames:
                yield from self._process_batch(executor, batch_frames, expected_count)

        cap.release()

    def _process_batch(
        self,
        executor: concurrent.futures.ProcessPoolExecutor,
        batch: list[dict[str, Union[int, float, np.ndarray]]],
        total_count: int,
    ) -> Generator[tuple[FrameData, int], None, None]:
        """Process a batch of frames in parallel.

        Args:
            executor: ProcessPoolExecutor for parallel processing
            batch: List of frame data dictionaries
            total_count: Total expected frame count

        Yields:
            Tuple of (FrameData object, total expected frame count)
        """
        # Prepare arguments for parallel hashing
        tasks: list[tuple[Any, int, int]] = [
            (item["img"], self.grid_x, self.grid_y) for item in batch
        ]

        # Run parallel hashing
        results = list(executor.map(_compute_frame_hash, tasks))

        # Yield fully formed FrameData objects
        for i, hashes in enumerate(results):
            item = batch[i]
            yield (
                FrameData(
                    int(item["index"]),
                    float(item["time"]),
                    item["img"],  # type: ignore[arg-type]
                    hashes,
                ),
                total_count,
            )


class SegmentDetector:
    """State machine for detecting static and moving segments.

    This detector uses a sophisticated tentative buffer system to accurately
    detect transitions between static and moving segments, even when the
    transition boundary is ambiguous.

    Important:
        This detector is designed for single-use per video. If you need to analyze
        multiple videos, create a new SegmentDetector instance for each video.
        The state (segments, current_seg, buffers) accumulates during analysis
        and is not automatically reset.

    Attributes:
        threshold: Maximum hash distance for cells to be considered unchanged
        ratio: Minimum ratio of unchanged cells for frame similarity
        min_frames: Minimum consecutive frames to confirm a static segment
        segments: List of detected segments (accumulated during analysis)
    """

    def __init__(
        self, threshold: int = 5, static_ratio: float = 0.8, min_frames: int = 3
    ) -> None:
        """Initialize the segment detector.

        Args:
            threshold: Maximum hash distance for cells to match (default: 5)
            static_ratio: Minimum ratio of matching cells (default: 0.8)
            min_frames: Minimum static frames required (default: 3)
        """
        self.threshold = threshold
        self.ratio = static_ratio
        self.min_frames = min_frames

        # State
        self.segments: list[Segment] = []
        self.current_seg: Optional[Segment] = None
        self.anchor_hash: Optional[list[imagehash.ImageHash]] = None

        # Tentative buffer for detecting transitions
        self.tentative_static: list[FrameData] = []
        self.tentative_start_hash: Optional[list[imagehash.ImageHash]] = None

    def analyze(
        self, frame_stream: Generator[tuple[FrameData, int], None, None]
    ) -> Generator[tuple[int, int, int], None, None]:
        """Analyze video frames from the stream.

        Args:
            frame_stream: Generator yielding (FrameData, total_count) tuples

        Yields:
            Progress tuples of (segment_count, current_frame_index, total_frames)

        Note:
            Detected segments are accessible via self.segments after completion
        """
        for frame, total in frame_stream:
            self._process_frame(frame)
            yield len(self.segments), frame.index, total

        self._finalize()

    def _process_frame(self, frame: FrameData) -> None:
        """Process a single frame through the state machine.

        Args:
            frame: FrameData object to process
        """
        if self.current_seg is None:
            self._start_new_segment("static", frame)
            return

        is_similar = self._compare(self.anchor_hash, frame.hashes)

        if self.current_seg.type == "static":
            self._handle_static_state(frame, is_similar)
        else:
            self._handle_moving_state(frame)

    def _handle_static_state(self, frame: FrameData, is_similar: bool) -> None:
        """Handle frame processing when in static state.

        Args:
            frame: Current frame to process
            is_similar: Whether frame is similar to anchor
        """
        if is_similar:
            self.current_seg.frames.append(frame.index)  # type: ignore[union-attr]
            self.current_seg.end_time = frame.timestamp  # type: ignore[union-attr]
            self.current_seg.last_frame = frame.image  # type: ignore[union-attr]
        else:
            # Transition: Static -> Moving
            self._commit_current_segment()
            self._start_new_segment("moving", frame)

            # Initialize tentative buffer
            self.tentative_static = [frame]
            self.tentative_start_hash = frame.hashes

    def _handle_moving_state(self, frame: FrameData) -> None:
        """Handle frame processing when in moving state.

        Uses tentative buffer to detect potential static segments within
        moving sections.

        Args:
            frame: Current frame to process
        """
        # Check if this frame matches the start of tentative static buffer
        if self.tentative_start_hash:
            is_tentative_match = self._compare(self.tentative_start_hash, frame.hashes)

            if is_tentative_match:
                self.tentative_static.append(frame)

                # Confirm new static segment if buffer is long enough
                if len(self.tentative_static) >= self.min_frames:
                    # Remove these frames from current moving segment
                    frame_ids_to_remove = {f.index for f in self.tentative_static}
                    self.current_seg.frames = [  # type: ignore[union-attr]
                        f
                        for f in self.current_seg.frames  # type: ignore[union-attr]
                        if f not in frame_ids_to_remove
                    ]
                    self.current_seg.end_time = self.tentative_static[  # type: ignore[union-attr]
                        0
                    ].timestamp

                    # Save the moving segment
                    self._commit_current_segment()

                    # Start the new static segment
                    first = self.tentative_static[0]
                    self._start_new_segment("static", first)
                    for buf_frame in self.tentative_static[1:]:
                        self.current_seg.frames.append(buf_frame.index)  # type: ignore[union-attr]
                        self.current_seg.end_time = buf_frame.timestamp  # type: ignore[union-attr]
                        self.current_seg.last_frame = buf_frame.image  # type: ignore[union-attr]

                    # Clear buffer
                    self.tentative_static = []
                    self.tentative_start_hash = None
                    # Frame already added to new static segment, skip moving segment append
                    return
            else:
                # Noise - reset tentative buffer
                self.tentative_static = [frame]
                self.tentative_start_hash = frame.hashes

        # Add to current moving segment (only reached if still in moving state)
        self.current_seg.frames.append(frame.index)  # type: ignore[union-attr]
        self.current_seg.end_time = frame.timestamp  # type: ignore[union-attr]

    def _compare(
        self,
        hash1: Optional[list[imagehash.ImageHash]],
        hash2: Optional[list[imagehash.ImageHash]],
    ) -> bool:
        """Compare two hash lists for similarity.

        Args:
            hash1: First hash list
            hash2: Second hash list

        Returns:
            True if hashes are similar based on threshold and ratio
        """
        if not hash1 or not hash2:
            return False
        matches = sum(1 for h1, h2 in zip(hash1, hash2) if h1 - h2 <= self.threshold)
        return (matches / len(hash1)) >= self.ratio

    def _start_new_segment(self, type_: str, frame: FrameData) -> None:
        """Start a new segment.

        Args:
            type_: Segment type ('static' or 'moving')
            frame: First frame of the segment
        """
        self.current_seg = Segment(
            type=type_, start_time=frame.timestamp, end_time=frame.timestamp
        )
        self.current_seg.frames = [frame.index]
        if type_ == "static":
            self.current_seg.representative_frame = frame.image
            self.current_seg.last_frame = frame.image
            self.anchor_hash = frame.hashes
        else:
            self.anchor_hash = None

    def _commit_current_segment(self) -> None:
        """Commit current segment to the results list.

        Merges with previous segment if they are the same type.
        """
        if self.current_seg and self.current_seg.frames:
            # Merge logic: if previous segment is same type, merge
            if self.segments and self.segments[-1].type == self.current_seg.type:
                self.segments[-1].frames.extend(self.current_seg.frames)
                self.segments[-1].end_time = self.current_seg.end_time
                if self.current_seg.type == "static":
                    self.segments[-1].last_frame = self.current_seg.last_frame
            else:
                self.segments.append(self.current_seg)

    def _finalize(self) -> None:
        """Finalize analysis by committing any pending segment."""
        self._commit_current_segment()


@dataclass
class AnalysisResult:
    """Result of video analysis.

    Attributes:
        segments: List of all detected segments (static and moving)
        total_frames: Total number of frames analyzed
        video_duration: Total video duration in seconds
    """

    segments: list[Segment]
    total_frames: int
    video_duration: float

    @property
    def static_segments(self) -> list[Segment]:
        """Get only static segments."""
        return [s for s in self.segments if s.type == "static"]

    @property
    def moving_segments(self) -> list[Segment]:
        """Get only moving segments."""
        return [s for s in self.segments if s.type == "moving"]


def analyze_video(
    video_path: str, config: AnalysisConfig, verbose: bool = False
) -> AnalysisResult:
    """Analyze video to detect static segments using streaming approach.

    This is the main entry point for video analysis. It uses a streaming
    pipeline with parallel hash computation for memory efficiency.

    Args:
        video_path: Path to the input video file
        config: Analysis configuration parameters
        verbose: Enable progress output (default: False)

    Returns:
        AnalysisResult containing detected segments

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be processed or config is invalid

    Example:
        >>> config: AnalysisConfig = {
        ...     "grid_cols": 4,
        ...     "grid_rows": 4,
        ...     "cell_hash_threshold": 5,
        ...     "min_static_cell_ratio": 0.8,
        ...     "min_static_frames": 3,
        ... }
        >>> result = analyze_video("video.mp4", config)
        >>> print(f"Found {len(result.static_segments)} static segments")
    """
    # Initialize pipeline components
    streamer = FrameStreamer(
        video_path,
        grid_x=config["grid_cols"],
        grid_y=config["grid_rows"],
        fps=1.0,
        max_width=1280,
    )
    detector = SegmentDetector(
        threshold=config["cell_hash_threshold"],
        static_ratio=config["min_static_cell_ratio"],
        min_frames=config["min_static_frames"],
    )

    # Get the streaming pipeline
    pipeline = detector.analyze(streamer.stream())

    # Process frames (with optional progress tracking)
    # Track actual frames processed, not metadata estimates
    actual_frames_processed = 0
    for _seg_count, f_idx, total in pipeline:
        actual_frames_processed = f_idx + 1  # f_idx is 0-based
        if verbose and f_idx % 10 == 0:
            print(f"Processing frame {f_idx}/{total}")

    # Calculate video duration from last processed frame
    video_duration = 0.0
    if detector.segments:
        video_duration = max(seg.end_time for seg in detector.segments)

    return AnalysisResult(
        segments=detector.segments,
        total_frames=actual_frames_processed,  # Use actual count, not estimate
        video_duration=video_duration,
    )
