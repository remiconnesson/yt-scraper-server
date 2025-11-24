"""Unit tests for video_service.py phase functions."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from slides_extractor.video_service import (
    _compress_segments,
    _detect_static_segments,
    _upload_segments,
    JobStatus,
)
from slides_extractor.extract_slides.video_analyzer import Segment


class TestDetectStaticSegments:
    """Test frame detection phase."""

    @pytest.fixture(autouse=True)
    def cleanup_jobs(self):
        """Clear JOBS before each test."""
        from slides_extractor.video_service import JOBS, JOBS_LOCK

        async def _clear() -> None:
            async with JOBS_LOCK:
                JOBS.clear()

        asyncio.run(_clear())
        yield
        asyncio.run(_clear())

    @pytest.mark.asyncio
    @patch("video_service.FrameStreamer")
    @patch("video_service.SegmentDetector")
    async def test_detect_static_segments_success(
        self, mock_detector_class, mock_streamer_class
    ):
        """Test successful segment detection."""
        # Mock the streamer to yield fake progress
        mock_streamer = Mock()
        mock_streamer.stream.return_value = iter(
            [
                (1, 0, 10),  # segment_count, frame_idx, total_frames
                (1, 5, 10),
                (2, 9, 10),
            ]
        )
        mock_streamer_class.return_value = mock_streamer

        # Mock the detector with some segments
        mock_detector = Mock()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_detector.segments = [
            Segment(type="static", representative_frame=frame),
            Segment(type="moving", representative_frame=None),
            Segment(type="static", representative_frame=frame),
        ]
        mock_detector.analyze.return_value = mock_streamer.stream()
        mock_detector_class.return_value = mock_detector

        segments, total_frames = await _detect_static_segments(
            "/tmp/video.mp4", "job-123"
        )

        assert len(segments) == 2  # Only static segments with frames
        assert total_frames == 10
        assert all(s.type == "static" for s in segments)
        assert all(s.representative_frame is not None for s in segments)

    @pytest.mark.asyncio
    @patch("video_service.FrameStreamer")
    @patch("video_service.SegmentDetector")
    async def test_detect_static_segments_no_segments(
        self, mock_detector_class, mock_streamer_class
    ):
        """Test when no static segments are found."""
        mock_streamer = Mock()
        mock_streamer.stream.return_value = iter([(0, 0, 10)])
        mock_streamer_class.return_value = mock_streamer

        mock_detector = Mock()
        mock_detector.segments = [
            Segment(type="moving", representative_frame=None),
        ]
        mock_detector.analyze.return_value = mock_streamer.stream()
        mock_detector_class.return_value = mock_detector

        segments, total_frames = await _detect_static_segments(
            "/tmp/video.mp4", "job-123"
        )

        assert len(segments) == 0
        assert total_frames == 10

    @pytest.mark.asyncio
    @patch("video_service.FrameStreamer")
    @patch("video_service.SegmentDetector")
    async def test_detect_static_segments_updates_progress(
        self, mock_detector_class, mock_streamer_class
    ):
        """Test that progress updates are sent."""
        from slides_extractor.video_service import JOBS, JOBS_LOCK

        mock_streamer = Mock()
        mock_streamer.stream.return_value = iter(
            [
                (1, 0, 100),
                (1, 50, 100),
                (1, 99, 100),
            ]
        )
        mock_streamer_class.return_value = mock_streamer

        mock_detector = Mock()
        mock_detector.segments = []
        mock_detector.analyze.return_value = mock_streamer.stream()
        mock_detector_class.return_value = mock_detector

        await _detect_static_segments("/tmp/video.mp4", "job-123")

        # Check that job status was updated
        async with JOBS_LOCK:
            assert "job-123" in JOBS
            assert JOBS["job-123"]["status"] == JobStatus.extracting


class TestCompressSegments:
    """Test compression phase."""

    @pytest.fixture(autouse=True)
    def cleanup_jobs(self):
        """Clear JOBS before each test."""
        from slides_extractor.video_service import JOBS, JOBS_LOCK

        async def _clear() -> None:
            async with JOBS_LOCK:
                JOBS.clear()

        asyncio.run(_clear())
        yield
        asyncio.run(_clear())

    @pytest.mark.asyncio
    @patch("video_service.compress_image")
    async def test_compress_segments_success(self, mock_compress):
        """Test successful compression of segments."""
        mock_compress.return_value = (
            b"compressed_data",
            {"format": "WEBP", "quality": 90, "has_text": True},
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        segments = [
            Segment(
                type="static",
                start_time=0.0,
                end_time=5.0,
                representative_frame=frame,
                frames=[0, 1, 2],
            ),
            Segment(
                type="static",
                start_time=10.0,
                end_time=15.0,
                representative_frame=frame,
                frames=[10, 11, 12],
            ),
        ]

        compressed = await _compress_segments(segments, "job-123")

        assert len(compressed) == 2
        assert compressed[0][0] == 1  # idx
        assert compressed[0][2] == b"compressed_data"  # compressed bytes
        assert compressed[1][0] == 2  # idx
        assert mock_compress.call_count == 2

    @pytest.mark.asyncio
    @patch("video_service.compress_image")
    async def test_compress_segments_skips_none_frames(self, mock_compress):
        """Test that segments without frames are skipped."""
        mock_compress.return_value = (b"data", {"quality": 90})

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        segments = [
            Segment(type="static", representative_frame=frame),
            Segment(type="static", representative_frame=None),  # Should skip
            Segment(type="static", representative_frame=frame),
        ]

        compressed = await _compress_segments(segments, "job-123")

        assert len(compressed) == 2
        assert mock_compress.call_count == 2

    @pytest.mark.asyncio
    @patch("video_service.compress_image")
    async def test_compress_segments_updates_progress(self, mock_compress):
        """Test that progress updates during compression."""
        from slides_extractor.video_service import JOBS, JOBS_LOCK

        mock_compress.return_value = (b"data", {"quality": 90})

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        segments = [
            Segment(type="static", representative_frame=frame, frames=[1, 2, 3]),
        ]

        await _compress_segments(segments, "job-123")

        async with JOBS_LOCK:
            assert "job-123" in JOBS
            assert JOBS["job-123"]["status"] == JobStatus.compressing


class TestUploadSegments:
    """Test upload phase."""

    @pytest.fixture(autouse=True)
    def cleanup_jobs(self):
        """Clear JOBS before each test."""
        from slides_extractor.video_service import JOBS, JOBS_LOCK

        async def _clear() -> None:
            async with JOBS_LOCK:
                JOBS.clear()

        asyncio.run(_clear())
        yield
        asyncio.run(_clear())

    @pytest.mark.asyncio
    @patch("video_service.upload_to_vercel_blob")
    async def test_upload_segments_success(self, mock_upload):
        """Test successful upload of compressed segments."""
        mock_upload.return_value = (
            "https://blob.vercel-storage.com/video/abc/images/segment_001.webp"
        )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        compressed_segments = [
            (
                1,
                Segment(
                    type="static",
                    start_time=0.0,
                    end_time=5.0,
                    frames=[0, 1, 2],
                ),
                b"compressed_data_1",
                {"quality": 90, "has_text": True},
            ),
            (
                2,
                Segment(
                    type="static",
                    start_time=10.0,
                    end_time=15.0,
                    frames=[10, 11, 12],
                ),
                b"compressed_data_2",
                {"quality": 85, "has_text": False},
            ),
        ]

        metadata = await _upload_segments(compressed_segments, "video-abc", "job-123")

        assert len(metadata) == 2
        assert metadata[0]["segment_id"] == 1
        assert metadata[0]["start_time"] == 0.0
        assert metadata[0]["end_time"] == 5.0
        assert metadata[0]["duration"] == 5.0
        assert metadata[0]["frame_count"] == 3
        assert (
            metadata[0]["image_url"]
            == "https://blob.vercel-storage.com/video/abc/images/segment_001.webp"
        )
        assert metadata[0]["compression_info"]["quality"] == 90

        assert mock_upload.call_count == 2

    @pytest.mark.asyncio
    @patch("video_service.upload_to_vercel_blob")
    async def test_upload_segments_correct_blob_keys(self, mock_upload):
        """Test that blob keys are formatted correctly."""
        mock_upload.return_value = "https://blob.vercel-storage.com/url"

        compressed_segments = [
            (
                5,
                Segment(type="static", start_time=0.0, end_time=5.0, frames=[0]),
                b"data",
                {"quality": 90},
            ),
        ]

        await _upload_segments(compressed_segments, "my-video-id", "job-123")

        # Verify blob key format
        call_args = mock_upload.call_args
        assert call_args[0][1] == "video/my-video-id/images/segment_005.webp"

    @pytest.mark.asyncio
    @patch("video_service.upload_to_vercel_blob")
    async def test_upload_segments_includes_metadata(self, mock_upload):
        """Test that blob metadata is included."""
        mock_upload.return_value = "https://blob.vercel-storage.com/url"

        compressed_segments = [
            (
                1,
                Segment(type="static", start_time=1.5, end_time=6.5, frames=[1]),
                b"data",
                {"quality": 90},
            ),
        ]

        await _upload_segments(compressed_segments, "video-id", "job-123")

        call_args = mock_upload.call_args
        metadata = call_args[1]["metadata"]
        assert metadata["video_id"] == "video-id"
        assert metadata["segment_id"] == "1"
        assert metadata["start_time"] == "1.5"
        assert metadata["end_time"] == "6.5"

    @pytest.mark.asyncio
    @patch("video_service.upload_to_vercel_blob")
    async def test_upload_segments_updates_progress(self, mock_upload):
        """Test that progress updates during upload."""
        from slides_extractor.video_service import JOBS, JOBS_LOCK

        mock_upload.return_value = "https://blob.vercel-storage.com/url"

        compressed_segments = [
            (1, Segment(type="static", frames=[1]), b"data", {"quality": 90}),
        ]

        await _upload_segments(compressed_segments, "video-id", "job-123")

        async with JOBS_LOCK:
            assert "job-123" in JOBS
            assert JOBS["job-123"]["status"] == JobStatus.uploading


class TestExtractAndProcessFramesIntegration:
    """Integration tests for the full pipeline orchestration."""

    @pytest.fixture(autouse=True)
    def cleanup_jobs(self):
        """Clear JOBS before each test."""
        from slides_extractor.video_service import JOBS, JOBS_LOCK

        async def _clear() -> None:
            async with JOBS_LOCK:
                JOBS.clear()

        asyncio.run(_clear())
        yield
        asyncio.run(_clear())

    @pytest.mark.asyncio
    @patch("video_service._upload_segments")
    @patch("video_service._compress_segments")
    @patch("video_service._detect_static_segments")
    async def test_full_pipeline_success(self, mock_detect, mock_compress, mock_upload):
        """Test the full orchestration of all three phases."""
        # Mock each phase
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        segments = [Segment(type="static", representative_frame=frame, frames=[1, 2])]
        mock_detect.return_value = (segments, 100)

        compressed = [(1, segments[0], b"data", {"quality": 90})]
        mock_compress.return_value = compressed

        metadata = [
            {
                "segment_id": 1,
                "image_url": "https://blob.vercel-storage.com/url",
                "compression_info": {"quality": 90},
            }
        ]
        mock_upload.return_value = metadata

        # Import here to avoid circular dependency issues
        from slides_extractor.video_service import extract_and_process_frames

        result = await extract_and_process_frames(
            "/tmp/video.mp4", "video-id", "job-123"
        )

        assert result == metadata
        mock_detect.assert_called_once_with("/tmp/video.mp4", "job-123")
        mock_compress.assert_called_once_with(segments, "job-123")
        mock_upload.assert_called_once_with(compressed, "video-id", "job-123")

    @pytest.mark.asyncio
    @patch("video_service._detect_static_segments")
    async def test_full_pipeline_no_segments(self, mock_detect):
        """Test pipeline when no segments are detected."""
        from slides_extractor.video_service import (
            extract_and_process_frames,
            JOBS,
            JOBS_LOCK,
        )

        mock_detect.return_value = ([], 100)

        result = await extract_and_process_frames(
            "/tmp/video.mp4", "video-id", "job-123"
        )

        assert result == []

        # Should mark as completed
        async with JOBS_LOCK:
            assert JOBS["job-123"]["status"] == JobStatus.completed
            assert "No static segments" in JOBS["job-123"]["message"]
