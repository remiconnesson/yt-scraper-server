"""Unit tests for video_service.py phase functions."""

import asyncio
from unittest.mock import Mock, patch

import numpy as np
import pytest

from slides_extractor.video_service import (
    _detect_static_segments,
    _upload_segments,
    _build_segments_manifest,
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
    @patch("slides_extractor.video_service.FrameStreamer")
    @patch("slides_extractor.video_service.SegmentDetector")
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

        segments, all_segments, total_frames = await _detect_static_segments(
            "/tmp/video.mp4", "job-123"
        )

        assert len(segments) == 0
        assert total_frames == 10

    @pytest.mark.asyncio
    @patch("slides_extractor.video_service.FrameStreamer")
    @patch("slides_extractor.video_service.SegmentDetector")
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
    @patch("slides_extractor.video_service.cv2.imencode")
    @patch("slides_extractor.video_service.upload_to_s3")
    async def test_upload_segments_success(self, mock_upload, mock_imencode):
        """Test successful upload of segment frames."""
        mock_upload.return_value = (
            "https://s3-endpoint/bucket/video/abc/images/segment_001.png"
        )
        mock_imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))

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

        metadata = await _upload_segments(segments, "video-abc", "job-123")

        assert len(metadata) == 2
        assert metadata[0]["segment_id"] == 1
        assert metadata[0]["start_time"] == 0.0
        assert metadata[0]["end_time"] == 5.0
        assert metadata[0]["duration"] == 5.0
        assert metadata[0]["frame_count"] == 3
        assert (
            metadata[0]["image_url"]
            == "https://s3-endpoint/bucket/video/abc/images/segment_001.png"
        )
        assert (
            metadata[0]["s3_key"]
            == "video/video-abc/static_frames/static_frame_000001.png"
        )
        assert metadata[0]["s3_bucket"] is not None
        assert (
            metadata[0]["s3_uri"]
            == f"s3://{metadata[0]['s3_bucket']}/video/video-abc/static_frames/static_frame_000001.png"
        )
        assert mock_upload.call_count == 2
        assert mock_imencode.call_count == 2

    @pytest.mark.asyncio
    @patch("slides_extractor.video_service.cv2.imencode")
    @patch("slides_extractor.video_service.upload_to_s3")
    async def test_upload_segments_converts_rgb_to_bgr(
        self, mock_upload, mock_imencode
    ):
        """Ensure frames are converted to BGR before encoding."""

        mock_upload.return_value = "https://s3-endpoint/bucket/url"
        mock_imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))

        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        frame[0, 0] = np.array([255, 0, 0], dtype=np.uint8)  # Red in RGB

        segments = [
            Segment(
                type="static",
                start_time=0.0,
                end_time=1.0,
                representative_frame=frame,
                frames=[0],
            ),
        ]

        await _upload_segments(segments, "video-abc", "job-123")

        called_frame = mock_imencode.call_args[0][1]
        assert called_frame.shape == frame.shape
        assert np.array_equal(called_frame[0, 0], np.array([0, 0, 255], dtype=np.uint8))

    @pytest.mark.asyncio
    @patch("slides_extractor.video_service.cv2.imencode")
    @patch("slides_extractor.video_service.upload_to_s3")
    async def test_upload_segments_correct_blob_keys(self, mock_upload, mock_imencode):
        """Test that blob keys are formatted correctly."""
        mock_upload.return_value = "https://s3-endpoint/bucket/url"
        mock_imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        segments = [
            Segment(
                type="static",
                start_time=0.0,
                end_time=5.0,
                representative_frame=frame,
                frames=[0],
            ),
        ]

        await _upload_segments(segments, "my-video-id", "job-123")

        # Verify blob key format
        call_args = mock_upload.call_args
        assert (
            call_args[0][1] == "video/my-video-id/static_frames/static_frame_000001.png"
        )

    @pytest.mark.asyncio
    @patch("slides_extractor.video_service.cv2.imencode")
    @patch("slides_extractor.video_service.upload_to_s3")
    async def test_upload_segments_includes_metadata(self, mock_upload, mock_imencode):
        """Test that blob metadata is included."""
        mock_upload.return_value = "https://s3-endpoint/bucket/url"
        mock_imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))

        frame = np.zeros((25, 25, 3), dtype=np.uint8)
        segments = [
            Segment(
                type="static",
                start_time=1.5,
                end_time=6.5,
                representative_frame=frame,
                frames=[1],
            ),
        ]

        await _upload_segments(segments, "video-id", "job-123")

        call_args = mock_upload.call_args
        metadata = call_args[1]["metadata"]
        assert metadata["video_id"] == "video-id"
        assert metadata["segment_id"] == "1"
        assert metadata["start_time"] == "1.5"
        assert metadata["end_time"] == "6.5"

    @pytest.mark.asyncio
    @patch("slides_extractor.video_service.cv2.imencode")
    @patch("slides_extractor.video_service.upload_to_s3")
    async def test_upload_segments_updates_progress(self, mock_upload, mock_imencode):
        """Test that progress updates during upload."""
        from slides_extractor.video_service import JOBS, JOBS_LOCK

        mock_upload.return_value = "https://s3-endpoint/bucket/url"
        mock_imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        segments = [
            Segment(type="static", representative_frame=frame, frames=[1]),
        ]

        await _upload_segments(segments, "video-id", "job-123")

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
    @patch("slides_extractor.video_service.upload_to_s3")
    @patch("slides_extractor.video_service._upload_segments")
    @patch("slides_extractor.video_service._detect_static_segments")
    async def test_full_pipeline_success(
        self, mock_detect, mock_upload, mock_upload_s3
    ):
        """Test the full orchestration of detection and upload phases."""
        # Mock each phase
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        segments = [Segment(type="static", representative_frame=frame, frames=[1, 2])]
        mock_detect.return_value = (segments, segments, 100)

        metadata = [
            {
                "segment_id": 1,
                "image_url": "https://s3-endpoint/bucket/url",
                "s3_key": "key",
                "s3_bucket": "bucket",
                "s3_uri": "s3://bucket/key",
            }
        ]
        mock_upload.return_value = metadata
        mock_upload_s3.return_value = "https://s3-endpoint/bucket/manifest.json"

        # Import here to avoid circular dependency issues
        from slides_extractor.video_service import extract_and_process_frames

        result = await extract_and_process_frames(
            "/tmp/video.mp4", "video-id", "job-123"
        )

        assert result == metadata
        mock_detect.assert_called_once_with("/tmp/video.mp4", "job-123")
        mock_upload.assert_called_once_with(segments, "video-id", "job-123")
        # Verify manifest upload
        assert mock_upload_s3.called

    @pytest.mark.asyncio
    @patch("slides_extractor.video_service._detect_static_segments")
    async def test_full_pipeline_no_segments(self, mock_detect):
        """Test pipeline when no segments are detected."""
        from slides_extractor.video_service import (
            extract_and_process_frames,
            JOBS,
            JOBS_LOCK,
        )

        mock_detect.return_value = ([], [], 100)

        result = await extract_and_process_frames(
            "/tmp/video.mp4", "video-id", "job-123"
        )

        assert result == []

        # Should mark as completed
        async with JOBS_LOCK:
            assert JOBS["job-123"]["status"] == JobStatus.completed
            assert "No static segments" in JOBS["job-123"]["message"]


class TestBuildSegmentsManifest:
    """Test manifest construction."""

    def test_manifest_structure_includes_s3_info(self):
        video_id = "test-video"

        # Create segments
        segments = [
            Segment(
                type="moving",
                start_time=0.0,
                end_time=1.0,
                representative_frame=None,
                frames=[],
            ),
            Segment(
                type="static",
                start_time=1.0,
                end_time=2.0,
                representative_frame=None,
                frames=[],
            ),
        ]

        # Create metadata matching the static segment
        static_metadata = [
            {
                "frame_id": "frame_001.png",
                "image_url": "https://example.com/img.png",
                "s3_key": "video/test-video/static_frames/frame_001.png",
                "s3_bucket": "my-bucket",
                "s3_uri": "s3://my-bucket/video/test-video/static_frames/frame_001.png",
            }
        ]

        manifest = _build_segments_manifest(video_id, segments, static_metadata)

        # Verify structure
        assert video_id in manifest
        manifest_segments = manifest[video_id]["segments"]
        assert len(manifest_segments) == 2

        # Check moving segment
        assert manifest_segments[0]["kind"] == "moving"

        # Check static segment
        static_seg = manifest_segments[1]
        assert static_seg["kind"] == "static"
        assert static_seg["frame_id"] == "frame_001.png"
        assert static_seg["url"] == "https://example.com/img.png"
        assert static_seg["s3_key"] == "video/test-video/static_frames/frame_001.png"
        assert static_seg["s3_bucket"] == "my-bucket"
        assert (
            static_seg["s3_uri"]
            == "s3://my-bucket/video/test-video/static_frames/frame_001.png"
        )
