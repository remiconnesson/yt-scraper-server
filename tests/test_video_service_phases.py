"""Unit tests for video_service.py phase functions."""

import asyncio
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from slides_extractor.video_service import (
    _detect_static_segments,
    _upload_segments,
    _build_segments_manifest,
    JobStatus,
)
from slides_extractor.settings import SLIDE_IMAGE_QUALITY
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
            "/tmp/video.mp4", "video-123"
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

        await _detect_static_segments("/tmp/video.mp4", "video-123")

        # Check that job status was updated
        async with JOBS_LOCK:
            assert "video-123" in JOBS
            assert JOBS["video-123"]["status"] == JobStatus.extracting


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
    @patch("slides_extractor.video_service.upload_to_blob")
    async def test_upload_segments_success(self, mock_upload, mock_imencode):
        """Test successful upload of segment frames."""
        mock_upload.return_value = (
            "https://blob.vercel-storage.com/slides/abc/1-first.webp"
        )
        mock_imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        segments = [
            Segment(
                type="static",
                start_time=0.0,
                end_time=5.0,
                representative_frame=frame,
                last_frame=frame,
                frames=[0, 1, 2],
            ),
            Segment(
                type="static",
                start_time=10.0,
                end_time=15.0,
                representative_frame=frame,
                last_frame=frame,
                frames=[10, 11, 12],
            ),
        ]

        metadata = await _upload_segments(segments, "video-abc")

        assert len(metadata) == 2
        assert metadata[0]["segment_id"] == 1
        assert metadata[0]["first_frame"]["frame_id"] == "1-first.webp"
        assert metadata[0]["last_frame"]["frame_id"] == "1-last.webp"
        assert metadata[0]["last_frame"]["duplicate_of"] == {
            "segment_id": 1,
            "frame_position": "first",
        }
        assert metadata[1]["first_frame"]["duplicate_of"] == {
            "segment_id": 1,
            "frame_position": "first",
        }
        # Only 1 unique frame should be uploaded if deduplication works as expected in this test
        # Actually, in the old test it was 4 because it didn't deduplicate well or mocked it differently?
        # Let's check my implementation. My implementation deduplicates before uploading.
        # In this test, all frames are the same 'frame' (zeros).
        # So only the very first one should be uploaded.
        assert mock_upload.call_count == 1
        assert mock_imencode.call_count == 4 # still encodes all to check hashes

    @pytest.mark.asyncio
    @patch("slides_extractor.video_service.cv2.imencode")
    @patch("slides_extractor.video_service.upload_to_blob")
    async def test_upload_segments_converts_rgb_to_bgr(
        self, mock_upload, mock_imencode
    ):
        """Ensure frames are converted to BGR before encoding."""

        mock_upload.return_value = "https://blob.vercel-storage.com/url"
        mock_imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))

        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        frame[0, 0] = np.array([255, 0, 0], dtype=np.uint8)  # Red in RGB

        segments = [
            Segment(
                type="static",
                start_time=0.0,
                end_time=1.0,
                representative_frame=frame,
                last_frame=frame,
                frames=[0],
            ),
        ]

        await _upload_segments(segments, "video-abc")

        called_frame = mock_imencode.call_args_list[0][0][1]
        assert called_frame.shape == frame.shape
        assert np.array_equal(called_frame[0, 0], np.array([0, 0, 255], dtype=np.uint8))

    @pytest.mark.asyncio
    @patch("slides_extractor.video_service.cv2.imencode")
    @patch("slides_extractor.video_service.upload_to_blob")
    async def test_upload_segments_correct_blob_keys(self, mock_upload, mock_imencode):
        """Test that blob keys are formatted correctly."""
        mock_upload.return_value = "https://blob.vercel-storage.com/url"
        mock_imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        segments = [
            Segment(
                type="static",
                start_time=0.0,
                end_time=5.0,
                representative_frame=frame,
                last_frame=frame,
                frames=[0],
            ),
        ]

        await _upload_segments(segments, "my-video-id")

        # Verify blob path format
        paths = {call_args[0][1] for call_args in mock_upload.call_args_list}
        # My implementation deduplicates, and since representative_frame == last_frame, 
        # only the first one is uploaded.
        assert paths == {
            "slides/my-video-id/1-first.webp",
        }

    @pytest.mark.asyncio
    @patch("slides_extractor.video_service.cv2.imencode")
    @patch("slides_extractor.video_service.upload_to_blob")
    async def test_upload_segments_updates_progress(self, mock_upload, mock_imencode):
        """Test that progress updates during upload."""
        from slides_extractor.video_service import JOBS, JOBS_LOCK

        mock_upload.return_value = "https://blob.vercel-storage.com/url"
        mock_imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        segments = [
            Segment(type="static", representative_frame=frame, frames=[1]),
        ]

        await _upload_segments(segments, "video-id")

        async with JOBS_LOCK:
            assert "video-id" in JOBS
            assert JOBS["video-id"]["status"] == JobStatus.uploading


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
    @patch("slides_extractor.video_service.upload_to_blob")
    @patch("slides_extractor.video_service._upload_segments")
    @patch("slides_extractor.video_service._detect_static_segments")
    async def test_full_pipeline_success(
        self, mock_detect, mock_upload_segments, mock_upload_to_blob
    ):
        """Test the full orchestration of detection and upload phases."""
        # Mock each phase
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        segments = [
            Segment(
                type="static",
                representative_frame=frame,
                last_frame=frame,
                frames=[1, 2],
            )
        ]
        mock_detect.return_value = (segments, segments, 100)

        metadata = [
            {
                "segment_id": 1,
                "start_time": 0.0,
                "end_time": 0.0,
                "duration": 0.0,
                "frame_count": 2,
                "first_frame": {
                    "frame_id": "1-first.webp",
                    "url": "https://blob.vercel-storage.com/url",
                },
                "last_frame": {
                    "frame_id": "1-last.webp",
                    "url": "https://blob.vercel-storage.com/url",
                },
            }
        ]
        mock_upload_segments.return_value = metadata
        mock_upload_to_blob.return_value = "https://blob.vercel-storage.com/manifest"

        # Import here to avoid circular dependency issues
        from slides_extractor.video_service import extract_and_process_frames

        result = await extract_and_process_frames("/tmp/video.mp4", "video-id")

        assert result == metadata
        mock_detect.assert_called_once_with("/tmp/video.mp4", "video-id")
        mock_upload_segments.assert_called_once_with(
            segments,
            "video-id",
            local_output_dir=None,
        )
        # Verify manifest upload
        assert mock_upload_to_blob.called
        assert mock_upload_to_blob.call_args[0][1] == "manifests/video-id"

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

        result = await extract_and_process_frames("/tmp/video.mp4", "video-id")

        assert result == []

        # Should mark as completed
        async with JOBS_LOCK:
            assert JOBS["video-id"]["status"] == JobStatus.completed
            assert "No static segments" in JOBS["video-id"]["message"]


class TestBuildSegmentsManifest:
    """Test manifest construction."""

    def test_manifest_structure_includes_blob_info(self):
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
                "first_frame": {
                    "frame_id": "1-first.webp",
                    "url": "https://blob.vercel-storage.com/1-first.webp",
                },
                "last_frame": {
                    "frame_id": "1-last.webp",
                    "url": "https://blob.vercel-storage.com/1-last.webp",
                },
            }
        ]

        manifest = _build_segments_manifest(video_id, segments, static_metadata)

        assert "updated_at" in manifest[video_id]

        # Verify structure
        assert video_id in manifest
        manifest_segments = manifest[video_id]["segments"]
        assert len(manifest_segments) == 2

        # Check moving segment
        assert manifest_segments[0]["kind"] == "moving"

        # Check static segment
        static_seg = manifest_segments[1]
        assert static_seg["kind"] == "static"
        assert static_seg["first_frame"]["frame_id"] == "1-first.webp"
        assert static_seg["last_frame"]["frame_id"] == "1-last.webp"
        assert static_seg["url"] == "https://blob.vercel-storage.com/1-first.webp"

    def test_manifest_exact_structure_with_multiple_static_and_moving_segments(self):
        """Ensure manifest ordering matches segments and static metadata 1:1."""

        video_id = "vid-123"
        segments = [
            Segment(type="moving", start_time=0.0, end_time=1.0, frames=[]),
            Segment(type="static", start_time=1.0, end_time=2.0, frames=[]),
            Segment(type="moving", start_time=2.0, end_time=3.0, frames=[]),
            Segment(type="static", start_time=3.0, end_time=4.0, frames=[]),
        ]

        static_metadata = [
            {
                "first_frame": {
                    "frame_id": "1-first.webp",
                    "url": "https://blob.vercel-storage.com/1-first.webp",
                },
                "last_frame": {
                    "frame_id": "1-last.webp",
                    "url": "https://blob.vercel-storage.com/1-last.webp",
                },
            },
            {
                "first_frame": {
                    "frame_id": "2-first.webp",
                    "url": "https://blob.vercel-storage.com/2-first.webp",
                },
                "last_frame": {
                    "frame_id": "2-last.webp",
                    "url": "https://blob.vercel-storage.com/2-last.webp",
                },
            },
        ]

        manifest = _build_segments_manifest(video_id, segments, static_metadata)

        expected = {
            video_id: {
                "segments": [
                    {
                        "kind": "moving",
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "duration": 1.0,
                    },
                    {
                        "kind": "static",
                        "start_time": 1.0,
                        "end_time": 2.0,
                        "duration": 1.0,
                        "first_frame": {
                            "frame_id": "1-first.webp",
                            "url": "https://blob.vercel-storage.com/1-first.webp",
                        },
                        "last_frame": {
                            "frame_id": "1-last.webp",
                            "url": "https://blob.vercel-storage.com/1-last.webp",
                        },
                        "url": "https://blob.vercel-storage.com/1-first.webp",
                    },
                    {
                        "kind": "moving",
                        "start_time": 2.0,
                        "end_time": 3.0,
                        "duration": 1.0,
                    },
                    {
                        "kind": "static",
                        "start_time": 3.0,
                        "end_time": 4.0,
                        "duration": 1.0,
                        "first_frame": {
                            "frame_id": "2-first.webp",
                            "url": "https://blob.vercel-storage.com/2-first.webp",
                        },
                        "last_frame": {
                            "frame_id": "2-last.webp",
                            "url": "https://blob.vercel-storage.com/2-last.webp",
                        },
                        "url": "https://blob.vercel-storage.com/2-first.webp",
                    },
                ]
            }
        }

        assert "updated_at" in manifest[video_id]
        manifest_copy = {video_id: dict(manifest[video_id])}
        manifest_copy[video_id].pop("updated_at", None)

        assert manifest_copy == expected


class TestExtractAndProcessFramesManifestUpload:
    """Validate manifest upload behavior during extraction pipeline."""

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
    @patch("slides_extractor.video_service.upload_to_blob")
    @patch("slides_extractor.video_service._upload_segments")
    @patch("slides_extractor.video_service._detect_static_segments")
    async def test_manifest_upload_key_and_job_status(
        self, mock_detect, mock_upload_segments, mock_upload_to_blob
    ):
        """Manifest uploads use expected key and propagate metadata URL."""

        from slides_extractor.video_service import (
            JOBS,
            JOBS_LOCK,
            extract_and_process_frames,
        )

        segments = [Segment(type="static", frames=[0])]
        mock_detect.return_value = (segments, segments, 50)

        mock_upload_segments.return_value = [
            {
                "segment_id": 1,
                "start_time": 0.0,
                "end_time": 1.0,
                "duration": 1.0,
                "frame_count": 1,
                "first_frame": {
                    "frame_id": "1-first.webp",
                    "url": "https://blob.vercel-storage.com/url",
                },
                "last_frame": {
                    "frame_id": "1-last.webp",
                    "url": "https://blob.vercel-storage.com/url",
                },
            }
        ]

        mock_upload_to_blob.return_value = (
            "https://blob.vercel-storage.com/manifests/vid"
        )

        await extract_and_process_frames("/tmp/video.mp4", "vid")

        mock_upload_to_blob.assert_called_once()
        args, _ = mock_upload_to_blob.call_args
        assert args[1] == "manifests/vid"

        async with JOBS_LOCK:
            assert (
                JOBS["vid"]["metadata_uri"]
                == "https://blob.vercel-storage.com/manifests/vid"
            )
            assert JOBS["vid"]["status"] == JobStatus.completed
