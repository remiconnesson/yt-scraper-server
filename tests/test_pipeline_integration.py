import os
import tempfile
import pytest
from unittest.mock import patch
from slides_extractor.extract_slides.video_analyzer import analyze_video, AnalysisConfig
from slides_extractor.video_service import extract_and_process_frames
from tests.utils import create_test_video


@pytest.mark.integration
def test_video_analysis_only():
    """Test just the analysis part without S3 upload (unit test style)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "test_video.mp4")
        # Increase duration to 20s (4s per segment) to ensure static segments > 3 frames at 1 FPS analysis
        create_test_video(video_path, duration_seconds=20, fps=5)

        config: AnalysisConfig = {
            "grid_cols": 4,
            "grid_rows": 4,
            "cell_hash_threshold": 5,
            "min_static_cell_ratio": 0.8,
            "min_static_frames": 3,
        }

        result = analyze_video(video_path, config)

        assert len(result.segments) > 0
        assert len(result.static_segments) == 3  # Red, Blue, Green
        assert len(result.moving_segments) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_with_test_blob():
    """Test the full pipeline using a configured test Vercel Blob token."""
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "test_video.mp4")
        create_test_video(video_path, duration_seconds=20, fps=5)

        video_id = "test_video_001"

        with (
            patch(
                "slides_extractor.settings.BLOB_READ_WRITE_TOKEN",
                "test-blob-token",
            ),
        ):
            if not os.getenv("BLOB_READ_WRITE_TOKEN"):
                pytest.skip(
                    "BLOB_READ_WRITE_TOKEN not set, skipping real Blob integration test"
                )

            try:
                metadata = await extract_and_process_frames(video_path, video_id)

                assert len(metadata) == 3
                for segment in metadata:
                    assert "vercel-storage.com" in segment["first_frame"]["url"]

            except Exception as e:
                pytest.fail(f"Pipeline failed (check Blob token?): {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_local_save():
    """Test the full pipeline saving to local disk (no Blob)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "test_video.mp4")
        output_dir = os.path.join(temp_dir, "output")
        create_test_video(video_path, duration_seconds=20, fps=5)

        video_id = "test_video_local"

        metadata = await extract_and_process_frames(
            video_path, video_id, local_output_dir=output_dir
        )

        assert len(metadata) == 3
        # Check files exist on disk
        assert os.path.exists(os.path.join(output_dir, "manifests", video_id))
        assert os.path.exists(
            os.path.join(
                output_dir,
                "slides",
                video_id,
                "1-first.webp",
            )
        )
