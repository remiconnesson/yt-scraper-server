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
async def test_full_pipeline_with_test_bucket():
    """Test the full pipeline using a configured test S3 bucket."""
    # NOTE: This test requires AWS credentials to be available in the environment
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, "test_video.mp4")
        create_test_video(video_path, duration_seconds=20, fps=5)

        video_id = "test_video_001"

        # Patch the bucket name to use 'test_bucket' (or a specific test bucket name you use)
        # In a real scenario, you might use moto to mock S3 entirely, or a dedicated test bucket.
        # Since the user asked to use the 'test_bucket', we patch it here.
        with (
            patch(
                "slides_extractor.settings.S3_BUCKET_NAME",
                "test-bucket-slides-extractor",
            ),
            patch(
                "slides_extractor.video_service.S3_BUCKET_NAME",
                "test-bucket-slides-extractor",
            ),
        ):
            # We also need to ensure S3_ACCESS_KEY and S3_ENDPOINT are set, or mock them if we want to avoid real calls.
            # If we want real integration tests, we assume env vars are set.
            # If we want to simulate success without real S3, we might need to mock upload_to_s3 as well,
            # but the user asked for an "actual integration test".
            # However, without actual credentials, this might fail in this environment.
            # I will assume the environment has credentials or the user accepts failure if not.
            # But strictly speaking, if we want to ensure it passes in a CI without creds, we might need more mocks.
            # For now I follow the user request to use "test_bucket".

            # Check if we have creds, if not, maybe skip or warn?
            if not os.getenv("S3_ACCESS_KEY"):
                pytest.skip("S3_ACCESS_KEY not set, skipping real S3 integration test")

            try:
                metadata = await extract_and_process_frames(video_path, video_id)

                assert len(metadata) == 3
                for segment in metadata:
                    assert segment["first_frame"]["s3_bucket"] == "test-bucket-slides-extractor"
                    assert segment["first_frame"]["url"].startswith("s3://")

            except Exception as e:
                pytest.fail(f"Pipeline failed (check S3 creds?): {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_local_save():
    """Test the full pipeline saving to local disk (no S3)."""
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
        assert os.path.exists(
            os.path.join(output_dir, "video", video_id, "video_segments.json")
        )
        assert os.path.exists(
            os.path.join(
                output_dir,
                "video",
                video_id,
                "static_frames",
                "static_frame_000001_first.webp",
            )
        )
