import asyncio
import logging
from typing import Optional

from slides_extractor.downloader import (
    cleanup_old_downloads,
    download_youtube_with_ytdlp,
)
from slides_extractor.video_service import (
    JobStatus,
    extract_and_process_frames,
    update_job_status,
)

logger = logging.getLogger("scraper")


def process_video_task(
    video_url: str, video_id: str, local_output_dir: Optional[str] = None
) -> None:
    asyncio.run(
        update_job_status(
            video_id,
            JobStatus.pending,
            0.0,
            "Processing started",
        )
    )
    logger.info("Job Started: %s", video_url)
    try:
        video_result = download_youtube_with_ytdlp(video_url, filename_prefix=video_id)

        if not video_result.success:
            logger.error("Video download failed: %s", video_result.error)
            try:
                asyncio.run(
                    update_job_status(
                        video_id,
                        JobStatus.failed,
                        0.0,
                        "Video download failed",
                        error=str(video_result.error),
                    )
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Unable to record failure status for %s (video_id=%s)",
                    video_url,
                    video_id,
                )
            return

        logger.info("Starting slide extraction...")
        video_path = video_result.path
        if video_path is None:
            raise RuntimeError("Download succeeded but no output path was returned")

        try:
            asyncio.run(
                extract_and_process_frames(
                    video_path=video_path,
                    video_id=video_id,
                    local_output_dir=local_output_dir,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Job Failed during slide extraction for %s (video_id=%s)",
                video_url,
                video_id,
            )
            try:
                asyncio.run(
                    update_job_status(
                        video_id,
                        JobStatus.failed,
                        0.0,
                        "Slide extraction failed",
                        error=str(exc),
                    )
                )
            except Exception as status_exc:  # noqa: BLE001
                logger.error(
                    "Unable to record failure status for %s (video_id=%s): %s",
                    video_url,
                    video_id,
                    status_exc,
                )
            return

        logger.info("Job Finished: %s", video_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Job Failed prior to slide extraction for %s (video_id=%s)",
            video_url,
            video_id,
        )
        try:
            asyncio.run(
                update_job_status(
                    video_id,
                    JobStatus.failed,
                    0.0,
                    "Job failed before slide extraction",
                    error=str(exc),
                )
            )
        except Exception as status_exc:  # noqa: BLE001
            logger.error(
                "Unable to record failure status for %s (video_id=%s): %s",
                video_url,
                video_id,
                status_exc,
            )
    finally:
        cleanup_old_downloads()
