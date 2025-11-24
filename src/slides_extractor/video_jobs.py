import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from slides_extractor.downloader import (
    VIDEO_DOWNLOAD_THREADS,
    cleanup_old_downloads,
    download_file_parallel,
    get_stream_urls,
)
from slides_extractor.video_service import (
    JobStatus,
    extract_and_process_frames,
    update_job_status,
)

logger = logging.getLogger("scraper")


def _safe_title(title: str) -> str:
    return "".join(
        [c for c in title if c.isalpha() or c.isdigit() or c == " "]
    ).rstrip()


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
    logger.info(f"Job Started: {video_url}")
    try:
        vid_url, _, title = get_stream_urls(video_url)

        if vid_url:
            safe_title = _safe_title(title or "video")

            with ThreadPoolExecutor(max_workers=2) as executor:
                video_future = executor.submit(
                    download_file_parallel,
                    vid_url,
                    f"{safe_title}_video.mp4",
                    num_threads=VIDEO_DOWNLOAD_THREADS,
                )

                video_result = video_future.result()

            for kind, result in {"video": video_result}.items():
                if not result.success:
                    logger.error(f"{kind.title()} download failed: {result.error}")
                    try:
                        asyncio.run(
                            update_job_status(
                                video_id,
                                JobStatus.failed,
                                0.0,
                                "Video download failed",
                                error=str(result.error),
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
            video_path = video_result.path or f"{safe_title}_video.mp4"
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

            logger.info(f"Job Finished: {safe_title}")
        else:
            logger.error("Job Failed during Phase A (no video stream URL)")
            try:
                asyncio.run(
                        update_job_status(
                            video_id,
                            JobStatus.failed,
                            0.0,
                            "Unable to resolve video stream URL",
                        )
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Unable to record failure status for %s (video_id=%s)",
                    video_url,
                    video_id,
                )
            return
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
