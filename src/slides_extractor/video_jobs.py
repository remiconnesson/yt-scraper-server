import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

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


def process_video_task(video_url: str, video_id: str, job_id: str) -> None:
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
                    return

            logger.info("Starting slide extraction...")
            video_path = video_result.path or f"{safe_title}_video.mp4"
            try:
                asyncio.run(
                    extract_and_process_frames(
                        video_path=video_path,
                        video_id=video_id,
                        job_id=job_id,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Job Failed during slide extraction for %s (job_id=%s)",
                    video_url,
                    job_id,
                )
                try:
                    asyncio.run(
                        update_job_status(
                            job_id,
                            JobStatus.failed,
                            0.0,
                            "Slide extraction failed",
                            error=str(exc),
                        )
                    )
                except Exception as status_exc:  # noqa: BLE001
                    logger.error(
                        "Unable to record failure status for %s (job_id=%s): %s",
                        video_url,
                        job_id,
                        status_exc,
                    )
                return

            logger.info(f"Job Finished: {safe_title}")
        else:
            logger.error("Job Failed during Phase A")
    finally:
        cleanup_old_downloads()
