import logging
from concurrent.futures import ThreadPoolExecutor

from slides_extractor.downloader import (
    VIDEO_DOWNLOAD_THREADS,
    cleanup_old_downloads,
    download_file_parallel,
    get_stream_urls,
)

logger = logging.getLogger("scraper")


def _safe_title(title: str) -> str:
    return "".join(
        [c for c in title if c.isalpha() or c.isdigit() or c == " "]
    ).rstrip()


def process_video_task(video_url: str):
    logger.info(f"Job Started: {video_url}")
    try:
        vid_url, _, title = get_stream_urls(video_url)

        if vid_url:
            safe_title = _safe_title(title)

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

            logger.info(f"Job Finished: {safe_title}")
        else:
            logger.error("Job Failed during Phase A")
    finally:
        cleanup_old_downloads()
