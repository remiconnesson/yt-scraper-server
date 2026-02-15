import logging
import os
import re
import subprocess
import time
from typing import Optional

from slides_extractor.job_tracker import remove_progress_entry, update_progress
from slides_extractor.settings import (
    DATACENTER_PROXY,
    DOWNLOAD_DIR,
    DOWNLOAD_RETENTION_HOURS,
    DOWNLOAD_RETENTION_HOURS_RAW,
    YT_COOKIES_PATH,
    YT_FORMAT_SPEC,
    YT_REMOTE_COMPONENTS,
)

logger = logging.getLogger("scraper")


class DownloadResult:
    """Outcome of a download operation."""

    def __init__(
        self, success: bool, error: Optional[str] = None, path: Optional[str] = None
    ):
        self.success = success
        self.error = error
        self.path = path


def cleanup_old_downloads(retention_hours: int = DOWNLOAD_RETENTION_HOURS) -> None:
    """Delete downloaded files older than the configured retention window."""

    cutoff = time.time() - (retention_hours * 3600)
    for filename in os.listdir(DOWNLOAD_DIR):
        path = os.path.join(DOWNLOAD_DIR, filename)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                remove_progress_entry(filename)
                logger.info("Removed expired download: %s", filename)
        except OSError as exc:
            logger.warning(
                "Failed to remove %s during cleanup (retention=%s, env='%s'): %s",
                filename,
                retention_hours,
                DOWNLOAD_RETENTION_HOURS_RAW,
                exc,
            )


def _normalize_proxy(proxy: Optional[str]) -> Optional[str]:
    """Ensure the proxy string includes a scheme prefix."""
    if not proxy or len(proxy) <= 5:
        return None
    clean = proxy.strip()
    if not re.match(r"^[a-zA-Z]+://", clean):
        clean = f"http://{clean}"
    return clean


def download_video_with_ytdlp(
    video_url: str,
    filename_prefix: str = "yt",
) -> DownloadResult:
    """Download a YouTube video using yt-dlp CLI (via ``uv run``).

    Uses cookies, remote-components, and proxy settings from environment
    configuration to handle authenticated / geo-restricted downloads
    end-to-end, avoiding the 403 chunk failures that occur when
    downloading signed googlevideo URLs separately with ``requests``.
    """

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Deterministic output template so we can locate the file afterwards.
    outtmpl = os.path.join(DOWNLOAD_DIR, f"{filename_prefix}_%(id)s.%(ext)s")

    cmd = [
        "uv",
        "run",
        "yt-dlp",
        "-v",
        video_url,
        "-f",
        YT_FORMAT_SPEC,
        "--cookies",
        YT_COOKIES_PATH,
        "--remote-components",
        YT_REMOTE_COMPONENTS,
        "--no-part",
        "--retries",
        "5",
        "--fragment-retries",
        "5",
        "-o",
        outtmpl,
    ]

    proxy = _normalize_proxy(DATACENTER_PROXY)
    if proxy:
        cmd.extend(["--proxy", proxy])

    progress_key = os.path.basename(outtmpl)
    logger.info("yt-dlp download start: %s", video_url)
    update_progress(progress_key, status="downloading")

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError as exc:
        update_progress(progress_key, status="failed")
        return DownloadResult(False, error=f"Failed to start yt-dlp: {exc}")

    if proc.returncode != 0:
        logger.error("yt-dlp failed (%s):\n%s", proc.returncode, proc.stdout[-4000:])
        update_progress(progress_key, status="failed")
        return DownloadResult(False, error=f"yt-dlp exited with {proc.returncode}")

    # Find the actual produced file.
    # yt-dlp prints lines like: "Destination: /path/to/file.ext"
    m = re.findall(r"Destination:\s+(.*)$", proc.stdout, flags=re.MULTILINE)
    if m:
        path = m[-1].strip()
    else:
        # Fallback: scan DOWNLOAD_DIR for matching prefix and most recent file
        candidates = [
            os.path.join(DOWNLOAD_DIR, f)
            for f in os.listdir(DOWNLOAD_DIR)
            if f.startswith(f"{filename_prefix}_")
        ]
        path = max(candidates, key=os.path.getmtime) if candidates else None

    if not path or not os.path.exists(path):
        update_progress(progress_key, status="failed")
        return DownloadResult(False, error="yt-dlp succeeded but output file not found")

    update_progress(os.path.basename(path), status="complete")
    logger.info("yt-dlp saved: %s", path)
    return DownloadResult(True, path=path)
