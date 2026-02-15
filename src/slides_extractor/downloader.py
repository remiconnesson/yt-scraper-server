import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import requests
from urllib.parse import parse_qs, urlparse

from slides_extractor.job_tracker import remove_progress_entry, update_progress
from slides_extractor.settings import (
    DATACENTER_PROXY,
    DOWNLOAD_DIR,
    DOWNLOAD_RETENTION_HOURS,
    DOWNLOAD_RETENTION_HOURS_RAW,
    YT_COOKIES_PATH,
    YT_DLP_REMOTE_COMPONENTS,
)

logger = logging.getLogger("scraper")


@dataclass
class DownloadResult:
    """Outcome of a download operation."""

    success: bool
    error: Optional[str] = None
    path: Optional[str] = None


def get_file_size(url: str, headers: dict[str, str], proxies: dict[str, str]) -> int:
    """Best-effort probe to determine the size of a remote file."""

    def _size_from_clen(param_url: str) -> Optional[int]:
        query = parse_qs(urlparse(param_url).query)
        raw_clen = query.get("clen", [None])[0]
        if raw_clen is None:
            return None

        try:
            clen_value = int(raw_clen)
            return clen_value if clen_value > 0 else None
        except (TypeError, ValueError):
            logger.debug("Invalid clen value encountered: %s", raw_clen)
            return None

    def _size_from_head() -> Optional[int]:
        try:
            head_resp = requests.head(url, headers=headers, proxies=proxies, timeout=10)
        except requests.RequestException as exc:
            logger.debug("HEAD request failed for %s: %s", url, exc)
            return None

        if head_resp.status_code != 200:
            return None

        try:
            size = int(head_resp.headers.get("content-length", "0"))
        except (TypeError, ValueError):
            logger.debug(
                "Invalid content-length header for %s: %s", url, head_resp.headers
            )
            return None

        return size if size > 0 else None

    def _size_from_range_probe() -> Optional[int]:
        ranged_headers = headers.copy()
        ranged_headers["Range"] = "bytes=0-0"

        try:
            response = requests.get(
                url, headers=ranged_headers, proxies=proxies, timeout=10, stream=True
            )
        except requests.RequestException as exc:
            logger.debug("Range probe failed for %s: %s", url, exc)
            return None

        if response.status_code not in (200, 206):
            return None

        match = re.search(r"/(\d+)$", response.headers.get("Content-Range", ""))
        if not match:
            return None

        try:
            size = int(match.group(1))
        except ValueError:
            logger.debug(
                "Unable to parse Content-Range header for %s: %s", url, response.headers
            )
            return None

        return size if size > 0 else None

    return _size_from_clen(url) or _size_from_head() or _size_from_range_probe() or 0


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_proxy(proxy: str) -> str:
    clean_proxy = proxy.strip()
    if clean_proxy and not re.match(r"^[a-zA-Z]+://", clean_proxy):
        return f"http://{clean_proxy}"
    return clean_proxy


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


def download_youtube_with_ytdlp(
    video: str,
    *,
    cookies_path: str = YT_COOKIES_PATH,
    proxy: str = DATACENTER_PROXY or "",
    remote_components: str = YT_DLP_REMOTE_COMPONENTS,
    format_spec: str = "136",
    filename_prefix: str = "yt",
) -> DownloadResult:
    """Download a YouTube video using yt-dlp CLI via uv."""

    _ensure_dir(DOWNLOAD_DIR)
    normalized_video = video.strip()
    outtmpl = os.path.join(DOWNLOAD_DIR, f"{filename_prefix}_%(id)s.%(ext)s")
    normalized_proxy = _normalize_proxy(proxy)

    shared_args = [
        "-f",
        format_spec,
        "--cookies",
        cookies_path,
        "--remote-components",
        remote_components,
        "-o",
        outtmpl,
    ]
    if normalized_proxy:
        shared_args.extend(["--proxy", normalized_proxy])

    get_filename_cmd = [
        "uv",
        "run",
        "yt-dlp",
        "--get-filename",
        normalized_video,
        *shared_args,
    ]

    try:
        filename_output = subprocess.check_output(
            get_filename_cmd,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr_output = exc.stderr[-4000:] if isinstance(exc.stderr, str) else str(exc)
        logger.error(
            "yt-dlp failed to get filename (%s): %s", exc.returncode, stderr_output
        )
        return DownloadResult(
            False,
            error=f"Failed to determine filename with yt-dlp: exit {exc.returncode}",
        )
    except OSError as exc:
        logger.error("yt-dlp failed to get filename: %s", exc)
        return DownloadResult(
            False, error=f"Failed to determine filename with yt-dlp: {exc}"
        )

    final_path = (
        filename_output.strip().splitlines()[-1] if filename_output.strip() else ""
    )
    if not final_path:
        return DownloadResult(False, error="Failed to determine filename with yt-dlp")

    tracking_name = os.path.basename(final_path)
    update_progress(tracking_name, status="downloading")
    logger.info("yt-dlp download start: %s -> %s", normalized_video, tracking_name)

    cmd = [
        "uv",
        "run",
        "yt-dlp",
        "-v",
        normalized_video,
        *shared_args,
        "--no-part",
        "--retries",
        "5",
        "--fragment-retries",
        "5",
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError as exc:
        update_progress(tracking_name, status="failed")
        return DownloadResult(False, error=f"Failed to start yt-dlp: {exc}")

    if proc.returncode != 0:
        logger.error("yt-dlp failed (%s):\n%s", proc.returncode, proc.stdout[-4000:])
        update_progress(tracking_name, status="failed")
        return DownloadResult(False, error=f"yt-dlp exited with {proc.returncode}")

    if not os.path.exists(final_path):
        update_progress(tracking_name, status="failed")
        logger.error(
            "yt-dlp succeeded but output file not found: %s\n%s",
            final_path,
            proc.stdout[-4000:],
        )
        return DownloadResult(False, error="yt-dlp succeeded but output file not found")

    update_progress(tracking_name, status="complete")
    logger.info("yt-dlp saved: %s", final_path)
    return DownloadResult(True, path=final_path)
