import logging
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

import requests
import yt_dlp

from job_tracker import remove_progress_entry, update_progress
from settings import (
    AUDIO_DOWNLOAD_THREADS,
    DATACENTER_PROXY,
    DEFAULT_USER_AGENT,
    DOWNLOAD_DIR,
    DOWNLOAD_RETENTION_HOURS,
    DOWNLOAD_RETENTION_HOURS_RAW,
    LOG_FILE,
    MIN_SIZE_FOR_PARALLEL_DOWNLOAD,
    PARALLEL_CHUNK_SIZE,
    SINGLE_THREAD_CHUNK_SIZE,
    VIDEO_DOWNLOAD_THREADS,
    ZYTE_API_KEY,
    ZYTE_HOST,
)

logger = logging.getLogger("scraper")


class DownloadResult:
    """Outcome of a download operation."""

    def __init__(self, success: bool, error: Optional[str] = None, path: Optional[str] = None):
        self.success = success
        self.error = error
        self.path = path


def get_file_size(url: str, headers: Dict[str, str], proxies: Dict[str, str]) -> int:
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
            logger.debug("Invalid content-length header for %s: %s", url, head_resp.headers)
            return None

        return size if size > 0 else None

    def _size_from_range_probe() -> Optional[int]:
        ranged_headers = headers.copy()
        ranged_headers["Range"] = "bytes=0-0"

        try:
            response = requests.get(url, headers=ranged_headers, proxies=proxies, timeout=10, stream=True)
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
            logger.debug("Unable to parse Content-Range header for %s: %s", url, response.headers)
            return None

        return size if size > 0 else None

    return _size_from_clen(url) or _size_from_head() or _size_from_range_probe() or 0


def get_stream_urls(video_url):
    if not ZYTE_API_KEY:
        logger.error("CRITICAL: ZYTE_API_KEY is missing")
        return None, None, None

    zyte_proxy = f"http://{ZYTE_API_KEY.strip()}:@{ZYTE_HOST}:8011/"

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "proxy": zyte_proxy,
    }

    try:
        logger.info(f"Phase A: Fetching metadata for {video_url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            title = info.get("title", "video")
            formats = info.get("formats", [])

            # Video: Max 1000p (so 720p or 480p, but NOT 1080p)
            # If you want 1080p, change 1000 to 1080.
            video_streams = [
                f
                for f in formats
                if f.get("vcodec") != "none"
                and f.get("acodec") == "none"
                and f.get("protocol", "").startswith("http")
                and f.get("height", 0) <= 1000
            ]
            video_streams.sort(key=lambda x: x.get("height", 0), reverse=True)

            # Audio: Best Available
            audio_streams = [
                f
                for f in formats
                if f.get("acodec") != "none"
                and f.get("vcodec") == "none"
                and f.get("protocol", "").startswith("http")
            ]
            audio_streams.sort(key=lambda x: x.get("abr", 0), reverse=True)

            if video_streams and audio_streams:
                v_res = video_streams[0].get("height")
                logger.info(f"Metadata Success: {v_res}p | Title: {title[:30]}...")
                return video_streams[0]["url"], audio_streams[0]["url"], title

            return None, None, None

    except Exception as e:  # pragma: no cover - integration with yt-dlp is best-effort
        logger.error(f"Phase A Failed: {e}")
        return None, None, None


def _get_proxy_config() -> Dict[str, str]:
    """Normalize proxy configuration from environment variables."""

    if DATACENTER_PROXY and len(DATACENTER_PROXY) > 5:
        clean_proxy = DATACENTER_PROXY.strip()
        if not clean_proxy.startswith("http"):
            clean_proxy = f"http://{clean_proxy}"
        return {"http": clean_proxy, "https": clean_proxy}
    return {}


def _get_default_headers() -> Dict[str, str]:
    return {"User-Agent": DEFAULT_USER_AGENT}


def _should_use_single_thread(total_size: int) -> bool:
    return total_size < MIN_SIZE_FOR_PARALLEL_DOWNLOAD


def cleanup_old_downloads(retention_hours: int = DOWNLOAD_RETENTION_HOURS) -> None:
    """Delete downloaded files older than the configured retention window."""

    cutoff = time.time() - (retention_hours * 3600)
    for filename in os.listdir(DOWNLOAD_DIR):
        path = os.path.join(DOWNLOAD_DIR, filename)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                remove_progress_entry(filename)
                logger.info(f"Removed expired download: {filename}")
        except OSError as exc:
            logger.warning(
                "Failed to remove %s during cleanup (retention=%s, env='%s'): %s",
                filename,
                retention_hours,
                DOWNLOAD_RETENTION_HOURS_RAW,
                exc,
            )


def download_chunk(
    url: str,
    headers: Dict[str, str],
    start: int,
    end: int,
    part_filename: str,
    proxies: Dict[str, str],
    parent_filename: str,
) -> bool:
    headers["Range"] = f"bytes={start}-{end}"
    try:
        with requests.get(url, headers=headers, proxies=proxies, stream=True, timeout=(20, 120)) as r:
            r.raise_for_status()
            with open(part_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=PARALLEL_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        update_progress(parent_filename, bytes_added=len(chunk))
        return True
    except Exception as e:  # pragma: no cover - network failure is non-deterministic
        logger.error(f"Chunk Fail {start}-{end}: {e}")
        return False


def _download_chunks_parallel(
    url: str,
    filename: str,
    total_size: int,
    num_threads: int,
    proxies: Dict[str, str],
    headers: Dict[str, str],
) -> Optional[list[str]]:
    chunk_size = total_size // num_threads
    path = os.path.join(DOWNLOAD_DIR, filename)
    futures = []
    temp_parts = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start = i * chunk_size
            end = start + chunk_size - 1 if i < num_threads - 1 else total_size - 1
            part_name = f"{path}.part{i}"
            temp_parts.append(part_name)
            futures.append(
                executor.submit(
                    download_chunk,
                    url,
                    headers.copy(),
                    start,
                    end,
                    part_name,
                    proxies,
                    filename,
                )
            )

        for future in as_completed(futures):
            if not future.result():
                update_progress(filename, status="failed")
                return None

    return temp_parts


def _merge_chunks(temp_parts: list[str], target_path: str) -> None:
    update_progress(os.path.basename(target_path), status="merging")
    logger.info(f"Merging chunks for {os.path.basename(target_path)}...")
    with open(target_path, "wb") as outfile:
        for part in temp_parts:
            if os.path.exists(part):
                with open(part, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)
                os.remove(part)


def download_file_parallel(url: str, filename: str, num_threads: int = VIDEO_DOWNLOAD_THREADS) -> DownloadResult:
    """Download a file using multiple threads, falling back to single-threaded mode."""

    path = os.path.join(DOWNLOAD_DIR, filename)
    proxies = _get_proxy_config()
    connection_mode = "Proxy" if proxies else "Direct"
    logger.info(f"Using {connection_mode} connection for {filename}")

    headers = _get_default_headers()
    total_size = get_file_size(url, headers, proxies)

    if _should_use_single_thread(total_size):
        logger.warning(f"Size {total_size} too small/unknown. Using Single Thread.")
        return download_file_single(url, filename, proxies)

    update_progress(filename, total_size=total_size, status="downloading")
    logger.info(
        f"PARALLEL START: {filename} ({total_size / (1024 * 1024):.1f} MB) | {num_threads} Threads"
    )

    temp_parts = _download_chunks_parallel(url, filename, total_size, num_threads, proxies, headers)
    if temp_parts is None:
        return DownloadResult(success=False, error="Parallel download failed")

    _merge_chunks(temp_parts, path)
    update_progress(filename, status="complete")
    logger.info(f"SAVED: {filename}")
    return DownloadResult(success=True, path=path)


def download_file_single(url: str, filename: str, proxies: Dict[str, str]) -> DownloadResult:
    path = os.path.join(DOWNLOAD_DIR, filename)
    headers = _get_default_headers()

    logger.info(f"SINGLE THREAD START: {filename}")
    try:
        with requests.get(url, headers=headers, proxies=proxies, stream=True, timeout=(20, 120)) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            update_progress(filename, total_size=total_size, status="downloading")

            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=SINGLE_THREAD_CHUNK_SIZE):
                    f.write(chunk)
                    update_progress(filename, bytes_added=len(chunk))

        update_progress(filename, status="complete")
        logger.info(f"SAVED Single: {filename}")
        return DownloadResult(success=True, path=path)
    except Exception as e:  # pragma: no cover - network failure is non-deterministic
        logger.error(f"Single Download Failed: {e}")
        update_progress(filename, status="failed")
        return DownloadResult(success=False, error=str(e))

