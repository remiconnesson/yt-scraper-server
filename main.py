import asyncio
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Coroutine, Dict, Optional, TypeVar
from urllib.parse import parse_qs, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yt_dlp
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI

# Load environment variables
load_dotenv()

app = FastAPI(title="Turbo Scraper (VPS Edition)")

# Setup storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
LOG_FILE = os.path.join(BASE_DIR, "app.log")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# --- GLOBAL STATE ---
JOB_PROGRESS = {}
PROGRESS_LOCK: asyncio.Lock | None = None
EVENT_LOOP: asyncio.AbstractEventLoop | None = None

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("scraper")

T = TypeVar("T")


def _parse_retention_hours(raw_value: Optional[str], default: int = 24) -> int:
    """Parse the retention env var defensively, returning a fallback on error."""

    if raw_value is None:
        return default

    normalized = raw_value.strip()
    if normalized == "":
        logger.warning(
            "Invalid DOWNLOAD_RETENTION_HOURS value '%s'; using default %s (empty string)",
            raw_value,
            default,
        )
        return default

    try:
        return int(normalized)
    except ValueError as exc:
        logger.warning(
            "Invalid DOWNLOAD_RETENTION_HOURS value '%s'; using default %s (%s)",
            raw_value,
            default,
            exc,
        )
        return default


# --- CONFIGURATION ---
ZYTE_API_KEY = os.getenv("ZYTE_API_KEY")
ZYTE_HOST = os.getenv("ZYTE_HOST", "api.zyte.com")
DATACENTER_PROXY = os.getenv("DATACENTER_PROXY")

MIN_SIZE_FOR_PARALLEL_DOWNLOAD = 1 * 1024 * 1024  # 1MB
VIDEO_DOWNLOAD_THREADS = 32
AUDIO_DOWNLOAD_THREADS = 8
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
PARALLEL_CHUNK_SIZE = 1024 * 1024
SINGLE_THREAD_CHUNK_SIZE = 32 * 1024
DEFAULT_RETENTION_HOURS = 24
DOWNLOAD_RETENTION_HOURS_RAW = os.getenv("DOWNLOAD_RETENTION_HOURS")
DOWNLOAD_RETENTION_HOURS = _parse_retention_hours(
    DOWNLOAD_RETENTION_HOURS_RAW, DEFAULT_RETENTION_HOURS
)

# --- HELPERS ---


@dataclass
class DownloadResult:
    """Outcome of a download operation."""

    success: bool
    error: Optional[str] = None
    path: Optional[str] = None


async def _ensure_progress_lock() -> asyncio.Lock:
    """Lazy-create the asyncio lock after the event loop is running."""

    global PROGRESS_LOCK
    if PROGRESS_LOCK is None:
        PROGRESS_LOCK = asyncio.Lock()
    return PROGRESS_LOCK


async def _update_progress(
    filename: str,
    bytes_added: int = 0,
    total_size: Optional[int] = None,
    status: Optional[str] = None,
) -> None:
    lock = await _ensure_progress_lock()
    async with lock:
        if filename not in JOB_PROGRESS:
            JOB_PROGRESS[filename] = {
                "total": 0,
                "current": 0,
                "status": "init",
                "start_time": time.time(),
            }

        if total_size is not None:
            JOB_PROGRESS[filename]["total"] = total_size

        if bytes_added:
            JOB_PROGRESS[filename]["current"] += bytes_added

        if status:
            JOB_PROGRESS[filename]["status"] = status


async def _remove_progress_entry(filename: str) -> None:
    """Drop a progress entry if it exists, guarding with the async lock."""

    lock = await _ensure_progress_lock()
    async with lock:
        JOB_PROGRESS.pop(filename, None)


def _ensure_sync_entrypoint(sync_only_message: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    else:
        raise RuntimeError(sync_only_message)


def _run_in_event_loop(coro: Coroutine[Any, Any, T], async_use_hint: str) -> T:
    """Bridge a coroutine into the app event loop from sync contexts."""

    _ensure_sync_entrypoint(async_use_hint)

    if EVENT_LOOP and EVENT_LOOP.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, EVENT_LOOP)
        return future.result()

    raise RuntimeError(
        "Event loop not initialized; progress updates require FastAPI startup to complete."
    )


def update_progress(
    filename: str,
    bytes_added: int = 0,
    total_size: Optional[int] = None,
    status: Optional[str] = None,
) -> None:
    """Thread-safe wrapper to update download progress from sync code paths."""

    _run_in_event_loop(
        _update_progress(
            filename, bytes_added=bytes_added, total_size=total_size, status=status
        ),
        "update_progress() cannot be used from async contexts; call `_update_progress` directly instead.",
    )


def remove_progress_entry(filename: str) -> None:
    """Thread-safe wrapper to prune a progress record from sync code paths."""

    try:
        _run_in_event_loop(
            _remove_progress_entry(filename),
            "remove_progress_entry() cannot be used from async contexts; call `_remove_progress_entry` directly instead.",
        )
    except RuntimeError as exc:
        logger.warning("%s", exc)


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

    return (
        _size_from_clen(url)
        or _size_from_head()
        or _size_from_range_probe()
        or 0
    )


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
                # TODO: this should be a constant at least MAX_HEIGHT
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

    except Exception as e:
        logger.error(f"Phase A Failed: {e}")
        return None, None, None


def _get_proxy_config() -> Dict[str, str]:
    """Normalize proxy configuration from environment variables.

    Note: I'm not convinced of the usefulness of this function.
    """

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
    """Delete downloaded files older than the configured retention window.

    Progress entries for removed files are pruned to keep API responses aligned
    with available downloads.
    """

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


# download_chunk, _download_chunks_parallel, _merge_chunks, download_file_parallel, download_file_single
# should be extracted to a separate module
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
        with requests.get(
            url, headers=headers, proxies=proxies, stream=True, timeout=(20, 120)
        ) as r:
            r.raise_for_status()
            with open(part_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=PARALLEL_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        update_progress(parent_filename, bytes_added=len(chunk))
        return True
    except Exception as e:
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


def download_file_parallel(
    url: str, filename: str, num_threads: int = VIDEO_DOWNLOAD_THREADS
) -> DownloadResult:
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

    temp_parts = _download_chunks_parallel(
        url, filename, total_size, num_threads, proxies, headers
    )
    if temp_parts is None:
        return DownloadResult(success=False, error="Parallel download failed")

    _merge_chunks(temp_parts, path)
    update_progress(filename, status="complete")
    logger.info(f"SAVED: {filename}")
    return DownloadResult(success=True, path=path)


def download_file_single(
    url: str, filename: str, proxies: Dict[str, str]
) -> DownloadResult:
    path = os.path.join(DOWNLOAD_DIR, filename)
    headers = _get_default_headers()

    logger.info(f"SINGLE THREAD START: {filename}")
    try:
        with requests.get(
            url, headers=headers, proxies=proxies, stream=True, timeout=(20, 120)
        ) as r:
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
    except Exception as e:
        logger.error(f"Single Download Failed: {e}")
        update_progress(filename, status="failed")
        return DownloadResult(success=False, error=str(e))


@app.on_event("startup")
async def on_startup() -> None:
    """Capture the running event loop and trigger cleanup tasks."""

    global EVENT_LOOP
    EVENT_LOOP = asyncio.get_running_loop()
    # TODO: potential bug: the job thing is in memory, so will the remove_progress function work on startup?
    await asyncio.to_thread(cleanup_old_downloads)


# TODO: Let's split download video and download audio, because for now we don't need the audio
def process_video_task(video_url: str):
    logger.info(f"Job Started: {video_url}")
    try:
        vid_url, aud_url, title = get_stream_urls(video_url)

        if vid_url and aud_url:
            safe_title = "".join(
                [c for c in title if c.isalpha() or c.isdigit() or c == " "]
            ).rstrip()

            # CONCURRENT DOWNLOADS
            # We use a ThreadPool to run both downloads at the exact same time
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both jobs
                # Video: 32 threads for max saturation
                video_future = executor.submit(
                    download_file_parallel,
                    vid_url,
                    f"{safe_title}_video.mp4",
                    num_threads=VIDEO_DOWNLOAD_THREADS,
                )
                # Audio: 8 threads (plenty for small files)
                audio_future = executor.submit(
                    download_file_parallel,
                    aud_url,
                    f"{safe_title}_audio.m4a",
                    num_threads=AUDIO_DOWNLOAD_THREADS,
                )

                # Wait for completion
                video_result = video_future.result()
                audio_result = audio_future.result()

            for kind, result in {"video": video_result, "audio": audio_result}.items():
                if not result.success:
                    logger.error(f"{kind.title()} download failed: {result.error}")
                    return

            logger.info(f"Job Finished: {safe_title}")
        else:
            logger.error("Job Failed during Phase A")
    finally:
        # TODO: this should not be done here... We need to move this to a cron job.
        # We don't need the try catch block and this should be moved in the orchestrator function
        cleanup_old_downloads()


# --- ROUTES ---


@app.get("/")
def home():
    return {
        "status": "Running on VPS",
        "endpoints": {
            "start": "/scrape?url=...",
            "progress": "/progress",
            "logs": "/logs",
            "files": "/list",
        },
    }


@app.get("/scrape")
def scrape(url: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_video_task, url)
    return {"message": "Download started", "url": url, "track": "/progress"}


@app.get("/progress")
async def get_progress():
    results = {}
    lock = await _ensure_progress_lock()
    async with lock:
        for filename, data in JOB_PROGRESS.items():
            pct = 0
            if data["total"] > 0:
                pct = (data["current"] / data["total"]) * 100

            results[filename] = {
                "status": data["status"],
                "percent": round(pct, 1),
                "downloaded_mb": round(data["current"] / (1024 * 1024), 1),
                "total_mb": round(data["total"] / (1024 * 1024), 1),
            }
    return results


@app.get("/logs")
def view_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            return {"recent_logs": f.readlines()[-50:][::-1]}
    return {"error": "Log file empty or missing"}


@app.get("/list")
def list_files():
    files = os.listdir(DOWNLOAD_DIR)
    data = [{"filename": f, "url": f"/files/{f}"} for f in files]
    return {"files": data}
