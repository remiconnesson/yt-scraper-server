import logging
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yt_dlp
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

app = FastAPI(title="Turbo Scraper (VPS Edition)")

# Setup storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
LOG_FILE = os.path.join(BASE_DIR, "app.log")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

app.mount("/files", StaticFiles(directory=DOWNLOAD_DIR), name="files")

# --- GLOBAL STATE ---
JOB_PROGRESS = {}
PROGRESS_LOCK = threading.Lock()

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("scraper")

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

# --- HELPERS ---


@dataclass
class DownloadResult:
    """Outcome of a download operation."""

    success: bool
    error: Optional[str] = None
    path: Optional[str] = None


def update_progress(filename: str, bytes_added: int = 0, total_size: Optional[int] = None, status: Optional[str] = None) -> None:
    """Update the in-memory progress tracker for a file download."""

    with PROGRESS_LOCK:
        if filename not in JOB_PROGRESS:
            JOB_PROGRESS[filename] = {"total": 0, "current": 0, "status": "init", "start_time": time.time()}

        if total_size is not None:
            JOB_PROGRESS[filename]["total"] = total_size

        if bytes_added:
            JOB_PROGRESS[filename]["current"] += bytes_added

        if status:
            JOB_PROGRESS[filename]["status"] = status


def get_file_size(url: str, headers: Dict[str, str], proxies: Dict[str, str]) -> int:
    # Method 1: URL 'clen' parameter
    if "clen=" in url:
        try:
            query = parse_qs(urlparse(url).query)
            if 'clen' in query:
                size = int(query['clen'][0])
                return size
        except:
            pass

    # Method 2: HEAD Request
    try:
        head_resp = requests.head(url, headers=headers, proxies=proxies, timeout=10)
        if head_resp.status_code == 200:
            size = int(head_resp.headers.get('content-length', 0))
            if size > 0: return size
    except:
        pass

    # Method 3: Range Probe
    try:
        h = headers.copy()
        h['Range'] = 'bytes=0-0'
        r = requests.get(url, headers=h, proxies=proxies, timeout=10, stream=True)
        if r.status_code in [206, 200]:
            match = re.search(r'/(\d+)', r.headers.get('Content-Range', ''))
            if match:
                return int(match.group(1))
    except:
        pass

    return 0

def get_stream_urls(video_url):
    if not ZYTE_API_KEY:
        logger.error("CRITICAL: ZYTE_API_KEY is missing")
        return None, None, None

    zyte_proxy = f"http://{ZYTE_API_KEY.strip()}:@{ZYTE_HOST}:8011/"
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'nocheckcertificate': True,
        'proxy': zyte_proxy,
    }

    try:
        logger.info(f"Phase A: Fetching metadata for {video_url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            title = info.get('title', 'video')
            formats = info.get('formats', [])
            
            # Video: Max 1000p (so 720p or 480p, but NOT 1080p)
            # If you want 1080p, change 1000 to 1080.
            video_streams = [
                f for f in formats 
                if f.get('vcodec') != 'none' and f.get('acodec') == 'none' 
                and f.get('protocol', '').startswith('http')
                and f.get('height', 0) <= 1000
            ]
            video_streams.sort(key=lambda x: x.get('height', 0), reverse=True)
            
            # Audio: Best Available
            audio_streams = [
                f for f in formats 
                if f.get('acodec') != 'none' and f.get('vcodec') == 'none'
                and f.get('protocol', '').startswith('http')
            ]
            audio_streams.sort(key=lambda x: x.get('abr', 0), reverse=True)

            if video_streams and audio_streams:
                v_res = video_streams[0].get('height')
                logger.info(f"Metadata Success: {v_res}p | Title: {title[:30]}...")
                return video_streams[0]['url'], audio_streams[0]['url'], title
            
            return None, None, None

    except Exception as e:
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
    return {'User-Agent': DEFAULT_USER_AGENT}


def _should_use_single_thread(total_size: int) -> bool:
    return total_size < MIN_SIZE_FOR_PARALLEL_DOWNLOAD


def download_chunk(url: str, headers: Dict[str, str], start: int, end: int, part_filename: str, proxies: Dict[str, str], parent_filename: str) -> bool:
    headers['Range'] = f"bytes={start}-{end}"
    try:
        with requests.get(url, headers=headers, proxies=proxies, stream=True, timeout=(20, 120)) as r:
            r.raise_for_status()
            with open(part_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=PARALLEL_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        update_progress(parent_filename, bytes_added=len(chunk))
        return True
    except Exception as e:
        logger.error(f"Chunk Fail {start}-{end}: {e}")
        return False


def _download_chunks_parallel(url: str, filename: str, total_size: int, num_threads: int, proxies: Dict[str, str], headers: Dict[str, str]) -> Optional[list[str]]:
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
    with open(target_path, 'wb') as outfile:
        for part in temp_parts:
            if os.path.exists(part):
                with open(part, 'rb') as infile:
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
    logger.info(f"PARALLEL START: {filename} ({total_size / (1024*1024):.1f} MB) | {num_threads} Threads")

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
            total_size = int(r.headers.get('content-length', 0))
            update_progress(filename, total_size=total_size, status="downloading")

            with open(path, 'wb') as f:
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

def process_video_task(video_url: str):
    logger.info(f"Job Started: {video_url}")
    vid_url, aud_url, title = get_stream_urls(video_url)

    if vid_url and aud_url:
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()

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

# --- ROUTES ---

@app.get("/")
def home():
    return {"status": "Running on VPS", "endpoints": {"start": "/scrape?url=...", "progress": "/progress", "logs": "/logs", "files": "/list"}}

@app.get("/scrape")
def scrape(url: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_video_task, url)
    return {"message": "Download started", "url": url, "track": "/progress"}

@app.get("/progress")
def get_progress():
    results = {}
    with PROGRESS_LOCK:
        for filename, data in JOB_PROGRESS.items():
            pct = 0
            if data['total'] > 0:
                pct = (data['current'] / data['total']) * 100
            
            results[filename] = {
                "status": data['status'],
                "percent": round(pct, 1),
                "downloaded_mb": round(data['current'] / (1024*1024), 1),
                "total_mb": round(data['total'] / (1024*1024), 1)
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
