import os
import logging
import sys
import shutil
import threading
import time
import re
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
import yt_dlp
import requests
from dotenv import load_dotenv

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

# --- HELPERS ---

def update_progress(filename, bytes_added=0, total_size=None, status=None):
    with PROGRESS_LOCK:
        if filename not in JOB_PROGRESS:
            JOB_PROGRESS[filename] = {"total": 0, "current": 0, "status": "init", "start_time": time.time()}
        
        if total_size:
            JOB_PROGRESS[filename]["total"] = total_size
        
        if bytes_added:
            JOB_PROGRESS[filename]["current"] += bytes_added
            
        if status:
            JOB_PROGRESS[filename]["status"] = status

def get_file_size(url, headers, proxies):
    """
    Robust size detection. Tries 3 methods to unlock Parallel Downloading.
    """
    # Method 1: Parse 'clen' from URL (Fastest, specific to GoogleVideo)
    if "clen=" in url:
        try:
            query = parse_qs(urlparse(url).query)
            if 'clen' in query:
                size = int(query['clen'][0])
                logger.info(f"Size detected via URL clen: {size}")
                return size
        except:
            pass

    # Method 2: HEAD Request (Standard)
    try:
        head_resp = requests.head(url, headers=headers, proxies=proxies, timeout=10)
        if head_resp.status_code == 200:
            size = int(head_resp.headers.get('content-length', 0))
            if size > 0:
                return size
    except:
        pass

    # Method 3: Range Probe (The "Hammer")
    # Request just the first byte. Server replies with 'Content-Range: bytes 0-0/123456'
    try:
        h = headers.copy()
        h['Range'] = 'bytes=0-0'
        r = requests.get(url, headers=h, proxies=proxies, timeout=10, stream=True)
        if r.status_code in [206, 200]:
            cr = r.headers.get('Content-Range', '')
            # Parse "bytes 0-0/123456"
            match = re.search(r'/(\d+)', cr)
            if match:
                size = int(match.group(1))
                logger.info(f"Size detected via Range Probe: {size}")
                return size
    except Exception as e:
        logger.warning(f"Range probe failed: {e}")

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
            
            # Video: Max 1080p
            video_streams = [
                f for f in formats 
                if f.get('vcodec') != 'none' and f.get('acodec') == 'none' 
                and f.get('protocol', '').startswith('http')
                and f.get('height', 0) <= 1080
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
            
            logger.error("No compatible streams found.")
            return None, None, None

    except Exception as e:
        logger.error(f"Phase A Failed: {e}")
        return None, None, None

def download_chunk(url, headers, start, end, part_filename, proxies, parent_filename):
    headers['Range'] = f"bytes={start}-{end}"
    try:
        with requests.get(url, headers=headers, proxies=proxies, stream=True, timeout=(20, 120)) as r:
            r.raise_for_status()
            with open(part_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=4*1024*1024): # 4MB Buffer
                    if chunk:
                        f.write(chunk)
                        update_progress(parent_filename, bytes_added=len(chunk))
        return True
    except Exception as e:
        logger.error(f"Chunk Fail {start}-{end}: {e}")
        return False

def download_file_parallel(url, filename, num_threads=8):
    path = os.path.join(DOWNLOAD_DIR, filename)
    
    proxies = {}
    if DATACENTER_PROXY and len(DATACENTER_PROXY) > 5:
        clean_proxy = DATACENTER_PROXY.strip()
        if not clean_proxy.startswith("http"):
            clean_proxy = f"http://{clean_proxy}"
        proxies = {"http": clean_proxy, "https": clean_proxy}
        logger.info(f"Using Proxy for {filename}")
    else:
        logger.info(f"Using DIRECT connection for {filename}")

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    # 1. Robust Size Detection
    total_size = get_file_size(url, headers, proxies)

    # 2. If size unknown or small, use Single Thread (Safe Mode)
    if total_size < 5 * 1024 * 1024:
        logger.warning(f"Size {total_size} too small/unknown for parallel. Using Single Thread.")
        return download_file_single(url, filename, proxies)

    # 3. Parallel Mode
    update_progress(filename, total_size=total_size, status="downloading")
    logger.info(f"PARALLEL DOWNLOAD START: {filename} ({total_size / (1024*1024):.1f} MB) | {num_threads} Threads")

    chunk_size = total_size // num_threads
    futures = []
    temp_parts = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start = i * chunk_size
            end = start + chunk_size - 1 if i < num_threads - 1 else total_size - 1
            part_name = f"{path}.part{i}"
            temp_parts.append(part_name)
            futures.append(executor.submit(download_chunk, url, headers.copy(), start, end, part_name, proxies, filename))

        for future in as_completed(futures):
            if not future.result():
                update_progress(filename, status="failed")
                return False

    # 4. Merge
    update_progress(filename, status="merging")
    logger.info(f"Merging chunks for {filename}...")
    with open(path, 'wb') as outfile:
        for part in temp_parts:
            if os.path.exists(part):
                with open(part, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile)
                os.remove(part)

    update_progress(filename, status="complete")
    logger.info(f"SAVED: {filename}")
    return True

def download_file_single(url, filename, proxies):
    path = os.path.join(DOWNLOAD_DIR, filename)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        with requests.get(url, headers=headers, proxies=proxies, stream=True, timeout=(20, 120)) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            update_progress(filename, total_size=total_size, status="downloading")
            
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=4*1024*1024):
                    f.write(chunk)
                    update_progress(filename, bytes_added=len(chunk))
        
        update_progress(filename, status="complete")
        logger.info(f"SAVED Single: {filename}")
        return True
    except Exception as e:
        logger.error(f"Single Download Failed: {e}")
        update_progress(filename, status="failed")
        return False

def process_video_task(video_url: str):
    logger.info(f"Job Started: {video_url}")
    vid_url, aud_url, title = get_stream_urls(video_url)
    
    if vid_url and aud_url:
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        
        # Max Speed: 8 Threads for Video, 4 for Audio
        download_file_parallel(vid_url, f"{safe_title}_video.mp4", num_threads=8)
        download_file_parallel(aud_url, f"{safe_title}_audio.m4a", num_threads=4)
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
