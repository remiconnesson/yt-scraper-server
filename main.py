import os
import logging
import sys
import shutil
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
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=DOWNLOAD_DIR), name="files")

# Standard Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("scraper")

# --- CONFIGURATION ---
ZYTE_API_KEY = os.getenv("ZYTE_API_KEY")
ZYTE_HOST = os.getenv("ZYTE_HOST", "api.zyte.com")
DATACENTER_PROXY = os.getenv("DATACENTER_PROXY")

# --- CORE LOGIC ---

def get_stream_urls(video_url):
    """Phase A: Get Metadata via Zyte"""
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
            
            # Video: Filter HTTP streams <= 1000p (Excludes 1080p, targets 720p)
            video_streams = [
                f for f in formats 
                if f.get('vcodec') != 'none' and f.get('acodec') == 'none' 
                and f.get('protocol', '').startswith('http')
                and f.get('height', 0) <= 1000
            ]
            video_streams.sort(key=lambda x: x.get('height', 0), reverse=True)
            
            # Audio: Filter HTTP streams
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

def download_chunk(url, headers, start, end, part_filename, proxies):
    """Downloads a specific byte range of the file"""
    headers['Range'] = f"bytes={start}-{end}"
    try:
        with requests.get(url, headers=headers, proxies=proxies, stream=True, timeout=(15, 60)) as r:
            r.raise_for_status()
            with open(part_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Chunk Fail {start}-{end}: {e}")
        return False

def download_file_parallel(url, filename, num_threads=4):
    """Phase B: Content Ingestion (The Muscle - Multi-Threaded)"""
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

    # 1. Get File Size
    try:
        head_resp = requests.head(url, headers=headers, proxies=proxies, timeout=15)
        total_size = int(head_resp.headers.get('content-length', 0))
    except:
        logger.warning("Could not get size, falling back to single-thread")
        return download_file_single(url, filename, proxies)

    if total_size < 5 * 1024 * 1024: # If smaller than 5MB, just single thread it
        return download_file_single(url, filename, proxies)

    logger.info(f"Parallel Download: {filename} ({total_size / (1024*1024):.1f} MB) | Threads: {num_threads}")

    # 2. Calculate Chunks
    chunk_size = total_size // num_threads
    futures = []
    temp_parts = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start = i * chunk_size
            end = start + chunk_size - 1 if i < num_threads - 1 else total_size - 1
            part_name = f"{path}.part{i}"
            temp_parts.append(part_name)
            futures.append(executor.submit(download_chunk, url, headers.copy(), start, end, part_name, proxies))

        # Wait for all chunks
        for future in as_completed(futures):
            if not future.result():
                logger.error("Parallel download failed part. Aborting.")
                return False

    # 3. Merge Chunks
    logger.info(f"Merging chunks for {filename}...")
    with open(path, 'wb') as outfile:
        for part in temp_parts:
            with open(part, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)
            os.remove(part) # Cleanup

    logger.info(f"Saved: {filename}")
    return True

def download_file_single(url, filename, proxies):
    """Fallback for small files or errors"""
    path = os.path.join(DOWNLOAD_DIR, filename)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        with requests.get(url, headers=headers, proxies=proxies, stream=True, timeout=(15, 60)) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Single Download Failed: {e}")
        return False

def process_video_task(video_url: str):
    """Background Worker"""
    logger.info(f"Job Started: {video_url}")
    vid_url, aud_url, title = get_stream_urls(video_url)
    
    if vid_url and aud_url:
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        
        # Parallel Download
        download_file_parallel(vid_url, f"{safe_title}_video.mp4", num_threads=8)
        download_file_parallel(aud_url, f"{safe_title}_audio.m4a", num_threads=4)
        logger.info(f"Job Finished: {safe_title}")
    else:
        logger.error("Job Failed during Phase A")

# --- ROUTES ---

@app.get("/")
def home():
    return {"status": "Running on VPS", "usage": "/scrape?url=..."}

@app.get("/scrape")
def scrape(url: str, background_tasks: BackgroundTasks):
    """Triggers the download in the background"""
    background_tasks.add_task(process_video_task, url)
    return {"message": "Download started", "url": url}

@app.get("/list")
def list_files():
    files = os.listdir(DOWNLOAD_DIR)
    data = [{"filename": f, "url": f"/files/{f}"} for f in files]
    return {"files": data}
