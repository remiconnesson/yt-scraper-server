import asyncio
import logging
import os
import sys
from fastapi import BackgroundTasks, FastAPI

from slides_extractor.downloader import DOWNLOAD_DIR, LOG_FILE, cleanup_old_downloads
from slides_extractor.job_tracker import capture_event_loop, progress_snapshot
from slides_extractor.video_jobs import process_video_task


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        ],
    )
    return logging.getLogger("scraper")


logger = configure_logging()
app = FastAPI(title="Turbo Scraper (VPS Edition)")


@app.on_event("startup")
async def on_startup() -> None:  # pragma: no cover - exercised by FastAPI runtime
    await capture_event_loop()
    await asyncio.to_thread(cleanup_old_downloads)


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
    return await progress_snapshot()


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


logger.info("FastAPI application created")
