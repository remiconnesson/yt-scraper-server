import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI

from slides_extractor.downloader import DOWNLOAD_DIR, cleanup_old_downloads
from slides_extractor.job_tracker import progress_snapshot
from slides_extractor.video_jobs import process_video_task

LOG_FILE = "app.log"


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


@asynccontextmanager
async def app_lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Kick off cleanup before serving requests.
    await asyncio.to_thread(cleanup_old_downloads)
    yield


app = FastAPI(title="Turbo Scraper (VPS Edition)", lifespan=app_lifespan)


@app.get("/")
def home():
    return {
        "status": "Running on VPS",
        "endpoints": {
            "process": "/process/youtube/{video_id}",
            "progress": "/progress",
        },
    }


@app.post("/process/youtube/{video_id}")
def process_youtube_video(video_id: str, background_tasks: BackgroundTasks):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    job_id = str(uuid4())
    background_tasks.add_task(process_video_task, video_url, video_id, job_id)
    return {
        "message": "Download started",
        "video_id": video_id,
        "job_id": job_id,
        "track": "/progress",
    }


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
