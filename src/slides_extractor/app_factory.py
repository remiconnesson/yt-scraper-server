import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from slides_extractor.downloader import DOWNLOAD_DIR, cleanup_old_downloads
from slides_extractor.job_tracker import progress_snapshot
from slides_extractor.video_jobs import process_video_task
from slides_extractor.video_service import (
    JOBS,
    JOBS_LOCK,
    JobStatus,
    stream_job_progress,
    update_job_status,
)

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
            "job_status": "/jobs/{job_id}",
            "job_stream": "/jobs/{job_id}/stream",
        },
    }


@app.post("/process/youtube/{video_id}")
def process_youtube_video(video_id: str, background_tasks: BackgroundTasks):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    job_id = str(uuid4())
    asyncio.run(
        update_job_status(
            job_id,
            status=JobStatus.pending,
            progress=0.0,
            message="Job accepted",
        )
    )
    background_tasks.add_task(process_video_task, video_url, video_id, job_id)
    return {
        "message": "Download started",
        "video_id": video_id,
        "job_id": job_id,
        "track": f"/jobs/{job_id}",
        "stream": f"/jobs/{job_id}/stream",
    }


@app.get("/progress")
async def get_progress():
    return await progress_snapshot()


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    async with JOBS_LOCK:
        job = dict(JOBS.get(job_id, {}))

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return job


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str) -> StreamingResponse:
    async with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    async def _event_stream() -> AsyncIterator[str]:
        try:
            async for event in stream_job_progress(job_id):
                yield event
        except TimeoutError as exc:
            yield f"event: error\ndata: {exc}\n\n"
        except KeyError as exc:
            yield f"event: error\ndata: {exc}\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


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
