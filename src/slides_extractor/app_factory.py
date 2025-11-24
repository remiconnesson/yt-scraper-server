import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import StreamingResponse

from slides_extractor.downloader import DOWNLOAD_DIR, cleanup_old_downloads
from slides_extractor.job_tracker import progress_snapshot
from slides_extractor.settings import API_PASSWORD
from slides_extractor.video_jobs import process_video_task
from slides_extractor.video_service import (
    JOBS,
    JOBS_LOCK,
    JobStatus,
    check_s3_job_exists,
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


bearer_scheme = HTTPBearer(auto_error=False)


def require_api_password(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    configured_password = os.getenv("API_PASSWORD") or API_PASSWORD

    if configured_password is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API password is not configured",
        )

    provided_password = credentials.credentials if credentials else None
    if provided_password != configured_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API password",
        )


@asynccontextmanager
async def app_lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Kick off cleanup before serving requests.
    await asyncio.to_thread(cleanup_old_downloads)
    yield


app = FastAPI(
    title="Turbo Scraper (VPS Edition)",
    lifespan=app_lifespan,
    dependencies=[Depends(require_api_password)],
)


@app.get("/")
def home():
    return {
        "status": "Running on VPS",
        "endpoints": {
            "process": "/process/youtube/{video_id}",
            "progress": "/progress",
            "job_status": "/jobs/{video_id}",
            "job_stream": "/jobs/{video_id}/stream",
        },
    }


@app.post("/process/youtube/{video_id}")
async def process_youtube_video(video_id: str, background_tasks: BackgroundTasks):
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    async with JOBS_LOCK:
        existing_job = dict(JOBS.get(video_id, {}))
        existing_status = existing_job.get("status")
        existing_status_value = (
            existing_status.value
            if isinstance(existing_status, JobStatus)
            else str(existing_status).lower()
            if existing_status is not None
            else None
        )
        if existing_job and existing_status_value != JobStatus.failed.value:
            message = (
                "Job already completed"
                if existing_status_value == JobStatus.completed.value
                else "Job already in progress"
            )
            return {
                "message": message,
                "video_id": video_id,
                "track": f"/jobs/{video_id}",
                "stream": f"/jobs/{video_id}/stream",
                "job": existing_job,
            }

        JOBS[video_id] = {
            "status": JobStatus.pending,
            "progress": 0.0,
            "message": "Job initialized",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    metadata_uri = check_s3_job_exists(video_id)
    if metadata_uri:
        job_state = await update_job_status(
            video_id,
            status=JobStatus.completed,
            progress=100.0,
            message="Job already completed",
            metadata_uri=metadata_uri,
        )
        return {
            "message": "Job already completed",
            "video_id": video_id,
            "track": f"/jobs/{video_id}",
            "stream": f"/jobs/{video_id}/stream",
            "job": job_state,
        }

    await update_job_status(
        video_id,
        status=JobStatus.pending,
        progress=0.0,
        message="Job accepted",
    )
    background_tasks.add_task(process_video_task, video_url, video_id)
    return {
        "message": "Download started",
        "video_id": video_id,
        "track": f"/jobs/{video_id}",
        "stream": f"/jobs/{video_id}/stream",
    }


@app.get("/progress")
async def get_progress():
    return await progress_snapshot()


@app.get("/jobs/{video_id}")
async def get_job(video_id: str) -> dict[str, Any]:
    async with JOBS_LOCK:
        job = dict(JOBS.get(video_id, {}))

    if not job:
        metadata_uri = check_s3_job_exists(video_id)
        if metadata_uri:
            job = await update_job_status(
                video_id,
                status=JobStatus.completed,
                progress=100.0,
                message="Job already completed",
                metadata_uri=metadata_uri,
            )
        else:
            raise HTTPException(status_code=404, detail=f"Job not found: {video_id}")

    return job


@app.get("/jobs/{video_id}/stream")
async def stream_job(video_id: str) -> StreamingResponse:
    async with JOBS_LOCK:
        if video_id not in JOBS:
            raise HTTPException(status_code=404, detail=f"Job not found: {video_id}")

    async def _event_stream() -> AsyncIterator[str]:
        try:
            async for event in stream_job_progress(video_id):
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
