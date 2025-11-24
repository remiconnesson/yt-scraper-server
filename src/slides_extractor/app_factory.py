import asyncio
import logging
import os
import sys
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import StreamingResponse

from slides_extractor.downloader import DOWNLOAD_DIR, cleanup_old_downloads
from slides_extractor.job_tracker import (
    has_active_progress_entries,
    progress_snapshot,
)
from slides_extractor.settings import API_PASSWORD
from slides_extractor.video_jobs import process_video_task
from slides_extractor.video_service import (
    JOBS,
    JOBS_LOCK,
    JobStatus,
    check_s3_job_exists,
    has_active_jobs,
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


def _parse_graceful_shutdown_timeout(default_seconds: int = 1800) -> int:
    raw_value = os.getenv("GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS")

    if raw_value is None:
        return default_seconds

    try:
        return int(raw_value)
    except ValueError:
        logger.exception(
            "Invalid GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS=%s; using default %s",
            raw_value,
            default_seconds,
        )
        return default_seconds


GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = _parse_graceful_shutdown_timeout()

SHUTDOWN_REQUESTED = asyncio.Event()
READINESS_EVENT = asyncio.Event()
READINESS_EVENT.set()


def reset_shutdown_state() -> None:
    """Reset shutdown markers so applications and tests start from a clean slate."""

    SHUTDOWN_REQUESTED.clear()
    READINESS_EVENT.set()


def reset_shutdown_state_for_tests() -> None:
    """Backward-compatible alias used by tests; prefer reset_shutdown_state()."""

    reset_shutdown_state()


def _start_draining(reason: str) -> None:
    if SHUTDOWN_REQUESTED.is_set():
        return

    SHUTDOWN_REQUESTED.set()
    READINESS_EVENT.clear()
    logger.info("Graceful shutdown initiated: %s", reason)


async def wait_for_active_jobs(
    timeout_seconds: int = GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS,
) -> None:
    start = time.monotonic()
    attempt = 0

    while True:
        has_jobs = await has_active_jobs()
        has_progress = has_active_progress_entries()
        if not has_jobs and not has_progress:
            logger.info("All jobs finished; proceeding with shutdown")
            return

        if time.monotonic() - start >= timeout_seconds:
            logger.warning("Shutdown timeout reached with in-flight jobs")
            return

        attempt += 1
        sleep_for = min(1.0 + 0.5 * attempt, 5.0)
        logger.info(
            "Waiting for active jobs before shutdown",
            extra={"jobs_active": has_jobs, "progress_active": has_progress},
        )
        await asyncio.sleep(sleep_for)


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
    reset_shutdown_state()
    await asyncio.to_thread(cleanup_old_downloads)
    try:
        yield
    finally:
        _start_draining("shutdown event")
        await wait_for_active_jobs(GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS)


app = FastAPI(
    title="Turbo Scraper (VPS Edition)",
    lifespan=app_lifespan,
)

AUTH_DEPENDENCIES = [Depends(require_api_password)]


@app.get("/healthz/live")
def liveness_probe():
    return {"status": "alive"}


@app.get("/healthz/ready")
def readiness_probe():
    if not READINESS_EVENT.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Draining for shutdown",
        )

    return {"status": "ready"}


@app.post(
    "/drain",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=AUTH_DEPENDENCIES,
)
async def initiate_drain():
    _start_draining("preStop hook")
    return {"status": "draining"}


@app.get("/", dependencies=AUTH_DEPENDENCIES)
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


@app.post("/process/youtube/{video_id}", dependencies=AUTH_DEPENDENCIES)
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


@app.get("/progress", dependencies=AUTH_DEPENDENCIES)
async def get_progress():
    return await progress_snapshot()


@app.get("/jobs/{video_id}", dependencies=AUTH_DEPENDENCIES)
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


@app.get("/jobs/{video_id}/stream", dependencies=AUTH_DEPENDENCIES)
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




@app.get("/logs", dependencies=AUTH_DEPENDENCIES)
def view_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            recent_logs = deque(f, maxlen=50)
            return {"recent_logs": list(reversed(recent_logs))}
    return {"error": "Log file empty or missing"}


@app.get("/list", dependencies=AUTH_DEPENDENCIES)
def list_files():
    files = os.listdir(DOWNLOAD_DIR)
    data = [{"filename": f, "url": f"/files/{f}"} for f in files]
    return {"files": data}


logger.info("FastAPI application created")
