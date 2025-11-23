"""
S3 storage structure:

s3://bucket-name/
  video/
    {video_id}/
      images/
        segment_001.webp
        segment_002.webp
      metadata.json

Where `video_id` is the YouTube video ID for YouTube sources or the first 16
characters of the SHA256 hash for S3-sourced videos.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel


class JobStatus(str, Enum):
    pending = "pending"
    downloading = "downloading"
    extracting = "extracting"
    compressing = "compressing"
    uploading = "uploading"
    completed = "completed"
    failed = "failed"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    stream_url: str


class JobResult(BaseModel):
    job_id: str
    status: JobStatus
    metadata_url: Optional[str] = None
    error: Optional[str] = None
    frame_count: Optional[int] = None
    video_id: Optional[str] = None


class ProcessRequest(BaseModel):
    s3_uri: str


JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = asyncio.Lock()


def compress_image(
    frame: np.ndarray, quality: Optional[int] = None
) -> tuple[bytes, dict[str, Any]]:
    """Compress a single video frame into WebP without resizing.

    The compression pipeline should preserve the original frame dimensions,
    lightly denoise the image using OpenCV's ``fastNlMeansDenoisingColored``
    function, and adapt quality based on whether the content is text-heavy or
    natural imagery. When ``quality`` is not provided, the implementation
    should detect text/slides versus natural content and select a WebP quality
    of 90 for text-forward frames and 85 for photographic frames. The returned
    metadata should include the selected format, chosen quality, and a boolean
    indicating whether text-like content was detected.

    Args:
        frame: Numpy array representing the frame to compress.
        quality: Optional override for WebP quality; ``None`` triggers adaptive
            selection based on content detection.

    Returns:
        Tuple of the compressed WebP bytes and a dictionary containing
        compression details such as format, quality used, and a ``has_text``
        flag.
    """
    raise NotImplementedError("Image compression pipeline is not implemented yet.")


def upload_to_s3(
    data: bytes,
    key: str,
    content_type: str = "image/webp",
    metadata: Optional[dict[str, str]] = None,
) -> str:
    """Upload compressed image bytes to S3 with metadata.

    This function should use a boto3 S3 client to upload the provided data to
    the configured bucket, applying the supplied key and content type headers.
    When metadata is provided, it should be stored as object metadata. Upload
    failures should surface clear error messages to aid debugging.

    Args:
        data: Compressed image bytes to upload.
        key: Destination object key within the bucket.
        content_type: MIME type for the uploaded object; defaults to WebP.
        metadata: Optional dictionary of metadata to persist alongside the
            object.

    Returns:
        Public URL of the uploaded object.
    """
    raise NotImplementedError("S3 upload helper is not implemented yet.")


async def update_job_status(
    job_id: str,
    status: JobStatus,
    progress: float,
    message: str,
    metadata_url: Optional[str] = None,
    error: Optional[str] = None,
    frame_count: Optional[int] = None,
) -> dict[str, Any]:
    """Update the in-memory job status in a thread-safe manner.

    Each update records the provided status, progress percentage, user-facing
    message, metadata URL (when available), error context, and frame count.
    Timestamps are stored in UTC ISO 8601 format. Access to the shared ``JOBS``
    registry is protected by an ``asyncio.Lock`` to ensure thread safety.

    Args:
        job_id: Unique identifier for the job being updated.
        status: Current job status value.
        progress: Percentage completion for the job.
        message: Descriptive status message for clients.
        metadata_url: Optional URL pointing to job metadata output.
        error: Optional error string describing failures.
        frame_count: Optional count of frames processed for the job.

    Returns:
        A copy of the updated job record stored in ``JOBS``.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    async with JOBS_LOCK:
        job_entry = JOBS.setdefault(
            job_id,
            {
                "status": status,
                "progress": progress,
                "message": message,
                "updated_at": timestamp,
                "metadata_url": metadata_url,
                "error": error,
                "frame_count": frame_count,
            },
        )

        job_entry["status"] = status
        job_entry["progress"] = progress
        job_entry["message"] = message
        job_entry["updated_at"] = timestamp
        job_entry["metadata_url"] = metadata_url
        job_entry["error"] = error
        job_entry["frame_count"] = frame_count

        return dict(job_entry)


async def stream_job_progress(job_id: str) -> AsyncGenerator[str, None]:
    """Stream Server-Sent Events for job progress updates.

    The generator polls the ``JOBS`` registry once per second and yields
    serialized JSON payloads formatted for SSE as ``data: <json>\n\n`` whenever
    a job's state changes. Streaming stops when the job reaches a completed or
    failed status. If no updates are detected for five minutes, a timeout error
    is raised. Access to job state is synchronized via ``JOBS_LOCK`` and a
    missing job results in a ``KeyError``.

    Args:
        job_id: Identifier of the job to monitor.

    Yields:
        SSE-formatted strings representing job progress updates.
    """
    last_update: Optional[str] = None
    last_activity = datetime.now(timezone.utc)

    while True:
        async with JOBS_LOCK:
            job_state = JOBS.get(job_id)

        if job_state is None:
            raise KeyError(f"Job not found: {job_id}")

        updated_at = job_state.get("updated_at")
        if updated_at != last_update:
            yield f"data: {json.dumps(job_state)}\n\n"
            last_update = updated_at
            last_activity = datetime.now(timezone.utc)

        status = job_state.get("status", "")
        if isinstance(status, JobStatus):
            status_value = status.value
        else:
            status_value = str(status)

        if status_value.lower() in {JobStatus.completed.value, JobStatus.failed.value}:
            break

        if (datetime.now(timezone.utc) - last_activity).total_seconds() >= 300:
            raise TimeoutError(
                f"No updates for job {job_id} in the last 300 seconds"
            )

        await asyncio.sleep(1)
