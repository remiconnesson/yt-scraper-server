"""
Blob storage structure:

video/
  {video_id}/
    images/
      segment_001.png
      segment_002.png
    metadata.json

Where `video_id` is the YouTube video ID for YouTube sources.
"""

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from slides_extractor.extract_slides.video_analyzer import (
    FrameStreamer,
    Segment,
    SegmentDetector,
)

import cv2
import requests
from pydantic import BaseModel


class JobStatus(str, Enum):
    pending = "pending"
    downloading = "downloading"
    extracting = "extracting"
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


def upload_to_vercel_blob(
    data: bytes,
    key: str,
    content_type: str = "image/png",
    metadata: Optional[dict[str, str]] = None,
) -> str:
    """Upload image bytes to Vercel Blob with metadata.

    The upload uses the Vercel Blob REST API with a read/write or write-only
    token provided via ``VERCEL_BLOB_READ_WRITE_TOKEN`` or
    ``VERCEL_BLOB_WRITE_ONLY_TOKEN``. Objects are uploaded with public access
    enabled so that returned URLs are directly accessible. Metadata, when
    supplied, is serialized to JSON and attached to the blob for traceability.

    Args:
        data: Image bytes to upload.
        key: Destination object key within the blob store.
        content_type: MIME type for the uploaded object; defaults to PNG.
        metadata: Optional dictionary of metadata to persist alongside the
            object.

    Returns:
        Public URL of the uploaded object.
    """
    token = os.getenv("VERCEL_BLOB_READ_WRITE_TOKEN") or os.getenv(
        "VERCEL_BLOB_WRITE_ONLY_TOKEN"
    )

    if not token:
        raise RuntimeError(
            "Vercel Blob token missing; set VERCEL_BLOB_READ_WRITE_TOKEN or "
            "VERCEL_BLOB_WRITE_ONLY_TOKEN"
        )

    form_fields: dict[str, str] = {
        "access": "public",
        "slug": key,
        "contentType": content_type,
    }
    if metadata:
        form_fields["metadata"] = json.dumps(metadata)

    response = requests.post(
        "https://api.vercel.com/v2/blobs",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (key, data, content_type)},
        data=form_fields,
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    url = payload.get("url")
    if not url:
        raise ValueError("Vercel Blob response did not include a URL")

    return url


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

    async with JOBS_LOCK:
        if job_id not in JOBS:
            raise KeyError(f"Job not found: {job_id}")

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
            raise TimeoutError(f"No updates for job {job_id} in the last 300 seconds")

        await asyncio.sleep(1)


async def _detect_static_segments(
    video_path: str, job_id: str
) -> tuple[list[Segment], Optional[int]]:
    """Detect static segments while streaming frames and updating progress."""

    streamer = FrameStreamer(video_path)
    detector = SegmentDetector()

    await update_job_status(
        job_id,
        JobStatus.extracting,
        0.0,
        "Starting frame analysis",
    )

    last_progress = -1.0
    total_frames_seen: Optional[int] = None
    for segment_count, frame_idx, total_frames in detector.analyze(streamer.stream()):
        if total_frames <= 0:
            continue

        total_frames_seen = total_frames
        progress = min((frame_idx / total_frames) * 60.0, 60.0)
        if progress - last_progress >= 1:
            await update_job_status(
                job_id,
                JobStatus.extracting,
                progress,
                f"Analyzing frames: {segment_count} segments, frame {frame_idx}/{total_frames}",
                frame_count=total_frames,
            )
            last_progress = progress

    static_segments = [
        segment
        for segment in detector.segments
        if segment.type == "static" and segment.representative_frame is not None
    ]

    return static_segments, total_frames_seen


async def _upload_segments(
    segments: list[Segment], video_id: str, job_id: str
) -> list[dict[str, Any]]:
    """Upload representative frames to Vercel Blob and report progress."""

    total_static = len(segments) or 1
    segment_metadata: list[dict[str, Any]] = []

    for idx, segment in enumerate(segments, start=1):
        if segment.representative_frame is None:
            continue

        success, buffer = cv2.imencode(".png", segment.representative_frame)
        if not success:
            raise ValueError("Failed to encode frame to PNG")

        blob_key = f"video/{video_id}/images/segment_{idx:03d}.png"
        image_url = upload_to_vercel_blob(
            buffer.tobytes(),
            blob_key,
            content_type="image/png",
            metadata={
                "video_id": video_id,
                "segment_id": str(idx),
                "start_time": str(segment.start_time),
                "end_time": str(segment.end_time),
            },
        )

        await update_job_status(
            job_id,
            JobStatus.uploading,
            60.0 + (idx / total_static) * 40.0,
            f"Uploaded segment {idx}/{total_static}",
            frame_count=segment.frame_count,
        )

        segment_metadata.append(
            {
                "segment_id": idx,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.duration,
                "frame_count": segment.frame_count,
                "image_url": image_url,
            }
        )

    return segment_metadata


async def extract_and_process_frames(
    video_path: str, video_id: str, job_id: str
) -> list[dict[str, Any]]:
    """Orchestrate frame extraction and upload pipeline."""

    static_segments, total_frames_seen = await _detect_static_segments(
        video_path, job_id
    )

    if not static_segments:
        await update_job_status(
            job_id,
            JobStatus.completed,
            100.0,
            "No static segments detected",
            frame_count=total_frames_seen,
        )
        return []

    segment_metadata = await _upload_segments(static_segments, video_id, job_id)

    await update_job_status(
        job_id,
        JobStatus.completed,
        100.0,
        "Frame extraction completed",
        frame_count=total_frames_seen,
    )

    return segment_metadata
