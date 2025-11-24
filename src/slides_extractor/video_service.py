"""
S3 storage structure:

video/
  {video_id}/
    static_frames/
      static_frame_000001.png
      static_frame_000002.png
      ...
    video_segments.json

Where `video_id` is the YouTube video ID for YouTube sources and `frame_000001.png` is the first static frame of the video.

video_segments.json (example):
```
{
    "video_id": {
        "segments": [
      {
            "kind": "moving",
        "start_time": 0.0,
        "end_time": 1.0,
      },
      {
        "kind": "static",
        "frame_id": "static_frame_000001.png",
        "start_time": 0.0,
        "end_time": 1.0,
        "url": f"{s3_endpoint}/video/video_id/static_frames/static_frame_000001.png",
      },
      {
        "kind": "static",
        "frame_id": "static_frame_000002.png",
        "start_time": 1.0,
        "end_time": 2.0,
        "url": f"{s3_endpoint}/video/video_id/static_frames/static_frame_000002.png",
      },
      ...
    ],
  },
}
```
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import boto3
from botocore.client import Config
from pydantic import BaseModel
import cv2

from slides_extractor.extract_slides.video_analyzer import (
    FrameStreamer,
    Segment,
    SegmentDetector,
)
from slides_extractor.settings import S3_ACCESS_KEY, S3_BUCKET_NAME, S3_ENDPOINT


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


def upload_to_s3(
    data: bytes,
    key: str,
    content_type: str = "image/png",
    metadata: Optional[dict[str, str]] = None,
) -> str:
    """Upload image bytes to S3 with metadata.

    The upload uses the S3 API via boto3. Objects are uploaded with public access
    enabled (public-read) so that returned URLs are directly accessible.

    Args:
        data: Image bytes to upload.
        key: Destination object key within the bucket.
        content_type: MIME type for the uploaded object; defaults to PNG.
        metadata: Optional dictionary of metadata to persist alongside the
            object.

    Returns:
        Public URL of the uploaded object.
    """
    if not S3_ENDPOINT or not S3_ACCESS_KEY:
        raise RuntimeError(
            "S3 configuration missing; set S3_ENDPOINT and S3_ACCESS_KEY"
        )

    # Configure S3 client
    # Note: S3_ACCESS_KEY is used for both ID and Secret as per environment spec
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )

    extra_args: dict[str, Any] = {
        "ContentType": content_type,
        "ACL": "public-read",
    }
    if metadata:
        extra_args["Metadata"] = metadata

    s3.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=key,
        Body=data,
        **extra_args,
    )

    # Construct public URL
    endpoint = S3_ENDPOINT.rstrip("/")
    # Assuming path-style access or that endpoint is the public root
    return f"{endpoint}/{key}"


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
    """Upload representative frames to S3 and report progress."""

    total_static = len(segments) or 1
    segment_metadata: list[dict[str, Any]] = []

    for idx, segment in enumerate(segments, start=1):
        if segment.representative_frame is None:
            continue

        success, buffer = cv2.imencode(".png", segment.representative_frame)
        if not success:
            raise ValueError("Failed to encode frame to PNG")

        s3_key = f"video/{video_id}/images/segment_{idx:03d}.png"
        image_url = upload_to_s3(
            buffer.tobytes(),
            s3_key,
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
