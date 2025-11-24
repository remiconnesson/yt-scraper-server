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
        "s3_key": "video/video_id/static_frames/static_frame_000001.png",
        "s3_bucket": "slides-extractor",
        "s3_uri": "s3://slides-extractor/video/video_id/static_frames/static_frame_000001.png",
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
import logging
import os
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from pydantic import BaseModel
import cv2
import requests

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
_S3_CLIENT = None
_NORMALIZED_S3_ENDPOINT: str | None = None

logger = logging.getLogger(__name__)


def upload_to_s3(
    data: bytes,
    key: str,
    content_type: str = "image/png",
    metadata: Optional[dict[str, str]] = None,
) -> str:
    """Upload image bytes to S3 with metadata.

    The upload uses a presigned URL and standard HTTP PUT via requests to ensure
    compatibility with S3 providers that mishandle chunked transfer encoding.
    Objects are uploaded as private, accessible only with valid credentials.

    Args:
        data: Image bytes to upload.
        key: Destination object key within the bucket.
        content_type: MIME type for the uploaded object; defaults to PNG.
        metadata: Optional dictionary of metadata to persist alongside the
            object.

    Returns:
        The S3 URL of the uploaded object (requires auth to access).
    """
    s3 = _get_s3_client()

    # Generate presigned URL for PUT
    params: dict[str, Any] = {
        "Bucket": S3_BUCKET_NAME,
        "Key": key,
        "ContentType": content_type,
    }
    if metadata:
        params["Metadata"] = metadata

    try:
        presigned_url = s3.generate_presigned_url(
            "put_object",
            Params=params,
            ExpiresIn=300,  # 5 minutes
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to generate presigned URL: {exc}") from exc

    # Prepare headers for the request
    headers = {"Content-Type": content_type}
    if metadata:
        for k, v in metadata.items():
            headers[f"x-amz-meta-{k}"] = v

    # Upload via requests (avoids boto3 chunked encoding issues with this S3 provider)
    response = requests.put(
        presigned_url,
        data=data,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    # Construct public URL
    endpoint = _get_s3_endpoint()
    return f"{endpoint}/{S3_BUCKET_NAME}/{key}"


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
) -> tuple[list[Segment], list[Segment], Optional[int]]:
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

    return static_segments, detector.segments, total_frames_seen


async def _upload_segments(
    segments: list[Segment],
    video_id: str,
    job_id: str,
    local_output_dir: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Upload representative frames to S3 or save locally and report progress."""

    total_static = len(segments) or 1
    segment_metadata: list[dict[str, Any]] = []

    for idx, segment in enumerate(segments, start=1):
        if segment.representative_frame is None:
            continue

        bgr_frame = cv2.cvtColor(segment.representative_frame, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".png", bgr_frame)
        if not success:
            raise ValueError("Failed to encode frame to PNG")

        frame_id = f"static_frame_{idx:06d}.png"
        s3_key = f"video/{video_id}/static_frames/{frame_id}"

        if local_output_dir:
            full_dir = os.path.join(
                local_output_dir, "video", video_id, "static_frames"
            )
            os.makedirs(full_dir, exist_ok=True)
            full_path = os.path.join(full_dir, frame_id)
            with open(full_path, "wb") as f:
                f.write(buffer.tobytes())
            image_url = f"file://{os.path.abspath(full_path)}"
            bucket_name = "local"
            uri = image_url
        else:
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
            bucket_name = S3_BUCKET_NAME
            uri = f"s3://{S3_BUCKET_NAME}/{s3_key}"

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
                "frame_id": frame_id,
                "s3_key": s3_key,
                "s3_bucket": bucket_name,
                "s3_uri": uri,
            }
        )

    return segment_metadata


def _build_segments_manifest(
    video_id: str,
    segments: list[Segment],
    static_segment_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a structured manifest for all segments using uploaded frame data."""

    static_index = 0
    manifest_segments: list[dict[str, Any]] = []

    for segment in segments:
        entry: dict[str, Any] = {
            "kind": segment.type,
            "start_time": segment.start_time,
            "end_time": segment.end_time,
        }

        if segment.type == "static":
            if static_index >= len(static_segment_metadata):
                raise ValueError(
                    "Static segment metadata missing for video_segments manifest"
                )

            static_meta = static_segment_metadata[static_index]
            static_index += 1

            entry["frame_id"] = static_meta.get("frame_id")
            entry["url"] = static_meta.get("image_url")
            entry["s3_key"] = static_meta.get("s3_key")
            entry["s3_bucket"] = static_meta.get("s3_bucket")
            entry["s3_uri"] = static_meta.get("s3_uri")

        manifest_segments.append(entry)

    return {video_id: {"segments": manifest_segments}}


async def extract_and_process_frames(
    video_path: str, video_id: str, job_id: str, local_output_dir: Optional[str] = None
) -> list[dict[str, Any]]:
    """Orchestrate frame extraction and upload pipeline."""

    static_segments, all_segments, total_frames_seen = await _detect_static_segments(
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

    segment_metadata = await _upload_segments(
        static_segments, video_id, job_id, local_output_dir=local_output_dir
    )

    manifest = _build_segments_manifest(video_id, all_segments, segment_metadata)
    manifest_bytes = json.dumps(manifest, indent=2).encode()

    if local_output_dir:
        full_dir = os.path.join(local_output_dir, "video", video_id)
        os.makedirs(full_dir, exist_ok=True)
        full_path = os.path.join(full_dir, "video_segments.json")
        with open(full_path, "wb") as f:
            f.write(manifest_bytes)
        metadata_url = f"file://{os.path.abspath(full_path)}"
    else:
        metadata_url = upload_to_s3(
            manifest_bytes,
            f"video/{video_id}/video_segments.json",
            content_type="application/json",
            metadata={"video_id": video_id},
        )

    await update_job_status(
        job_id,
        JobStatus.completed,
        100.0,
        "Frame extraction completed",
        frame_count=total_frames_seen,
        metadata_url=metadata_url,
    )

    return segment_metadata


def _get_s3_endpoint() -> str:
    if not S3_ENDPOINT:
        raise RuntimeError("S3 configuration missing; set S3_ENDPOINT")

    global _NORMALIZED_S3_ENDPOINT
    if _NORMALIZED_S3_ENDPOINT is None:
        if S3_ENDPOINT.startswith(("http://", "https://")):
            endpoint = S3_ENDPOINT
        else:
            endpoint = f"https://{S3_ENDPOINT}"
        _NORMALIZED_S3_ENDPOINT = endpoint.rstrip("/")

    return _NORMALIZED_S3_ENDPOINT


def _get_s3_client():
    if not S3_ENDPOINT or not S3_ACCESS_KEY:
        raise RuntimeError(
            "S3 configuration missing; set S3_ENDPOINT and S3_ACCESS_KEY"
        )

    global _S3_CLIENT
    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client(
            "s3",
            endpoint_url=_get_s3_endpoint(),
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
        )

    return _S3_CLIENT


def check_s3_job_exists(video_id: str) -> str | None:
    """Return public S3 URL if the job output already exists."""

    if not S3_ENDPOINT or not S3_ACCESS_KEY:
        return None

    key = f"video/{video_id}/video_segments.json"
    try:
        s3 = _get_s3_client()
        s3.head_object(Bucket=S3_BUCKET_NAME, Key=key)
        endpoint = _get_s3_endpoint()
        return f"{endpoint}/{S3_BUCKET_NAME}/{key}"
    except ClientError as exc:
        if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
            return None
        logger.warning("Failed to check job existence in S3 for %s: %s", video_id, exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Unexpected error checking S3 for %s: %s", video_id, exc)
        return None
