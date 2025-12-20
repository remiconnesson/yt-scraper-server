"""
Vercel Blob storage structure:

slides/
  {video_id}/
    1-first.webp
    1-last.webp
    ...
manifests/
  {video_id}.json

Where `video_id` is the YouTube video ID.
"""

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import cv2
import imagehash
import numpy as np
from PIL import Image
from pydantic import BaseModel
from vercel.blob import AsyncBlobClient

from slides_extractor.constants import MANIFEST_PATH_TEMPLATE
from slides_extractor.extract_slides.video_analyzer import (
    FrameStreamer,
    Segment,
    SegmentDetector,
)
from slides_extractor.settings import (
    SLIDE_IMAGE_QUALITY,
)


class JobStatus(str, Enum):
    pending = "pending"
    downloading = "downloading"
    extracting = "extracting"
    uploading = "uploading"
    completed = "completed"
    failed = "failed"


class JobResponse(BaseModel):
    video_id: str
    status: JobStatus
    stream_url: str


class JobResult(BaseModel):
    video_id: str
    status: JobStatus
    metadata_uri: Optional[str] = None
    error: Optional[str] = None
    frame_count: Optional[int] = None


JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = asyncio.Lock()

logger = logging.getLogger(__name__)


_ACTIVE_JOB_STATUSES = {
    JobStatus.pending,
    JobStatus.downloading,
    JobStatus.extracting,
    JobStatus.uploading,
}


async def has_active_jobs() -> bool:
    """Return True when any tracked job is still running."""

    async with JOBS_LOCK:
        return any(job.get("status") in _ACTIVE_JOB_STATUSES for job in JOBS.values())


async def upload_to_blob(data: bytes, path: str, content_type: str) -> str:
    """Upload data to Vercel Blob Storage asynchronously."""
    client = AsyncBlobClient()
    response = await client.put(
        path, data, content_type=content_type, add_random_suffix=False, overwrite=True
    )
    return response.url


def generate_blob_path(video_id: str, slide_index: int, frame_position: str) -> str:
    """Generate a deterministic path for a slide frame."""
    return f"slides/{video_id}/{slide_index}-{frame_position}.webp"


def _compute_frame_hash(frame: np.ndarray) -> imagehash.ImageHash:
    """Compute a perceptual hash for deduplicating visually similar frames."""

    pil_image = Image.fromarray(frame)
    return imagehash.phash(pil_image, hash_size=16)


async def update_job_status(
    video_id: str,
    status: JobStatus,
    progress: float,
    message: str,
    metadata_uri: Optional[str] = None,
    error: Optional[str] = None,
    frame_count: Optional[int] = None,
) -> dict[str, Any]:
    """Update the in-memory job status in a thread-safe manner.

    Each update records the provided status, progress percentage, user-facing
    message, metadata URI (when available), error context, and frame count.
    Timestamps are stored in UTC ISO 8601 format. Access to the shared ``JOBS``
    registry is protected by an ``asyncio.Lock`` to ensure thread safety.

    Args:
        video_id: Unique identifier for the job being updated.
        status: Current job status value.
        progress: Percentage completion for the job.
        message: Descriptive status message for clients.
        metadata_uri: Optional URI pointing to job metadata output.
        error: Optional error string describing failures.
        frame_count: Optional count of frames processed for the job.

    Returns:
        A copy of the updated job record stored in ``JOBS``.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    async with JOBS_LOCK:
        job_entry = JOBS.setdefault(
            video_id,
            {
                "status": status,
                "progress": progress,
                "message": message,
                "updated_at": timestamp,
                "metadata_uri": metadata_uri,
                "error": error,
                "frame_count": frame_count,
            },
        )

        job_entry["status"] = status
        job_entry["progress"] = progress
        job_entry["message"] = message
        job_entry["updated_at"] = timestamp
        job_entry["metadata_uri"] = metadata_uri
        job_entry["error"] = error
        job_entry["frame_count"] = frame_count

        return dict(job_entry)


async def stream_job_progress(video_id: str) -> AsyncGenerator[str, None]:
    """Stream Server-Sent Events for job progress updates.

    The generator polls the ``JOBS`` registry once per second and yields
    serialized JSON payloads formatted for SSE as ``data: <json>\n\n`` whenever
    a job's state changes. Streaming stops when the job reaches a completed or
    failed status. If no updates are detected for five minutes, a timeout error
    is raised. Access to job state is synchronized via ``JOBS_LOCK`` and a
    missing job results in a ``KeyError``.

    Args:
        video_id: Identifier of the job to monitor.

    Yields:
        SSE-formatted strings representing job progress updates.
    """
    last_update: Optional[str] = None
    last_activity = datetime.now(timezone.utc)

    async with JOBS_LOCK:
        if video_id not in JOBS:
            raise KeyError(f"Job not found: {video_id}")

    while True:
        async with JOBS_LOCK:
            job_state = JOBS.get(video_id)
            if job_state is None:
                raise KeyError(f"Job not found: {video_id}")

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
            raise TimeoutError(f"No updates for job {video_id} in the last 300 seconds")

        await asyncio.sleep(1)


async def _detect_static_segments(
    video_path: str, video_id: str
) -> tuple[list[Segment], list[Segment], Optional[int]]:
    """Detect static segments while streaming frames and updating progress."""

    streamer = FrameStreamer(video_path)
    detector = SegmentDetector()

    await update_job_status(
        video_id,
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
                video_id,
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
    local_output_dir: Optional[str] = None,
    dedup_threshold: int = 5,
) -> list[dict[str, Any]]:
    """Upload representative and trailing frames to storage in parallel."""

    hash_records: list[tuple[imagehash.ImageHash, int, str]] = []
    segment_data_list: list[dict[str, Any]] = []

    # 1. Prepare all frames and determine duplicates
    # TODO_LATER: start=1 is the source of the issue of off by one error in the frontend(keep for now, we need to backfill)
    for idx, segment in enumerate(segments, start=1):
        frames_to_process = [
            ("first", segment.representative_frame),
            ("last", segment.last_frame),
        ]

        segment_info: dict[str, Any] = {
            "idx": idx,
            "segment": segment,
            "frames": {},
        }

        for position, frame_img in frames_to_process:
            if frame_img is None:
                segment_info["frames"][position] = None
                continue

            frame_hash = _compute_frame_hash(frame_img)
            duplicate_of: tuple[int, str] | None = None
            for existing_hash, origin_idx, origin_position in hash_records:
                if frame_hash - existing_hash <= dedup_threshold:
                    duplicate_of = (origin_idx, origin_position)
                    break

            if duplicate_of is None:
                hash_records.append((frame_hash, idx, position))

            bgr_frame = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode(
                ".webp", bgr_frame, [cv2.IMWRITE_WEBP_QUALITY, SLIDE_IMAGE_QUALITY]
            )
            if not success:
                raise ValueError(f"Failed to encode frame {idx} {position} to WebP")

            segment_info["frames"][position] = {
                "data": buffer.tobytes(),
                "duplicate_of": duplicate_of,
                "path": generate_blob_path(video_id, idx, position),
            }
        segment_data_list.append(segment_info)

    # 2. Collect upload tasks for unique frames
    tasks = []
    task_keys = []  # To map results back: (idx, position)
    for seg_info in segment_data_list:
        idx = seg_info["idx"]
        for position, frame_info in seg_info["frames"].items():
            if frame_info and frame_info["duplicate_of"] is None:
                if not local_output_dir:
                    tasks.append(
                        upload_to_blob(
                            frame_info["data"], frame_info["path"], "image/webp"
                        )
                    )
                    task_keys.append((idx, position))
                else:
                    # Handle local output synchronously for simplicity or use a thread pool
                    # But the requirement is focused on Blob migration.
                    full_dir = os.path.join(local_output_dir, "slides", video_id)
                    os.makedirs(full_dir, exist_ok=True)
                    filename = f"{idx}-{position}.webp"
                    full_path = os.path.join(full_dir, filename)
                    with open(full_path, "wb") as f:
                        f.write(frame_info["data"])
                    frame_info["url"] = f"file://{os.path.abspath(full_path)}"

    # 3. Execute uploads in parallel
    results = await asyncio.gather(*tasks) if tasks else []
    url_map = {key: url for key, url in zip(task_keys, results, strict=True)}

    # 4. Finalize segment metadata
    segment_metadata: list[dict[str, Any]] = []
    total_static = len(segments) or 1

    # TODO_LATER: start=1 is the source of the issue of off by one error in the frontend(keep for now, we need to backfill)
    for idx, seg_info in enumerate(segment_data_list, start=1):
        progress = 60.0 + ((idx - 1) / total_static) * 35.0
        await update_job_status(
            video_id,
            JobStatus.uploading,
            progress=progress,
            message=f"Processing slide {idx}/{total_static}",
        )

        segment = seg_info["segment"]
        current_seg_meta: dict[str, Any] = {
            "segment_id": idx,
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "duration": segment.duration,
            "frame_count": segment.frame_count,
        }

        for position in ["first", "last"]:
            frame_info = seg_info["frames"][position]
            image_meta: dict[str, Any] = {
                "frame_id": None,
                "duplicate_of": None,
                "url": None,
            }

            if frame_info is None:
                current_seg_meta[f"{position}_frame"] = image_meta
                continue

            frame_id = f"{idx}-{position}.webp"
            image_meta["frame_id"] = frame_id

            if frame_info["duplicate_of"]:
                origin_idx, origin_pos = frame_info["duplicate_of"]
                image_meta["duplicate_of"] = {
                    "segment_id": origin_idx,
                    "frame_position": origin_pos,
                }
                # Find the URL of the original frame
                # Duplicates will always refer to a previous frame that was unique
                # So it must be in our url_map or already have a "url" set if local
                if local_output_dir:
                    # Find original in previous segment_data_list entries
                    orig_seg = segment_data_list[origin_idx - 1]
                    image_meta["url"] = orig_seg["frames"][origin_pos]["url"]
                else:
                    image_meta["url"] = url_map.get((origin_idx, origin_pos))
            else:
                if local_output_dir:
                    image_meta["url"] = frame_info["url"]
                else:
                    image_meta["url"] = url_map.get((idx, position))

            current_seg_meta[f"{position}_frame"] = image_meta

        segment_metadata.append(current_seg_meta)

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
            "duration": segment.duration,
        }

        if segment.type == "static":
            if static_index >= len(static_segment_metadata):
                raise ValueError(
                    "Static segment metadata missing for video_segments manifest"
                )

            static_meta = static_segment_metadata[static_index]
            static_index += 1

            first_frame = static_meta.get("first_frame")
            entry["first_frame"] = first_frame
            entry["last_frame"] = static_meta.get("last_frame")

            if first_frame:
                entry["url"] = first_frame.get("url")

        manifest_segments.append(entry)

    return {
        video_id: {
            "segments": manifest_segments,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    }


async def extract_and_process_frames(
    video_path: str, video_id: str, local_output_dir: Optional[str] = None
) -> list[dict[str, Any]]:
    """Orchestrate frame extraction and upload pipeline."""

    static_segments, all_segments, total_frames_seen = await _detect_static_segments(
        video_path, video_id
    )

    if not static_segments:
        await update_job_status(
            video_id,
            JobStatus.completed,
            100.0,
            "No static segments detected",
            frame_count=total_frames_seen,
        )
        return []

    segment_metadata = await _upload_segments(
        static_segments,
        video_id,
        local_output_dir=local_output_dir,
    )

    manifest = _build_segments_manifest(video_id, all_segments, segment_metadata)
    manifest_bytes = json.dumps(manifest, indent=2).encode()

    if local_output_dir:
        full_dir = os.path.join(local_output_dir, "manifests")
        os.makedirs(full_dir, exist_ok=True)
        full_path = os.path.join(full_dir, video_id)
        with open(full_path, "wb") as f:
            f.write(manifest_bytes)
        metadata_uri = f"file://{os.path.abspath(full_path)}"
    else:
        metadata_uri = await upload_to_blob(
            manifest_bytes,
            MANIFEST_PATH_TEMPLATE.format(video_id=video_id),
            content_type="application/json",
        )

    await update_job_status(
        video_id,
        JobStatus.completed,
        100.0,
        "Frame extraction completed",
        frame_count=total_frames_seen,
        metadata_uri=metadata_uri,
    )

    return segment_metadata


async def check_blob_job_exists(video_id: str) -> str | None:
    """Return Vercel Blob URL if the job output already exists."""

    from slides_extractor.settings import BLOB_READ_WRITE_TOKEN

    if not BLOB_READ_WRITE_TOKEN:
        return None

    path = MANIFEST_PATH_TEMPLATE.format(video_id=video_id)
    try:
        client = AsyncBlobClient()
        response = await client.list_objects(prefix=path)

        # Check for exact match in the listed blobs
        for blob in response.blobs:
            if blob.pathname == path:
                if blob.size > 0:
                    return blob.url
                break
        return None
    except Exception as exc:
        logger.warning(
            "Failed to check job existence in Vercel Blob for %s: %s", video_id, exc
        )
        return None
