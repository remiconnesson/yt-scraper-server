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

from enum import Enum
from typing import Optional

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
