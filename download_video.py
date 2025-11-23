"""Utilities for downloading videos from YouTube or S3 sources."""

import logging
import os
from typing import Tuple
from urllib.parse import urlparse

import boto3
import yt_dlp
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


def _parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Return the bucket and key extracted from an S3 URI.

    Supports ``s3://`` URIs and HTTPS-style endpoints such as
    ``https://bucket.s3.amazonaws.com/key`` or ``https://s3.amazonaws.com/bucket/key``.
    Raises ``ValueError`` when the URI cannot be parsed.
    """

    parsed = urlparse(s3_uri)

    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
    elif parsed.scheme in {"http", "https"}:
        host_parts = parsed.netloc.split(".")
        path_segments = parsed.path.lstrip("/").split("/", 1)

        bucket = ""
        key = ""

        # Virtual-hosted-style: bucket.s3.amazonaws.com/key
        if len(host_parts) >= 3 and host_parts[1] == "s3":
            bucket = host_parts[0]
            key = parsed.path.lstrip("/")
        # Path-style: s3.amazonaws.com/bucket/key
        elif parsed.netloc.startswith("s3.") or parsed.netloc == "s3.amazonaws.com":
            if path_segments and path_segments[0]:
                bucket = path_segments[0]
                key = path_segments[1] if len(path_segments) > 1 else ""
        else:
            if path_segments and path_segments[0]:
                bucket = path_segments[0]
                key = path_segments[1] if len(path_segments) > 1 else ""

    else:
        raise ValueError(f"Unsupported S3 URI scheme: {parsed.scheme or 'missing'}")

    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: bucket='{bucket}', key='{key}'")

    return bucket, key


def download_video(video_url: str, output_path: str, source_type: str) -> bool:
    """Download a video from YouTube or S3.

    Args:
        video_url: YouTube URL or S3 URI/HTTPS link.
        output_path: Destination file path where the video will be saved.
        source_type: Either ``"youtube"`` or ``"s3"``.

    Returns:
        ``True`` when the download succeeds, otherwise ``False``.
    """

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if source_type == "youtube":
        ydl_opts = {
            "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "merge_output_format": "mp4",
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            logger.info("Starting YouTube download: %s -> %s", video_url, output_path)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            logger.info("YouTube download finished: %s", output_path)
            return True
        except Exception as exc:  # yt_dlp raises generic exceptions
            logger.error(
                "YouTube download failed for %s: %s", video_url, exc, exc_info=True
            )
            return False

    if source_type == "s3":
        try:
            bucket, key = _parse_s3_uri(video_url)
        except ValueError as exc:
            logger.error("Invalid S3 URI %s: %s", video_url, exc)
            return False

        try:
            logger.info("Starting S3 download: s3://%s/%s -> %s", bucket, key, output_path)
            s3_client = boto3.client("s3")
            s3_client.download_file(bucket, key, output_path)
            logger.info("S3 download finished: %s", output_path)
            return True
        except (ClientError, BotoCoreError, Exception) as exc:
            logger.error(
                "S3 download failed for s3://%s/%s: %s", bucket, key, exc, exc_info=True
            )
            return False

    logger.error("Unsupported source_type: %s", source_type)
    return False
