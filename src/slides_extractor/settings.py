import logging
import os
from dotenv import load_dotenv


def _parse_retention_hours(raw_value: str | None, default: int = 24) -> int:
    """Parse the retention env var defensively, returning a fallback on error."""

    logger = logging.getLogger("scraper")

    if raw_value is None:
        return default

    normalized = raw_value.strip()
    if normalized == "":
        logger.warning(
            "Invalid DOWNLOAD_RETENTION_HOURS value '%s'; using default %s (empty string)",
            raw_value,
            default,
        )
        return default

    try:
        return int(normalized)
    except ValueError as exc:
        logger.warning(
            "Invalid DOWNLOAD_RETENTION_HOURS value '%s'; using default %s (%s)",
            raw_value,
            default,
            exc,
        )
        return default


# Load environment variables
load_dotenv()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
LOG_FILE = os.path.join(BASE_DIR, "app.log")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


ZYTE_API_KEY = os.getenv("ZYTE_API_KEY")
ZYTE_HOST = os.getenv("ZYTE_HOST", "api.zyte.com")
DATACENTER_PROXY = os.getenv("DATACENTER_PROXY")

try:
    SLIDE_IMAGE_QUALITY = int(os.getenv("SLIDE_IMAGE_QUALITY", "80"))
except ValueError:
    SLIDE_IMAGE_QUALITY = 80


MIN_SIZE_FOR_PARALLEL_DOWNLOAD = 1 * 1024 * 1024  # 1MB
VIDEO_DOWNLOAD_THREADS = 32
AUDIO_DOWNLOAD_THREADS = 8
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
PARALLEL_CHUNK_SIZE = 1024 * 1024
SINGLE_THREAD_CHUNK_SIZE = 32 * 1024
DEFAULT_RETENTION_HOURS = 24
DOWNLOAD_RETENTION_HOURS_RAW = os.getenv("DOWNLOAD_RETENTION_HOURS")
DOWNLOAD_RETENTION_HOURS = _parse_retention_hours(
    DOWNLOAD_RETENTION_HOURS_RAW, DEFAULT_RETENTION_HOURS
)

API_PASSWORD = os.getenv("API_PASSWORD")
BLOB_READ_WRITE_TOKEN = os.getenv("BLOB_READ_WRITE_TOKEN")
