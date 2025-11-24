import threading
import time
from typing import Dict, Optional, TypedDict, cast


class ProgressState(TypedDict):
    total: float
    current: float
    status: str
    start_time: float


class ProgressSnapshot(TypedDict):
    status: str
    percent: float
    downloaded_mb: float
    total_mb: float


JOB_PROGRESS: Dict[str, ProgressState] = {}
PROGRESS_LOCK = threading.Lock()


def update_progress(
    filename: str,
    bytes_added: int = 0,
    total_size: Optional[int] = None,
    status: Optional[str] = None,
) -> None:
    """Thread-safe progress updates using a standard lock."""

    with PROGRESS_LOCK:
        if filename not in JOB_PROGRESS:
            JOB_PROGRESS[filename] = ProgressState(
                total=0.0,
                current=0.0,
                status="init",
                start_time=time.time(),
            )

        if total_size is not None:
            JOB_PROGRESS[filename]["total"] = float(total_size)

        if bytes_added:
            JOB_PROGRESS[filename]["current"] += float(bytes_added)

        if status:
            JOB_PROGRESS[filename]["status"] = status


def remove_progress_entry(filename: str) -> None:
    """Remove a progress record when a download completes or is cleaned up."""

    with PROGRESS_LOCK:
        JOB_PROGRESS.pop(filename, None)


async def progress_snapshot() -> dict[str, ProgressSnapshot]:
    """Return a copy of the progress table with percentage calculations."""

    results: dict[str, ProgressSnapshot] = {}
    with PROGRESS_LOCK:
        for filename, data in JOB_PROGRESS.items():
            pct = 0.0
            total = data["total"]
            current = data["current"]
            if total > 0:
                pct = (current / total) * 100

            results[filename] = cast(
                ProgressSnapshot,
                {
                    "status": data["status"],
                    "percent": round(pct, 1),
                    "downloaded_mb": round(current / (1024 * 1024), 1),
                    "total_mb": round(total / (1024 * 1024), 1),
                },
            )
    return results
