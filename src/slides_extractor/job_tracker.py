import asyncio
import logging
import time
from typing import Any, Coroutine, Dict, Optional, TypeVar, TypedDict, cast


logger = logging.getLogger("scraper")


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
PROGRESS_LOCK: asyncio.Lock | None = None
EVENT_LOOP: asyncio.AbstractEventLoop | None = None

T = TypeVar("T")


async def capture_event_loop() -> None:
    """Capture the running event loop for cross-thread progress updates."""

    global EVENT_LOOP
    EVENT_LOOP = asyncio.get_running_loop()


async def _ensure_progress_lock() -> asyncio.Lock:
    """Lazy-create the asyncio lock after the event loop is running."""

    global PROGRESS_LOCK
    if PROGRESS_LOCK is None:
        PROGRESS_LOCK = asyncio.Lock()
    return PROGRESS_LOCK


async def _update_progress(
    filename: str,
    bytes_added: int = 0,
    total_size: Optional[int] = None,
    status: Optional[str] = None,
) -> None:
    lock = await _ensure_progress_lock()
    async with lock:
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


async def _remove_progress_entry(filename: str) -> None:
    """Drop a progress entry if it exists, guarding with the async lock."""

    lock = await _ensure_progress_lock()
    async with lock:
        JOB_PROGRESS.pop(filename, None)


def _ensure_sync_entrypoint(sync_only_message: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    else:
        raise RuntimeError(sync_only_message)


def _run_in_event_loop(coro: Coroutine[Any, Any, T], async_use_hint: str) -> T:
    """Bridge a coroutine into the app event loop from sync contexts."""

    _ensure_sync_entrypoint(async_use_hint)

    if EVENT_LOOP and EVENT_LOOP.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, EVENT_LOOP)
        return future.result()

    raise RuntimeError(
        "Event loop not initialized; progress updates require FastAPI startup to complete."
    )


def update_progress(
    filename: str,
    bytes_added: int = 0,
    total_size: Optional[int] = None,
    status: Optional[str] = None,
) -> None:
    """Thread-safe wrapper to update download progress from sync code paths."""

    _run_in_event_loop(
        _update_progress(
            filename, bytes_added=bytes_added, total_size=total_size, status=status
        ),
        "update_progress() cannot be used from async contexts; call `_update_progress` directly instead.",
    )


def remove_progress_entry(filename: str) -> None:
    """Thread-safe wrapper to prune a progress record from sync code paths."""

    try:
        _run_in_event_loop(
            _remove_progress_entry(filename),
            "remove_progress_entry() cannot be used from async contexts; call `_remove_progress_entry` directly instead.",
        )
    except RuntimeError as exc:
        logger.warning("%s", exc)


async def progress_snapshot() -> dict[str, ProgressSnapshot]:
    """Return a copy of the progress table with percentage calculations."""

    results: dict[str, ProgressSnapshot] = {}
    lock = await _ensure_progress_lock()
    async with lock:
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

