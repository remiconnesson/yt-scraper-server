- Use Python uv for dependency management.
- Enforce strict type checking with `ty` and formatting/linting with `ruff` (add them as dev dependencies).
- Run tests with `pytest` to validate changes.
- Prefer test-driven development when applicable.

## Project context

- FastAPI entry point lives in `main.py` with routes for `/scrape`, `/progress`, `/logs`, `/list`, and static files under `/files`.
- Download orchestration runs in background tasks so request threads stay responsive.
- Stream URLs are resolved via `get_stream_urls`, and `download_file_parallel` performs chunked downloads with a `PROGRESS_LOCK` to keep progress updates thread-safe.
- `video_service.py` holds the Pydantic models and enums used for job tracking and S3 metadata.
- CLI entry points (`extract-slides`, `video-static-segments`) live under `src/extract_slides/`.

## Change management tips

- Keep logging clear and actionable because `/logs` exposes recent lines directly to callers.
- Preserve the `/progress` response shape (status, percent, downloaded_mb, total_mb) for compatibility with existing clients.
- When adding API routes, register them in `main.py` and prefer background tasks or dependencies to avoid blocking requests.
