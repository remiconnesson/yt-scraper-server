# Turbo Scraper Server

A FastAPI-based service for downloading YouTube (and other HTTP) video and audio streams through configurable proxies. Downloads run as background jobs with progress tracking, log inspection, and static hosting of completed files.

## Features

- Background download pipeline that fetches the best available video and audio streams concurrently via `yt-dlp`.
- Chunked, multi-threaded HTTP downloading with optional proxy support (Zyte or datacenter proxies).
- Progress tracking endpoint that reports per-file status and percentage completed.
- Recent log retrieval for quick debugging.
- Static file hosting for completed downloads under `/files`.
- CLI utilities for slide extraction and detecting static segments in videos (see below).

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) for dependency management (preferred)
- `yt-dlp` dependencies suitable for your platform
- Optional: Zyte API key or datacenter proxy for remote fetching

## Quickstart

1. Create and activate a virtual environment using uv:

   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```

2. Configure environment variables (create a `.env` file or export directly):

   - `ZYTE_API_KEY` (required for Zyte proxy usage)
   - `ZYTE_HOST` (defaults to `api.zyte.com`)
   - `DATACENTER_PROXY` (optional `user:pass@host:port` or full URL)

3. Run the API server:

   ```bash
   uv run uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## API Overview

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Service status and quick links to other endpoints. |
| `/scrape?url=...` | GET | Starts a background job to download video and audio streams for the given URL. |
| `/progress` | GET | Returns per-file download status, percent completion, and size information. |
| `/logs` | GET | Returns the 50 most recent log lines. |
| `/list` | GET | Lists completed download files with links served from `/files/{name}`. |

Example request to start a download:

```bash
curl "http://localhost:8000/scrape?url=https://www.youtube.com/watch?v=VIDEO_ID"
```

After the request returns, poll `/progress` to track job completion and retrieve files from `/list` once complete.

## Architecture overview

- **Entry point**: `main.py` wires up FastAPI routes for starting downloads (`/scrape`), polling progress (`/progress`), viewing logs (`/logs`), listing completed files (`/list`), and serving static files from `downloads/` via `/files`.
- **Download orchestration**: `process_video_task` coordinates concurrent video and audio downloads in a background task while keeping the request thread responsive.
- **Stream resolution**: `get_stream_urls` fetches the best video (up to 1000p) and audio-only HTTP streams through Zyte when configured.
- **Chunked transfer**: `download_file_parallel` handles multi-threaded chunk downloads with a `PROGRESS_LOCK` to keep in-memory progress updates thread-safe.
- **Data models**: `video_service.py` holds Pydantic models and enums for job tracking and S3 metadata.

### Download entry point

Use the FastAPI pipeline in `main.py` for all YouTube downloads. The previous standalone helper (`download_video.py`) duplicated logic and has been removed to keep behavior consistent with the multi-threaded downloader and progress tracking exposed by the API.

## How downloads work

- `yt-dlp` resolves the best video (up to 1000p) and audio-only HTTP streams through the Zyte proxy when configured.
- Video chunks download in parallel (32 threads by default) while audio downloads in parallel with fewer threads.
- Progress and timing data are stored in memory for quick polling; logs are written to `app.log`.
- Completed files are served from the `downloads/` directory via FastAPI's static file mount.

## Development workflow

Use the supplied development tooling to keep the project consistent:

```bash
# Format and lint
uv run ruff format .
uv run ruff check .

# Type checking (strict)
uv run ty

# Run tests
uv run pytest
```

## CLI utilities

The repository still ships the `extract-slides` command group for working with presentation files and detecting static segments in videos:

- `extract-slides` – Extract slides from a presentation file. Example:

  ```bash
  uv run extract-slides extract input.pptx -o output --format png
  ```

- `video-static-segments` – Analyze a video for static frames using perceptual hashing:

  ```bash
  uv run video-static-segments --input demo.mp4 --output-dir analysis
  ```

These commands live under `src/extract_slides/` and share the same virtual environment as the API server.
