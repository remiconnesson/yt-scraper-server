# Slides Extractor

A python package with several concerns:
1. Downloading YouTube videos and audio streams
2. Extracting slides from videos frames
3. Uploading slides to Vercel Blob Storage with a deterministic naming strategy
4. Running a webserver and updating clients about the progress of downloads and extraction

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Zyte API key and datacenter proxy for remote fetching of YouTube videos

## Quickstart

1. Install dependencies:

   ```bash
   uv sync --dev
   ```

2. Configure environment variables (create a `.env` file or export directly):

   - `ZYTE_API_KEY` (required for Zyte proxy usage)
   - `ZYTE_HOST` (defaults to `api.zyte.com`)
   - `DATACENTER_PROXY` (optional `user:pass@host:port` or full URL)
   - `API_PASSWORD` (required for authenticating API requests)
   - `BLOB_READ_WRITE_TOKEN` (required for Vercel Blob upload)

   **Note:** Vercel Blob uploads follow a deterministic naming strategy: `slides/{videoId}/{slideIndex}-{framePosition}.webp` and `manifests/{videoId}`.

3. Run the API server:

   ```bash
   uv run uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## API Overview

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Service status and quick links to other endpoints. |
| `/process/youtube/{video_id}` | POST | Starts a background job to download video and audio streams for the given YouTube video ID. |
| `/progress` | GET | Returns per-file download status, percent completion, and size information. |
| `/jobs/{video_id}` | GET | Returns structured status for a specific job, including status, messages, and metadata URLs when available. See Job Status Structure below. |
| `/jobs/{video_id}/stream` | GET | Streams Server-Sent Events with job updates until completion or failure. See Job Status Structure below. |

Example request to start a download:

```bash
curl -X POST "http://localhost:8000/process/youtube/dQw4w9WgXcQ" \
  -H "Authorization: Bearer $API_PASSWORD"
```

## Job Status Structure

The `/jobs/{video_id}` and `/jobs/{video_id}/stream` endpoints return job status updates with the following structure:

```json
{
  "status": "extracting",
  "message": "Analyzing frames: 42 segments detected",
  "updated_at": "2025-12-21T15:36:34.590Z",
  "metadata_uri": null,
  "error": null,
  "frame_count": 1000,
  "current_frame": 542,
  "slides_processed": null,
  "total_slides": null
}
```

### Job Status Lifecycle

Jobs progress through the following status values in order:

1. **`pending`**: Job has been accepted and is waiting to start
   - Fields: `status`, `message`, `updated_at`

2. **`downloading`**: Video streams are being downloaded (handled by downloader module)
   - Fields: Same as pending, plus download progress tracked separately via `/progress` endpoint

3. **`extracting`**: Frames are being analyzed to detect slide segments
   - Fields: `frame_count` (total frames in video), `current_frame` (current frame being analyzed)
   - Updates sent approximately every 100 frames

4. **`uploading`**: Detected slides are being uploaded to storage
   - Fields: `slides_processed` (number of slides uploaded), `total_slides` (total slides to upload)
   - Updates sent after each slide is processed

5. **`completed`**: All processing finished successfully
   - Fields: `metadata_uri` (URL to manifest JSON with all slide metadata), `frame_count`

6. **`failed`**: Processing encountered an error
   - Fields: `error` (description of what went wrong)

### Field Descriptions

| Field | Type | Description |
| --- | --- | --- |
| `status` | string | Current job status (see lifecycle above) |
| `message` | string | Human-readable status message |
| `updated_at` | string | ISO 8601 timestamp of last update |
| `metadata_uri` | string\|null | URL to manifest JSON (only set on completion) |
| `error` | string\|null | Error description (only set on failure) |
| `frame_count` | int\|null | Total frames in video (set during extracting phase) |
| `current_frame` | int\|null | Current frame being analyzed (only during extracting) |
| `slides_processed` | int\|null | Slides uploaded so far (only during uploading) |
| `total_slides` | int\|null | Total slides to upload (only during uploading) |

**Note**: Fields not applicable to the current status will be `null`.

## Development workflow

Use the supplied development tooling to keep the project consistent:

```bash
# Format and lint
uv run ruff format .
uv run ruff check .

# Type checking (strict)
uv run ty
uv run mypy

# Run tests
uv run pytest
```

Before running the commands above, install dependencies with uv to ensure the
headless OpenCV build is used (avoiding `libGL.so.1` import errors) and dev
tools like `ruff` and `ty` are available:

```bash
uv sync --dev
```

## Deployment

To deploy the application to Kubernetes:

1.  **Create the configuration ConfigMap:**

    ```bash
    kubectl create configmap slides-extractor-config \
      --from-literal=ZYTE_HOST=api.zyte.com
    ```

2.  **Create the secrets:**

    ```bash
    kubectl create secret generic slides-extractor-secrets \
      --from-literal=ZYTE_API_KEY=your_zyte_api_key \
      --from-literal=BLOB_READ_WRITE_TOKEN=your_vercel_blob_token \
      --from-literal=API_PASSWORD=your_api_password \
      --from-literal=DATACENTER_PROXY=DATACENTER_PROXY
    ```

3. **Build and push the image to the private local registry:**

    ```bash
    TAG=v0.0.0

    sudo docker build -t registry.localhost:5000/slides-extractor:${TAG} .
    sudo docker push registry.localhost:5000/slides-extractor:${TAG}
    ```

4. **Update the deployment manifest to use the new image:**

    ```bash
    sed -i "s#image: .*slides-extractor.*#image: registry.localhost:5000/slides-extractor:${TAG}#g" deploy/prod.yaml
    ```

3.  **Apply the deployment manifest:**

    ```bash
    kubectl apply -f deploy/prod.yaml
    ```

### Graceful shutdown during rollouts

Deployments are configured to let in-progress work finish before a pod is
terminated:

- A `preStop` hook calls `POST /drain` (authenticated with the shared API
  password) to mark the pod as **not ready** via `/healthz/ready`, stopping new
  work from being scheduled while shutdown begins.
- The application tracks active jobs and download progress; when draining it
  polls those states until everything is complete or the grace period expires
  (configured by `GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS`, default 1800 seconds).
- The Deployment uses `terminationGracePeriodSeconds: 1800`, `maxUnavailable: 0`,
  and `maxSurge: 1` so new pods come up before old ones are removed, giving
  long-running jobs time to finish.

During a rollout you can observe readiness flips at `/healthz/ready`, while
`/healthz/live` stays healthy unless the process crashes. The shutdown window
should cover the longest expected job duration to avoid forced termination.
