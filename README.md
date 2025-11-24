# Slides Extractor

A python package with several concerns:
1. Downloading YouTube videos and audio streams
2. Extracting slides from videos frames
3. Uploading slides to S3 with metadata about the timestamps of the slides in the video
4. Running a webserver and update clients about the progress of the downloads and the slides extraction

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
   - `S3_ENDPOINT` (required for slide upload)
   - `S3_ACCESS_KEY` (required)
   - `S3_BUCKET_NAME` (optional, defaults to `slides-extractor`)

   **Note:** S3 uploads are private. Consumers must have valid credentials or generate presigned URLs to access the uploaded slides.

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
| `/jobs/{video_id}` | GET | Returns structured status for a specific job, including progress, messages, and metadata URLs when available. |
| `/jobs/{video_id}/stream` | GET | Streams Server-Sent Events with job updates until completion or failure. |

Example request to start a download:

```bash
curl -X POST "http://localhost:8000/process/youtube/dQw4w9WgXcQ" \
  -H "Authorization: Bearer $API_PASSWORD"
```

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
      --from-literal=S3_ENDPOINT=your_s3_endpoint
    ```

2.  **Create the secrets:**

    ```bash
    kubectl create secret generic slides-extractor-secrets \
      --from-literal=ZYTE_API_KEY=your_zyte_api_key \
      --from-literal=S3_ACCESS_KEY=your_s3_access_key \
      --from-literal=API_PASSWORD=your_api_password \
      --from-literal=DATACENTER_PROXY=DATACENTER_PROXY

    ```

3. **Build and push the image to the prviate local registry:**

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
