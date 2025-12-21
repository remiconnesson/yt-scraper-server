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
   - `DATACENTER_PROXY` (optional, supports single or multiple comma-separated proxy IPs)
     - Single proxy: `user:pass@host:port` or full URL `http://user:pass@host:port`
     - Multiple proxies: `user:pass@host1:port,user:pass@host2:port,user:pass@host3:port`
     - When multiple proxies are configured, the system will automatically rotate between them
     - If a proxy fails (e.g., IP gets burnt by YouTube's bot detection), it will be temporarily excluded for 1 hour
     - The system will retry downloads with different proxy IPs when available
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
| `/jobs/{video_id}` | GET | Returns structured status for a specific job, including progress, messages, and metadata URLs when available. |
| `/jobs/{video_id}/stream` | GET | Streams Server-Sent Events with job updates until completion or failure. |

Example request to start a download:

```bash
curl -X POST "http://localhost:8000/process/youtube/dQw4w9WgXcQ" \
  -H "Authorization: Bearer $API_PASSWORD"
```

### Proxy Rotation and Retry Behavior

When multiple datacenter proxies are configured:

- The system automatically rotates between available proxy IPs for each download
- If a proxy fails (e.g., due to YouTube's bot detection), it is marked as "burnt" and temporarily excluded for 1 hour
- Downloads automatically retry with different proxy IPs when available
- The `/progress` endpoint will show `status: "retrying"` during proxy rotation attempts
- Consumers can monitor the job's progress through the SSE stream at `/jobs/{video_id}/stream` to observe retry attempts

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
      --from-literal=DATACENTER_PROXY=user:pass@proxy1:port,user:pass@proxy2:port
    ```

    **Note:** `DATACENTER_PROXY` now supports multiple comma-separated proxy IPs for automatic rotation and failover.

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
