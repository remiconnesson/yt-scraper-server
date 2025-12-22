import asyncio
import os

import pytest
from fastapi.testclient import TestClient

os.environ["API_PASSWORD"] = "testpassword"  # noqa: S105

from slides_extractor.app_factory import (
    READINESS_EVENT,
    SHUTDOWN_REQUESTED,
    app,
    reset_shutdown_state,
    reset_shutdown_state_for_tests,
    wait_for_active_jobs,
)
from slides_extractor.video_service import JOBS, JOBS_LOCK, JobStatus


async def _clear_jobs() -> None:
    async with JOBS_LOCK:
        JOBS.clear()


def test_readiness_probe_reflects_drain_state():
    reset_shutdown_state()

    with TestClient(app) as client:
        client.headers.update({"Authorization": "Bearer testpassword"})  # noqa: S106

        ready_response = client.get("/healthz/ready")
        assert ready_response.status_code == 200

        drain_response = client.post("/drain")
        assert drain_response.status_code == 202

        draining_response = client.get("/healthz/ready")
        assert draining_response.status_code == 503
        assert SHUTDOWN_REQUESTED.is_set()
        assert not READINESS_EVENT.is_set()

    asyncio.run(_clear_jobs())
    reset_shutdown_state()


def test_process_rejects_when_draining():
    reset_shutdown_state()

    with TestClient(app) as client:
        client.headers.update({"Authorization": "Bearer testpassword"})  # noqa: S106

        drain_response = client.post("/drain")
        assert drain_response.status_code == 202

        process_response = client.post("/process/youtube/testid")

        assert process_response.status_code == 503
        assert process_response.json()["detail"] == "Draining for shutdown"

    asyncio.run(_clear_jobs())
    reset_shutdown_state()


@pytest.mark.asyncio
async def test_wait_for_active_jobs_allows_completion():
    reset_shutdown_state()

    async with JOBS_LOCK:
        JOBS["abc123"] = {"status": JobStatus.downloading}

    waiter = asyncio.create_task(wait_for_active_jobs(timeout_seconds=2))
    await asyncio.sleep(0.1)

    async with JOBS_LOCK:
        JOBS["abc123"]["status"] = JobStatus.completed

    await waiter
    await _clear_jobs()
    reset_shutdown_state_for_tests()


def test_drain_rejects_missing_authentication():
    reset_shutdown_state_for_tests()

    with TestClient(app) as client:
        response = client.post("/drain")

        assert response.status_code == 401
        assert not SHUTDOWN_REQUESTED.is_set()

    reset_shutdown_state_for_tests()
