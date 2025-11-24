import asyncio

from fastapi.testclient import TestClient

from slides_extractor.app_factory import app
from slides_extractor.video_service import JOBS, JOBS_LOCK, JobStatus, update_job_status


client = TestClient(app)


def _clear_jobs() -> None:
    async def _clear() -> None:
        async with JOBS_LOCK:
            JOBS.clear()

    asyncio.run(_clear())


def setup_function() -> None:
    _clear_jobs()


def teardown_function() -> None:
    _clear_jobs()


def test_get_job_returns_latest_status() -> None:
    video_id = "job-123"
    asyncio.run(
        update_job_status(
            video_id,
            status=JobStatus.pending,
            progress=10.0,
            message="Preparing",
        )
    )

    response = client.get(f"/jobs/{video_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == JobStatus.pending.value
    assert payload["progress"] == 10.0
    assert payload["message"] == "Preparing"


def test_stream_job_sends_completion_event() -> None:
    video_id = "job-stream"
    asyncio.run(
        update_job_status(
            video_id,
            status=JobStatus.completed,
            progress=100.0,
            message="Done",
        )
    )

    with client.stream("GET", f"/jobs/{video_id}/stream") as response:
        assert response.status_code == 200
        events = [line.decode() if isinstance(line, bytes) else line for line in response.iter_lines() if line]

    assert any(event.startswith("data: ") for event in events)


def test_get_job_404s_when_missing() -> None:
    response = client.get("/jobs/missing-job")

    assert response.status_code == 404
    assert "Job not found" in response.json()["detail"]
