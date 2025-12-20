import pytest
from slides_extractor import settings, video_service

from slides_extractor.constants import MANIFEST_PATH_TEMPLATE


def _configure_blob(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "BLOB_READ_WRITE_TOKEN", "test-token")


class FakeBlob:
    def __init__(self, pathname, url, size):
        self.pathname = pathname
        self.url = url
        self.size = size


class FakeResponse:
    def __init__(self, blobs):
        self.blobs = blobs


@pytest.mark.asyncio
async def test_check_blob_job_exists_returns_none_when_no_token(
    monkeypatch: pytest.MonkeyPatch,
):
    # TODO: wait why do we allow to return None here?
    monkeypatch.setattr(settings, "BLOB_READ_WRITE_TOKEN", None)
    assert await video_service.check_blob_job_exists("abc123") is None


@pytest.mark.asyncio
async def test_check_blob_job_exists_returns_url_for_manifest(
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_blob(monkeypatch)

    class FakeClient:
        async def list_objects(self, prefix: str):
            return FakeResponse(
                [
                    FakeBlob(
                        MANIFEST_PATH_TEMPLATE.format(video_id="xyz789"),
                        "https://blob.vercel-storage.com/xyz789",
                        128,
                    )
                ]
            )

    monkeypatch.setattr(video_service, "AsyncBlobClient", lambda: FakeClient())

    assert (
        await video_service.check_blob_job_exists("xyz789")
        == "https://blob.vercel-storage.com/xyz789"
    )


@pytest.mark.asyncio
async def test_check_blob_job_exists_returns_none_if_not_found(
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_blob(monkeypatch)

    class FakeClient:
        async def list_objects(self, prefix: str):
            return FakeResponse([])

    monkeypatch.setattr(video_service, "AsyncBlobClient", lambda: FakeClient())

    assert await video_service.check_blob_job_exists("notfound") is None
