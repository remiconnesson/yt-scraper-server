import os
from datetime import datetime, timezone

import pytest

from slides_extractor import settings, video_service

os.environ.setdefault("API_PASSWORD", "testpassword")

from fastapi.testclient import TestClient

from slides_extractor.app_factory import app


client = TestClient(app)
client.headers.update({"Authorization": "Bearer testpassword"})


class FakeBlob:
    def __init__(self, pathname, url, size, uploaded_at=None):
        self.pathname = pathname
        self.url = url
        self.size = size
        self.uploaded_at = uploaded_at or datetime.now(timezone.utc)


class FakeListResponse:
    def __init__(self, blobs, cursor=None, has_more=False):
        self.blobs = blobs
        self.cursor = cursor
        self.has_more = has_more


def _configure_blob(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "BLOB_READ_WRITE_TOKEN", "test-token")


@pytest.mark.asyncio
async def test_list_processed_videos_returns_empty_without_token(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(settings, "BLOB_READ_WRITE_TOKEN", None)

    result = await video_service.list_processed_videos()
    assert result == {"videos": [], "cursor": None, "has_more": False}


@pytest.mark.asyncio
async def test_list_processed_videos_returns_video_entries(
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_blob(monkeypatch)
    uploaded = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    class FakeClient:
        async def list_objects(self, prefix=None, limit=None, cursor=None):
            return FakeListResponse(
                blobs=[
                    FakeBlob(
                        "manifests/abc123.json",
                        "https://blob.vercel-storage.com/manifests/abc123.json",
                        256,
                        uploaded,
                    ),
                    FakeBlob(
                        "manifests/def456.json",
                        "https://blob.vercel-storage.com/manifests/def456.json",
                        512,
                        uploaded,
                    ),
                ],
                cursor=None,
                has_more=False,
            )

    monkeypatch.setattr(video_service, "AsyncBlobClient", lambda: FakeClient())

    result = await video_service.list_processed_videos()
    assert len(result["videos"]) == 2
    assert result["videos"][0]["video_id"] == "abc123"
    assert result["videos"][1]["video_id"] == "def456"
    assert result["has_more"] is False
    assert result["cursor"] is None


@pytest.mark.asyncio
async def test_list_processed_videos_pagination(
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_blob(monkeypatch)

    class FakeClient:
        async def list_objects(self, prefix=None, limit=None, cursor=None):
            if cursor is None:
                return FakeListResponse(
                    blobs=[
                        FakeBlob(
                            "manifests/page1.json",
                            "https://blob.vercel-storage.com/manifests/page1.json",
                            100,
                        ),
                    ],
                    cursor="next-cursor-token",
                    has_more=True,
                )
            else:
                return FakeListResponse(
                    blobs=[
                        FakeBlob(
                            "manifests/page2.json",
                            "https://blob.vercel-storage.com/manifests/page2.json",
                            200,
                        ),
                    ],
                    cursor=None,
                    has_more=False,
                )

    monkeypatch.setattr(video_service, "AsyncBlobClient", lambda: FakeClient())

    # First page
    result = await video_service.list_processed_videos(limit=1)
    assert len(result["videos"]) == 1
    assert result["videos"][0]["video_id"] == "page1"
    assert result["has_more"] is True
    assert result["cursor"] == "next-cursor-token"

    # Second page
    result = await video_service.list_processed_videos(
        cursor="next-cursor-token", limit=1
    )
    assert len(result["videos"]) == 1
    assert result["videos"][0]["video_id"] == "page2"
    assert result["has_more"] is False


@pytest.mark.asyncio
async def test_list_processed_videos_skips_non_manifest_blobs(
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_blob(monkeypatch)

    class FakeClient:
        async def list_objects(self, prefix=None, limit=None, cursor=None):
            return FakeListResponse(
                blobs=[
                    FakeBlob("manifests/good.json", "https://example.com/good", 100),
                    FakeBlob("manifests/", "https://example.com/folder", 0),
                    FakeBlob("manifests/no-ext", "https://example.com/noext", 50),
                ],
            )

    monkeypatch.setattr(video_service, "AsyncBlobClient", lambda: FakeClient())

    result = await video_service.list_processed_videos()
    assert len(result["videos"]) == 1
    assert result["videos"][0]["video_id"] == "good"


def test_processed_videos_endpoint_requires_auth():
    response = TestClient(app).get("/videos/processed")
    assert response.status_code == 401


def test_processed_videos_endpoint(monkeypatch: pytest.MonkeyPatch):
    _configure_blob(monkeypatch)

    class FakeClient:
        async def list_objects(self, prefix=None, limit=None, cursor=None):
            return FakeListResponse(
                blobs=[
                    FakeBlob(
                        "manifests/vid1.json",
                        "https://blob.vercel-storage.com/manifests/vid1.json",
                        128,
                    ),
                ],
            )

    monkeypatch.setattr(video_service, "AsyncBlobClient", lambda: FakeClient())

    response = client.get("/videos/processed")
    assert response.status_code == 200
    data = response.json()
    assert len(data["videos"]) == 1
    assert data["videos"][0]["video_id"] == "vid1"
    assert data["has_more"] is False
