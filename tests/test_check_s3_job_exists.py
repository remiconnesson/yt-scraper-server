import pytest

from slides_extractor import video_service


def _configure_s3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_service, "S3_ENDPOINT", "https://example.com")
    monkeypatch.setattr(video_service, "S3_ACCESS_KEY", "test-key")
    monkeypatch.setattr(video_service, "S3_BUCKET_NAME", "bucket")
    monkeypatch.setattr(video_service, "_S3_CLIENT", None)


def test_check_s3_job_exists_ignores_directory_marker(monkeypatch: pytest.MonkeyPatch):
    _configure_s3(monkeypatch)

    class FakeS3:
        @staticmethod
        def head_object(Bucket: str, Key: str):  # noqa: N802 - boto3 style
            assert Bucket == "bucket"
            assert Key == "video/abc123/video_segments.json"
            return {"ContentType": "application/x-directory", "ContentLength": 0}

    monkeypatch.setattr(video_service, "_get_s3_client", lambda: FakeS3())
    monkeypatch.setattr(
        video_service, "_get_s3_endpoint", lambda: "https://example.com"
    )

    assert video_service.check_s3_job_exists("abc123") is None


def test_check_s3_job_exists_returns_url_for_manifest(monkeypatch: pytest.MonkeyPatch):
    _configure_s3(monkeypatch)

    class FakeS3:
        @staticmethod
        def head_object(Bucket: str, Key: str):  # noqa: N802 - boto3 style
            assert Bucket == "bucket"
            assert Key == "video/xyz789/video_segments.json"
            return {"ContentType": "application/json", "ContentLength": 128}

    monkeypatch.setattr(video_service, "_get_s3_client", lambda: FakeS3())
    monkeypatch.setattr(
        video_service, "_get_s3_endpoint", lambda: "https://example.com"
    )

    assert (
        video_service.check_s3_job_exists("xyz789")
        == "https://example.com/bucket/video/xyz789/video_segments.json"
    )
