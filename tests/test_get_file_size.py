import requests
from requests import RequestException

from main import get_file_size


class DummyResponse:
    def __init__(self, status_code: int, headers: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}


HEADERS = {"User-Agent": "pytest"}


def test_get_file_size_prefers_clen(monkeypatch):
    calls = {"head": 0, "get": 0}

    def fake_head(*args, **kwargs):
        calls["head"] += 1
        return DummyResponse(404)

    def fake_get(*args, **kwargs):
        calls["get"] += 1
        return DummyResponse(404)

    monkeypatch.setattr(requests, "head", fake_head)
    monkeypatch.setattr(requests, "get", fake_get)

    size = get_file_size("https://example.com/video?clen=12345", HEADERS, {})

    assert size == 12345
    assert calls == {"head": 0, "get": 0}


def test_get_file_size_uses_head(monkeypatch):
    def fake_head(*args, **kwargs):
        return DummyResponse(200, {"content-length": "54321"})

    def fake_get(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("Range probe should not run when HEAD succeeds")

    monkeypatch.setattr(requests, "head", fake_head)
    monkeypatch.setattr(requests, "get", fake_get)

    size = get_file_size("https://example.com/video", HEADERS, {})

    assert size == 54321


def test_get_file_size_uses_range_probe(monkeypatch):
    def fake_head(*args, **kwargs):
        return DummyResponse(200, {})

    def fake_get(*args, **kwargs):
        return DummyResponse(206, {"Content-Range": "bytes 0-0/999"})

    monkeypatch.setattr(requests, "head", fake_head)
    monkeypatch.setattr(requests, "get", fake_get)

    size = get_file_size("https://example.com/video", HEADERS, {})

    assert size == 999


def test_get_file_size_handles_failures(monkeypatch):
    def fake_head(*args, **kwargs):
        raise RequestException("network issue")

    def fake_get(*args, **kwargs):
        return DummyResponse(404, {})

    monkeypatch.setattr(requests, "head", fake_head)
    monkeypatch.setattr(requests, "get", fake_get)

    size = get_file_size("https://example.com/video", HEADERS, {})

    assert size == 0
