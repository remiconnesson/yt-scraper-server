from types import SimpleNamespace

from slides_extractor import downloader


def test_download_youtube_with_ytdlp_success(monkeypatch, tmp_path):
    output_file = tmp_path / "abc123.mp4"
    output_file.write_bytes(b"video")

    recorded = {}

    def fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        return SimpleNamespace(
            returncode=0,
            stdout=f"[download] Destination: {output_file}\n",
        )

    statuses: list[tuple[str, str]] = []

    def fake_update(name: str, status: str | None = None, **_kwargs):
        if status is not None:
            statuses.append((name, status))

    monkeypatch.setattr(downloader, "DOWNLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(downloader.subprocess, "run", fake_run)
    monkeypatch.setattr(downloader, "update_progress", fake_update)

    result = downloader.download_youtube_with_ytdlp(
        "S2GChOwivwQ",
        cookies_path="yt_cookies.txt",
        proxy="user:pass@127.0.0.1:3128",
        filename_prefix="abc",
    )

    assert result.success is True
    assert result.path == str(output_file)
    assert "--proxy" in recorded["cmd"]
    assert "http://user:pass@127.0.0.1:3128" in recorded["cmd"]
    assert statuses[0][1] == "downloading"
    assert statuses[-1] == (output_file.name, "complete")


def test_download_youtube_with_ytdlp_failure(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="boom")

    statuses: list[tuple[str, str]] = []

    def fake_update(name: str, status: str | None = None, **_kwargs):
        if status is not None:
            statuses.append((name, status))

    monkeypatch.setattr(downloader, "DOWNLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(downloader.subprocess, "run", fake_run)
    monkeypatch.setattr(downloader, "update_progress", fake_update)

    result = downloader.download_youtube_with_ytdlp(
        "S2GChOwivwQ", filename_prefix="abc"
    )

    assert result.success is False
    assert result.error == "yt-dlp exited with 1"
    assert statuses[-1][1] == "failed"


def test_normalize_proxy_adds_scheme():
    assert (
        downloader._normalize_proxy("user:pass@185.162.178.126:44072")
        == "http://user:pass@185.162.178.126:44072"
    )
    assert (
        downloader._normalize_proxy("http://user:pass@185.162.178.126:44072")
        == "http://user:pass@185.162.178.126:44072"
    )
