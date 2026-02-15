import os
import subprocess
from unittest.mock import patch

import pytest

from slides_extractor.downloader import (
    DownloadResult,
    _normalize_proxy,
    _rename_progress_entry,
    download_video_with_ytdlp,
)
from slides_extractor.job_tracker import JOB_PROGRESS, PROGRESS_LOCK


# ---------------------------------------------------------------------------
# _normalize_proxy
# ---------------------------------------------------------------------------

class TestNormalizeProxy:
    def test_none_returns_none(self):
        assert _normalize_proxy(None) is None

    def test_short_string_returns_none(self):
        assert _normalize_proxy("ab") is None

    def test_adds_http_scheme(self):
        assert _normalize_proxy("user:pass@host:1234") == "http://user:pass@host:1234"

    def test_preserves_existing_scheme(self):
        assert _normalize_proxy("socks5://host:1234") == "socks5://host:1234"

    def test_strips_whitespace(self):
        assert _normalize_proxy("  http://host:1234  ") == "http://host:1234"


# ---------------------------------------------------------------------------
# _rename_progress_entry
# ---------------------------------------------------------------------------

class TestRenameProgressEntry:
    def setup_method(self):
        with PROGRESS_LOCK:
            JOB_PROGRESS.clear()

    def teardown_method(self):
        with PROGRESS_LOCK:
            JOB_PROGRESS.clear()

    def test_renames_existing_entry(self):
        with PROGRESS_LOCK:
            JOB_PROGRESS["old_key"] = {
                "total": 100.0, "current": 50.0,
                "status": "downloading", "start_time": 0.0,
            }

        _rename_progress_entry("old_key", "new_key")

        with PROGRESS_LOCK:
            assert "old_key" not in JOB_PROGRESS
            assert JOB_PROGRESS["new_key"]["status"] == "downloading"
            assert JOB_PROGRESS["new_key"]["current"] == 50.0

    def test_noop_when_old_key_missing(self):
        _rename_progress_entry("nonexistent", "new_key")

        with PROGRESS_LOCK:
            assert "new_key" not in JOB_PROGRESS


# ---------------------------------------------------------------------------
# download_video_with_ytdlp
# ---------------------------------------------------------------------------

VIDEO_URL = "https://www.youtube.com/watch?v=S2GChOwivwQ"
VIDEO_ID = "S2GChOwivwQ"


@pytest.fixture(autouse=True)
def _clean_progress():
    """Ensure JOB_PROGRESS is empty before and after each test."""
    with PROGRESS_LOCK:
        JOB_PROGRESS.clear()
    yield
    with PROGRESS_LOCK:
        JOB_PROGRESS.clear()


class TestDownloadVideoSuccess:
    def test_returns_success_with_destination_in_stdout(self, tmp_path, monkeypatch):
        download_dir = str(tmp_path)
        monkeypatch.setattr("slides_extractor.downloader.DOWNLOAD_DIR", download_dir)

        # Create the file that yt-dlp would have produced.
        output_file = tmp_path / "yt_S2GChOwivwQ.mp4"
        output_file.write_bytes(b"\x00" * 100)

        fake_stdout = f"[download] Destination: {output_file}\n[download] 100%\n"
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=fake_stdout, stderr=""
        )

        with patch("slides_extractor.downloader.subprocess.run", return_value=completed):
            result = download_video_with_ytdlp(VIDEO_URL, video_id=VIDEO_ID)

        assert result.success is True
        assert result.path == str(output_file)
        assert result.error is None

        # Progress entry should exist under the final filename, not video_id.
        with PROGRESS_LOCK:
            assert VIDEO_ID not in JOB_PROGRESS
            assert "yt_S2GChOwivwQ.mp4" in JOB_PROGRESS
            assert JOB_PROGRESS["yt_S2GChOwivwQ.mp4"]["status"] == "complete"

    def test_falls_back_to_scanning_directory(self, tmp_path, monkeypatch):
        download_dir = str(tmp_path)
        monkeypatch.setattr("slides_extractor.downloader.DOWNLOAD_DIR", download_dir)

        output_file = tmp_path / "yt_S2GChOwivwQ.webm"
        output_file.write_bytes(b"\x00" * 100)

        # No "Destination:" line in output.
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="[download] 100%\n", stderr=""
        )

        with patch("slides_extractor.downloader.subprocess.run", return_value=completed):
            result = download_video_with_ytdlp(VIDEO_URL, video_id=VIDEO_ID)

        assert result.success is True
        assert result.path == str(output_file)


class TestDownloadVideoFailure:
    def test_nonzero_exit_code(self, tmp_path, monkeypatch):
        monkeypatch.setattr("slides_extractor.downloader.DOWNLOAD_DIR", str(tmp_path))

        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="ERROR: some failure", stderr=""
        )

        with patch("slides_extractor.downloader.subprocess.run", return_value=completed):
            result = download_video_with_ytdlp(VIDEO_URL, video_id=VIDEO_ID)

        assert result.success is False
        assert "exited with 1" in result.error

        with PROGRESS_LOCK:
            assert JOB_PROGRESS[VIDEO_ID]["status"] == "failed"

    def test_oserror_starting_process(self, tmp_path, monkeypatch):
        monkeypatch.setattr("slides_extractor.downloader.DOWNLOAD_DIR", str(tmp_path))

        with patch(
            "slides_extractor.downloader.subprocess.run",
            side_effect=OSError("not found"),
        ):
            result = download_video_with_ytdlp(VIDEO_URL, video_id=VIDEO_ID)

        assert result.success is False
        assert "Failed to start yt-dlp" in result.error

        with PROGRESS_LOCK:
            assert JOB_PROGRESS[VIDEO_ID]["status"] == "failed"

    def test_output_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr("slides_extractor.downloader.DOWNLOAD_DIR", str(tmp_path))

        # yt-dlp succeeds but no file is created.
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="[download] 100%\n", stderr=""
        )

        with patch("slides_extractor.downloader.subprocess.run", return_value=completed):
            result = download_video_with_ytdlp(VIDEO_URL, video_id=VIDEO_ID)

        assert result.success is False
        assert "output file not found" in result.error


class TestDownloadVideoProxyHandling:
    def test_proxy_added_to_command(self, tmp_path, monkeypatch):
        monkeypatch.setattr("slides_extractor.downloader.DOWNLOAD_DIR", str(tmp_path))
        monkeypatch.setattr(
            "slides_extractor.downloader.DATACENTER_PROXY",
            "user:pass@proxy.example.com:8080",
        )

        output_file = tmp_path / "yt_S2GChOwivwQ.mp4"
        output_file.write_bytes(b"\x00" * 100)

        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=f"[download] Destination: {output_file}\n",
                stderr="",
            )

        with patch("slides_extractor.downloader.subprocess.run", side_effect=fake_run):
            download_video_with_ytdlp(VIDEO_URL, video_id=VIDEO_ID)

        assert "--proxy" in captured_cmd
        idx = captured_cmd.index("--proxy")
        assert captured_cmd[idx + 1] == "http://user:pass@proxy.example.com:8080"

    def test_no_proxy_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.setattr("slides_extractor.downloader.DOWNLOAD_DIR", str(tmp_path))
        monkeypatch.setattr("slides_extractor.downloader.DATACENTER_PROXY", None)

        output_file = tmp_path / "yt_S2GChOwivwQ.mp4"
        output_file.write_bytes(b"\x00" * 100)

        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=f"[download] Destination: {output_file}\n",
                stderr="",
            )

        with patch("slides_extractor.downloader.subprocess.run", side_effect=fake_run):
            download_video_with_ytdlp(VIDEO_URL, video_id=VIDEO_ID)

        assert "--proxy" not in captured_cmd
