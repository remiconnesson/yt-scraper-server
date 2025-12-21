"""Tests for download retry logic with proxy rotation."""

from unittest.mock import MagicMock, patch

from slides_extractor.downloader import (
    _get_proxy_config,
    _get_proxy_manager,
    _mark_proxy_burnt,
    download_file_single,
)
from slides_extractor.proxy_manager import ProxyManager


class TestProxyRetryLogic:
    """Test suite for proxy retry functionality in downloader."""

    def test_get_proxy_config_returns_next_proxy(self, monkeypatch):
        """Test that _get_proxy_config returns the next available proxy."""
        # Reset the global proxy manager with test config
        import slides_extractor.downloader as downloader_module

        downloader_module._proxy_manager = ProxyManager(
            "proxy1.com:8080,proxy2.com:8080"
        )

        proxy1 = _get_proxy_config()
        assert proxy1 is not None
        assert "http" in proxy1

        # Get next proxy should rotate
        proxy2 = _get_proxy_config()
        assert proxy2 is not None
        assert proxy2["http"] != proxy1["http"]

    def test_mark_proxy_burnt_marks_proxy_as_unavailable(self, monkeypatch):
        """Test that marking a proxy as burnt prevents its reuse."""
        # Reset the global proxy manager with test config
        import slides_extractor.downloader as downloader_module

        downloader_module._proxy_manager = ProxyManager(
            "proxy1.com:8080,proxy2.com:8080"
        )

        manager = _get_proxy_manager()

        proxy1 = _get_proxy_config()
        _mark_proxy_burnt(proxy1)

        assert manager.get_burnt_count() == 1

    def test_download_file_single_no_retry_logic(self, monkeypatch, tmp_path):
        """Test that download_file_single works with single attempt."""
        # Reset the global proxy manager with test config
        import slides_extractor.downloader as downloader_module

        downloader_module._proxy_manager = ProxyManager("proxy1.com:8080")

        # Set download directory to temp path
        monkeypatch.setattr("slides_extractor.downloader.DOWNLOAD_DIR", str(tmp_path))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "100"}
        mock_response.__enter__ = lambda self: self
        mock_response.__exit__ = lambda self, *args: None
        mock_response.iter_content = lambda chunk_size: [b"test" * 25]
        mock_response.raise_for_status = lambda: None

        with patch("requests.get", return_value=mock_response):
            result = download_file_single(
                "http://example.com/file", "test.mp4", _get_proxy_config(), 0, 1
            )

            assert result.success is True

    def test_no_proxy_config_uses_direct_connection(self, monkeypatch):
        """Test that when no proxy is configured, direct connection is used."""
        # Reset the global proxy manager with no config
        import slides_extractor.downloader as downloader_module

        downloader_module._proxy_manager = ProxyManager(None)

        proxy = _get_proxy_config()
        assert proxy == {}
