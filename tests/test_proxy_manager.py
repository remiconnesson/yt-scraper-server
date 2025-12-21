"""Tests for ProxyManager."""

import time
from unittest.mock import patch

import pytest

from slides_extractor.proxy_manager import ProxyManager


class TestProxyManager:
    """Test suite for ProxyManager class."""

    def test_initialization_empty_config(self):
        """Test initialization with no proxy config."""
        manager = ProxyManager(None)
        assert manager.get_proxy_count() == 0
        assert manager.get_next_proxy() is None

    def test_initialization_single_proxy(self):
        """Test initialization with a single proxy."""
        manager = ProxyManager("user:pass@proxy1.com:8080")
        assert manager.get_proxy_count() == 1
        
        proxy = manager.get_next_proxy()
        assert proxy is not None
        assert proxy["http"] == "http://user:pass@proxy1.com:8080"
        assert proxy["https"] == "http://user:pass@proxy1.com:8080"

    def test_initialization_multiple_proxies(self):
        """Test initialization with multiple comma-separated proxies."""
        manager = ProxyManager(
            "user1:pass1@proxy1.com:8080,user2:pass2@proxy2.com:8080,proxy3.com:8080"
        )
        assert manager.get_proxy_count() == 3

    def test_proxy_rotation(self):
        """Test that proxies are rotated in round-robin fashion."""
        manager = ProxyManager("proxy1.com:8080,proxy2.com:8080,proxy3.com:8080")
        
        # Get proxies in sequence
        proxy1 = manager.get_next_proxy()
        proxy2 = manager.get_next_proxy()
        proxy3 = manager.get_next_proxy()
        proxy4 = manager.get_next_proxy()  # Should wrap back to first
        
        assert proxy1 is not None
        assert proxy2 is not None
        assert proxy3 is not None
        assert proxy4 is not None
        
        # Verify rotation
        assert proxy1["http"] != proxy2["http"]
        assert proxy2["http"] != proxy3["http"]
        assert proxy1["http"] == proxy4["http"]  # Wrapped around

    def test_mark_proxy_burnt(self):
        """Test marking a proxy as burnt."""
        manager = ProxyManager("proxy1.com:8080,proxy2.com:8080")
        
        proxy1 = manager.get_next_proxy()
        assert proxy1 is not None
        
        # Mark first proxy as burnt
        manager.mark_burnt(proxy1)
        assert manager.get_burnt_count() == 1
        
        # Next proxy should be the second one
        proxy2 = manager.get_next_proxy()
        assert proxy2 is not None
        assert proxy2["http"] != proxy1["http"]

    def test_all_proxies_burnt(self):
        """Test behavior when all proxies are burnt."""
        manager = ProxyManager("proxy1.com:8080,proxy2.com:8080")
        
        # Burn all proxies
        proxy1 = manager.get_next_proxy()
        proxy2 = manager.get_next_proxy()
        manager.mark_burnt(proxy1)
        manager.mark_burnt(proxy2)
        
        assert manager.get_burnt_count() == 2
        assert not manager.has_available_proxies()
        assert manager.get_next_proxy() is None

    def test_proxy_cooldown_expiration(self):
        """Test that burnt proxies become available after cooldown."""
        manager = ProxyManager("proxy1.com:8080")
        
        proxy = manager.get_next_proxy()
        assert proxy is not None
        
        # Mark as burnt
        manager.mark_burnt(proxy)
        assert manager.get_burnt_count() == 1
        assert manager.get_next_proxy() is None
        
        # Mock time to simulate cooldown expiration
        current_time = time.time()
        with patch("time.time") as mock_time:
            mock_time.return_value = current_time + ProxyManager.BURNT_IP_COOLDOWN_SECONDS + 1
            
            # Need to patch in proxy_manager module as well
            with patch("slides_extractor.proxy_manager.time.time") as mock_time_pm:
                mock_time_pm.return_value = current_time + ProxyManager.BURNT_IP_COOLDOWN_SECONDS + 1
                
                # Proxy should be available again
                assert manager.has_available_proxies()
                new_proxy = manager.get_next_proxy()
                assert new_proxy is not None
                assert new_proxy["http"] == proxy["http"]

    def test_proxy_format_normalization(self):
        """Test that various proxy formats are normalized correctly."""
        # Without http:// prefix
        manager1 = ProxyManager("proxy.com:8080")
        proxy1 = manager1.get_next_proxy()
        assert proxy1 is not None
        assert proxy1["http"].startswith("http://")
        
        # With http:// prefix
        manager2 = ProxyManager("http://proxy.com:8080")
        proxy2 = manager2.get_next_proxy()
        assert proxy2 is not None
        assert proxy2["http"].startswith("http://")
        
        # With credentials
        manager3 = ProxyManager("user:pass@proxy.com:8080")
        proxy3 = manager3.get_next_proxy()
        assert proxy3 is not None
        assert "user:pass" in proxy3["http"]

    def test_has_available_proxies(self):
        """Test the has_available_proxies method."""
        manager = ProxyManager("proxy1.com:8080,proxy2.com:8080")
        
        assert manager.has_available_proxies()
        
        # Burn one proxy
        proxy = manager.get_next_proxy()
        manager.mark_burnt(proxy)
        assert manager.has_available_proxies()  # Still one left
        
        # Burn second proxy
        proxy = manager.get_next_proxy()
        manager.mark_burnt(proxy)
        assert not manager.has_available_proxies()

    def test_whitespace_handling(self):
        """Test that whitespace in proxy config is handled correctly."""
        manager = ProxyManager("  proxy1.com:8080  ,  proxy2.com:8080  ")
        assert manager.get_proxy_count() == 2
        
        proxy = manager.get_next_proxy()
        assert proxy is not None
        assert "  " not in proxy["http"]

    def test_empty_string_config(self):
        """Test initialization with empty string config."""
        manager = ProxyManager("")
        assert manager.get_proxy_count() == 0
        assert manager.get_next_proxy() is None

    def test_concurrent_access(self):
        """Test that proxy manager is thread-safe."""
        import threading
        
        manager = ProxyManager("proxy1.com:8080,proxy2.com:8080,proxy3.com:8080")
        results = []
        
        def get_proxy():
            proxy = manager.get_next_proxy()
            results.append(proxy)
        
        threads = [threading.Thread(target=get_proxy) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All threads should have gotten a proxy
        assert len(results) == 10
        assert all(proxy is not None for proxy in results)
