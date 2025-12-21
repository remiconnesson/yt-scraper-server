"""Proxy manager for handling multiple datacenter proxy IPs with burnt IP tracking."""

import logging
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ProxyManager:
    """Manages multiple proxy IPs with burnt IP tracking and rotation.

    This class handles:
    - Parsing multiple proxy IPs from environment configuration
    - Tracking burnt IPs with timestamps to temporarily exclude them
    - Providing the next available proxy IP for downloads
    - Thread-safe access to proxy state
    """

    # Time in seconds to exclude a burnt IP before allowing retry (1 hour)
    BURNT_IP_COOLDOWN_SECONDS = 3600

    def __init__(self, proxy_config: Optional[str] = None):
        """Initialize the proxy manager with optional proxy configuration.

        Args:
            proxy_config: Comma-separated proxy IPs or URLs
                         Format: "user:pass@host:port,user:pass@host2:port2"
                         or "http://proxy1,http://proxy2"
        """
        self._lock = threading.Lock()
        self._proxies: List[str] = []
        self._burnt_ips: Dict[str, float] = {}  # proxy -> timestamp when burnt
        self._current_index = 0

        if proxy_config and len(proxy_config.strip()) > 0:
            self._parse_proxy_config(proxy_config)

    def _parse_proxy_config(self, config: str) -> None:
        """Parse comma-separated proxy configuration."""
        raw_proxies = [p.strip() for p in config.split(",") if p.strip()]

        for proxy in raw_proxies:
            # Normalize proxy format
            if not proxy.startswith("http://") and not proxy.startswith("https://"):
                proxy = f"http://{proxy}"
            self._proxies.append(proxy)

        logger.info(f"Initialized ProxyManager with {len(self._proxies)} proxy IP(s)")

    def get_next_proxy(self) -> Optional[Dict[str, str]]:
        """Get the next available proxy that hasn't been recently burnt.

        Returns:
            A dictionary with 'http' and 'https' keys pointing to the proxy URL,
            or None if no proxies are configured or all are burnt.
        """
        with self._lock:
            if not self._proxies:
                return None

            # Clean up expired burnt IPs
            current_time = time.time()
            expired_ips = [
                proxy
                for proxy, burnt_time in self._burnt_ips.items()
                if current_time - burnt_time > self.BURNT_IP_COOLDOWN_SECONDS
            ]
            for proxy in expired_ips:
                logger.info(f"Proxy cooldown expired, re-enabling: {self._mask_proxy(proxy)}")
                del self._burnt_ips[proxy]

            # Find next available (non-burnt) proxy
            attempts = 0
            while attempts < len(self._proxies):
                proxy = self._proxies[self._current_index]
                self._current_index = (self._current_index + 1) % len(self._proxies)
                attempts += 1

                if proxy not in self._burnt_ips:
                    logger.debug(f"Selected proxy: {self._mask_proxy(proxy)}")
                    return {"http": proxy, "https": proxy}

            # All proxies are burnt
            logger.warning("All proxy IPs are currently burnt. Waiting for cooldown.")
            return None

    def mark_burnt(self, proxy_dict: Dict[str, str]) -> None:
        """Mark a proxy as burnt so it won't be used temporarily.

        Args:
            proxy_dict: The proxy dictionary returned by get_next_proxy()
        """
        if not proxy_dict or "http" not in proxy_dict:
            return

        proxy = proxy_dict["http"]
        with self._lock:
            if proxy in self._proxies:
                self._burnt_ips[proxy] = time.time()
                logger.warning(
                    f"Proxy marked as burnt: {self._mask_proxy(proxy)} "
                    f"(will be excluded for {self.BURNT_IP_COOLDOWN_SECONDS}s)"
                )

    def has_available_proxies(self) -> bool:
        """Check if there are any available (non-burnt) proxies.

        Returns:
            True if at least one proxy is available, False otherwise.
        """
        with self._lock:
            if not self._proxies:
                return False

            current_time = time.time()
            for proxy in self._proxies:
                if proxy not in self._burnt_ips:
                    return True
                # Check if burnt time has expired
                burnt_time = self._burnt_ips[proxy]
                if current_time - burnt_time > self.BURNT_IP_COOLDOWN_SECONDS:
                    return True

            return False

    def get_proxy_count(self) -> int:
        """Get the total number of configured proxies."""
        with self._lock:
            return len(self._proxies)

    def get_burnt_count(self) -> int:
        """Get the number of currently burnt proxies."""
        with self._lock:
            current_time = time.time()
            return sum(
                1
                for proxy, burnt_time in self._burnt_ips.items()
                if current_time - burnt_time <= self.BURNT_IP_COOLDOWN_SECONDS
            )

    @staticmethod
    def _mask_proxy(proxy: str) -> str:
        """Mask sensitive parts of proxy URL for logging.

        Args:
            proxy: The proxy URL to mask

        Returns:
            Masked proxy URL with credentials hidden
        """
        # Simple masking: hide credentials if present
        if "@" in proxy:
            parts = proxy.split("@")
            if len(parts) == 2:
                protocol_user = parts[0].split("://")
                if len(protocol_user) == 2:
                    return f"{protocol_user[0]}://***:***@{parts[1]}"
        return proxy
