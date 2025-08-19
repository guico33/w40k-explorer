"""Simple HTTP client for URL collection."""

import asyncio
from typing import Optional, Tuple

from aiohttp import ClientSession, ClientTimeout


class HttpClient:
    """Simple HTTP client with basic rate limiting."""

    def __init__(self, requests_per_second: float = 0.3, timeout_seconds: int = 30):
        """Initialize simple HTTP client.

        Args:
            requests_per_second: Maximum requests per second
            timeout_seconds: Request timeout in seconds
        """
        self.requests_per_second = requests_per_second
        self.timeout_seconds = timeout_seconds
        self._session: Optional[ClientSession] = None
        self._last_request_time = 0.0

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _ensure_session(self) -> None:
        """Ensure HTTP session is created."""
        if not self._session:
            timeout = ClientTimeout(total=self.timeout_seconds)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
            }
            self._session = ClientSession(timeout=timeout, headers=headers)

    async def _rate_limit(self) -> None:
        """Simple rate limiting."""
        if self.requests_per_second <= 0:
            return

        min_interval = 1.0 / self.requests_per_second
        current_time = asyncio.get_event_loop().time()

        # Handle first request case
        if self._last_request_time == 0.0:
            self._last_request_time = current_time
            return

        time_since_last = current_time - self._last_request_time

        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            await asyncio.sleep(wait_time)

        self._last_request_time = asyncio.get_event_loop().time()

    async def get(self, url: str) -> Tuple[str, int]:
        """Perform GET request with rate limiting.

        Returns:
            Tuple of (content, status_code)

        Raises:
            aiohttp.ClientError: On request failure
        """
        await self._rate_limit()
        await self._ensure_session()

        if not self._session:
            raise RuntimeError("Session not initialized")

        async with self._session.get(url) as response:
            response.raise_for_status()
            content = await response.text()
            return content, response.status
