"""HTTP server for serving worker script.

This module provides an HTTP endpoint that serves the worker script
to users when they run the curl command.
"""

import logging
from pathlib import Path
from typing import Optional

from aiohttp import web

logger = logging.getLogger(__name__)


class HTTPServer:
    """HTTP server for serving worker script and health checks.

    Provides:
        - GET /worker - Returns the worker Python script
        - GET /health - Health check endpoint

    Attributes:
        host: Host address to bind to
        port: Port number for HTTP server
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        worker_script_path: Optional[Path] = None,
    ):
        """Initialize HTTP server.

        Args:
            host: Host address to bind to (default: 0.0.0.0)
            port: Port number for HTTP server (default: 8080)
            worker_script_path: Path to worker script file
        """
        self.host = host
        self.port = port
        self.worker_script_path = worker_script_path or Path("worker/statsbot_worker.py")
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    async def start(self) -> None:
        """Start the HTTP server."""
        self._app = web.Application()
        self._app.router.add_get("/worker", self._handle_worker_script)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        logger.info(f"HTTP server started on http://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the HTTP server gracefully."""
        if self._runner:
            await self._runner.cleanup()
            logger.info("HTTP server stopped")

    async def _handle_worker_script(self, request: web.Request) -> web.Response:
        """Handle GET /worker - serve the worker script.

        Args:
            request: The HTTP request

        Returns:
            Response with worker script content
        """
        try:
            if not self.worker_script_path.exists():
                logger.error(f"Worker script not found: {self.worker_script_path}")
                return web.Response(
                    text="# Error: Worker script not found\nprint('Worker script not available')",
                    content_type="text/x-python",
                    status=404,
                )

            script_content = self.worker_script_path.read_text()
            return web.Response(
                text=script_content,
                content_type="text/x-python",
            )

        except Exception as e:
            logger.error(f"Error serving worker script: {e}")
            return web.Response(
                text=f"# Error: {e}\nprint('Failed to load worker script')",
                content_type="text/x-python",
                status=500,
            )

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle GET /health - health check endpoint.

        Args:
            request: The HTTP request

        Returns:
            JSON response with health status
        """
        return web.json_response({
            "status": "healthy",
            "service": "statsbot-worker-server",
        })
