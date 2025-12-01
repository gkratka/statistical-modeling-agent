"""Local Worker module for hybrid ML execution.

This module provides WebSocket-based communication between the Railway-hosted
bot and locally-running ML workers on user machines.

Components:
    - websocket_server: WebSocket server for worker connections
    - token_manager: One-time token generation and validation
    - worker_manager: Connected worker tracking per user
    - job_queue: Job queuing and dispatching to workers
"""

# Lazy imports to avoid circular dependencies and missing module errors
__all__ = [
    "WebSocketServer",
    "HTTPServer",
    "TokenManager",
    "WorkerManager",
    "JobQueue",
]


def __getattr__(name: str):
    """Lazy import of submodules."""
    if name == "WebSocketServer":
        from src.worker.websocket_server import WebSocketServer
        return WebSocketServer
    elif name == "HTTPServer":
        from src.worker.http_server import HTTPServer
        return HTTPServer
    elif name == "TokenManager":
        from src.worker.token_manager import TokenManager
        return TokenManager
    elif name == "WorkerManager":
        from src.worker.worker_manager import WorkerManager
        return WorkerManager
    elif name == "JobQueue":
        from src.worker.job_queue import JobQueue
        return JobQueue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
