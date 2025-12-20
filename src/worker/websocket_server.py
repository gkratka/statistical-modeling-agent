"""WebSocket server for local worker connections.

This module provides the WebSocket server that workers connect to for
receiving ML jobs and sending results back to the bot.

Features:
    - Worker connection management
    - Message routing to/from workers
    - Connection state tracking
    - Graceful disconnection handling
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional, Tuple

import websockets
from websockets.server import WebSocketServerProtocol

from src.worker.worker_manager import WorkerManager
from src.worker.token_manager import TokenManager

logger = logging.getLogger(__name__)


class WebSocketServer:
    """WebSocket server for worker connections.

    Manages WebSocket connections from local workers, handles authentication,
    and routes messages between the bot and workers.

    Attributes:
        host: Host address to bind to
        port: Port number for WebSocket server
        connections: Dict mapping user_id to their worker's WebSocket
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        token_validator: Optional[Callable[[str], Optional[int]]] = None,
        on_disconnect: Optional[Callable[[int], None]] = None,
    ):
        """Initialize WebSocket server.

        Args:
            host: Host address to bind to (default: 0.0.0.0)
            port: Port number for WebSocket server (default: 8765)
            token_validator: Callback to validate tokens, returns user_id or None
            on_disconnect: Callback when worker disconnects, receives user_id
        """
        self.host = host
        self.port = port
        self.connections: Dict[int, WebSocketServerProtocol] = {}
        self._server: Optional[websockets.WebSocketServer] = None
        self._token_validator = token_validator
        self._on_disconnect = on_disconnect
        self._message_handlers: Dict[str, Callable] = {}
        self.worker_manager = WorkerManager()
        self.token_manager = TokenManager()

    async def start(self) -> None:
        """Start the WebSocket server."""
        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            max_size=10 * 1024 * 1024 * 1024,  # 10GB for large model results
        )
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the WebSocket server gracefully."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("WebSocket server stopped")

    def register_connection(
        self, user_id: int, websocket: WebSocketServerProtocol, machine_name: str
    ) -> None:
        """Register a worker connection for a user.

        Args:
            user_id: Telegram user ID
            websocket: WebSocket connection to the worker
            machine_name: Hostname of the worker machine
        """
        # Close existing connection if any
        if user_id in self.connections:
            logger.warning(f"Replacing existing connection for user {user_id}")

        self.connections[user_id] = websocket

        # Register with worker manager for tracking
        self.worker_manager.register_worker(user_id, websocket, machine_name)
        logger.info(f"Worker connected for user {user_id} from {machine_name}")

    def unregister_connection(self, user_id: int) -> None:
        """Unregister a worker connection.

        Args:
            user_id: Telegram user ID
        """
        if user_id in self.connections:
            del self.connections[user_id]
            logger.info(f"Worker disconnected for user {user_id}")

    def is_connected(self, user_id: int) -> bool:
        """Check if a user has a connected worker.

        Args:
            user_id: Telegram user ID

        Returns:
            True if worker is connected, False otherwise
        """
        return user_id in self.connections

    def get_connection(self, user_id: int) -> Optional[WebSocketServerProtocol]:
        """Get the WebSocket connection for a user.

        Args:
            user_id: Telegram user ID

        Returns:
            WebSocket connection or None if not connected
        """
        return self.connections.get(user_id)

    @property
    def connection_count(self) -> int:
        """Get the number of connected workers."""
        return len(self.connections)

    async def send_to_worker(self, user_id: int, message: Dict[str, Any]) -> None:
        """Send a message to a user's worker.

        Args:
            user_id: Telegram user ID
            message: Message dict to send (will be JSON encoded)

        Raises:
            ConnectionError: If no worker is connected for the user
        """
        websocket = self.connections.get(user_id)
        if not websocket:
            raise ConnectionError(f"No worker connected for user {user_id}")

        await websocket.send(json.dumps(message))
        logger.debug(f"Sent message to worker for user {user_id}: {message.get('type')}")

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected workers.

        Args:
            message: Message dict to send to all workers
        """
        if not self.connections:
            return

        json_message = json.dumps(message)
        await asyncio.gather(
            *[ws.send(json_message) for ws in self.connections.values()],
            return_exceptions=True,
        )
        logger.debug(f"Broadcast message to {len(self.connections)} workers")

    def register_message_handler(
        self,
        message_type: str,
        handler: Callable[[int, Dict[str, Any]], Any],
    ) -> None:
        """Register a handler for a specific message type.

        Args:
            message_type: The 'type' field value to handle (e.g., 'progress', 'result')
            handler: Async function(user_id, message) to handle the message
        """
        self._message_handlers[message_type] = handler
        logger.debug(f"Registered handler for message type: {message_type}")

    async def _handle_connection(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
        """
        user_id = None
        try:
            # First message must be auth
            auth_result = await self._handle_auth(websocket)
            if auth_result is None:
                await websocket.close(1008, "Authentication failed")
                return

            user_id, machine_id = auth_result

            # Register the connection
            self.register_connection(user_id, websocket, machine_id)

            # Send connection confirmation
            await websocket.send(json.dumps({
                "type": "connected",
                "message": "Worker connected successfully",
            }))

            # Handle messages until disconnection
            async for message in websocket:
                await self._handle_message(user_id, message)

        except websockets.ConnectionClosed as e:
            logger.info(f"Connection closed for user {user_id}: {e.code} {e.reason}")
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            if user_id:
                self.unregister_connection(user_id)
                if self._on_disconnect:
                    self._on_disconnect(user_id)

    async def _handle_auth(
        self, websocket: WebSocketServerProtocol
    ) -> Optional[Tuple[int, str]]:
        """Handle authentication message from worker.

        Args:
            websocket: The WebSocket connection

        Returns:
            Tuple of (user_id, machine_id) if successful, None otherwise
        """
        try:
            # Wait for auth message with timeout
            raw_message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            message = json.loads(raw_message)

            if message.get("type") != "auth":
                logger.warning("First message was not auth")
                return None

            token = message.get("token")
            if not token:
                logger.warning("Auth message missing token")
                return None

            # Extract machine_id from auth message
            machine_id = message.get("machine_id", "unknown")

            # Validate token and get user_id
            user_id = self._validate_token(token)
            if user_id is None:
                logger.warning(f"Invalid token: {token[:8]}...")
                await websocket.send(json.dumps({
                    "type": "auth_failed",
                    "error": "Invalid or expired token",
                }))
                return None

            logger.info(f"Worker authenticated for user {user_id} from {machine_id}")
            await websocket.send(json.dumps({
                "type": "auth_response",
                "success": True,
                "user_id": user_id,
            }))
            return (user_id, machine_id)

        except asyncio.TimeoutError:
            logger.warning("Auth timeout")
            return None
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in auth message")
            return None

    def _validate_token(self, token: str) -> Optional[int]:
        """Validate authentication token.

        Args:
            token: The token to validate

        Returns:
            User ID if valid, None otherwise
        """
        # DEBUG: Print validation path
        print(f"🌐 WS _validate_token called for: {token[:8]}...", flush=True)
        print(f"🌐 Using custom validator: {self._token_validator is not None}", flush=True)
        print(f"🌐 TokenManager available: {self.token_manager is not None}", flush=True)
        print(f"🌐 WebSocketServer id: {id(self)}", flush=True)
        if self.token_manager:
            print(f"🌐 TokenManager id in WS: {id(self.token_manager)}", flush=True)

        if self._token_validator:
            return self._token_validator(token)
        # Use internal token_manager
        if self.token_manager:
            return self.token_manager.validate_token(token)
        return None

    async def _handle_message(self, user_id: int, raw_message: str) -> None:
        """Handle an incoming message from a worker.

        Args:
            user_id: The user ID of the worker
            raw_message: Raw JSON message string
        """
        try:
            message = json.loads(raw_message)
            message_type = message.get("type")
            job_id = message.get("job_id", "unknown")

            # Debug logging for message reception
            print(f"📨 WS received: type={message_type}, job={job_id}, user={user_id}", flush=True)
            logger.info(f"WS message received: type={message_type}, job_id={job_id}, user={user_id}")

            if not message_type:
                logger.warning(f"Message without type from user {user_id}")
                return

            handler = self._message_handlers.get(message_type)
            if handler:
                print(f"🔄 Calling handler for {message_type} (job={job_id})", flush=True)
                await handler(user_id, message)
                print(f"✅ Handler completed for {message_type} (job={job_id})", flush=True)
            else:
                logger.debug(f"No handler for message type: {message_type}")

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from user {user_id}")
        except Exception as e:
            logger.error(f"Error handling message from user {user_id}: {e}")
            print(f"❌ Handler error: {e}", flush=True)
