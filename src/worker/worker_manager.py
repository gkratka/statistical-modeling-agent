"""Worker manager for tracking connected workers.

Handles registration, tracking, and communication with connected local workers.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConnectedWorker:
    """Represents a connected local worker.

    Attributes:
        user_id: Telegram user ID that owns this worker
        websocket: The WebSocket connection to the worker
        machine_name: Hostname of the worker machine
        connected_at: When the worker connected
        is_busy: Whether the worker is currently processing a job
        current_job_id: ID of the current job being processed
    """

    user_id: int
    websocket: Any  # WebSocket connection object
    machine_name: str
    connected_at: datetime
    is_busy: bool = False
    current_job_id: Optional[str] = None


class WorkerManager:
    """Manages connected local workers.

    Tracks worker connections, handles registration/unregistration,
    and provides worker lookup by user or websocket.

    Attributes:
        max_workers_per_user: Maximum concurrent workers per user (MVP: 1)
    """

    def __init__(self, max_workers_per_user: int = 1):
        """Initialize worker manager.

        Args:
            max_workers_per_user: Max workers per user (default 1)
        """
        self.max_workers_per_user = max_workers_per_user
        self._workers: Dict[Any, ConnectedWorker] = {}  # websocket -> worker
        self._user_workers: Dict[int, Any] = {}  # user_id -> websocket

        # Callbacks
        self._on_connect_callback: Optional[Callable[[int, str], None]] = None
        self._on_disconnect_callback: Optional[Callable[[int], None]] = None

    def on_connect(self, callback: Callable[[int, str], Any]) -> None:
        """Set callback for worker connection events.

        Args:
            callback: Function(user_id, machine_name) called on connect
                     Can be sync or async
        """
        self._on_connect_callback = callback

    def on_disconnect(self, callback: Callable[[int], Any]) -> None:
        """Set callback for worker disconnection events.

        Args:
            callback: Function(user_id) called on disconnect
                     Can be sync or async
        """
        self._on_disconnect_callback = callback

    def register_worker(
        self, user_id: int, websocket: Any, machine_name: str
    ) -> bool:
        """Register a new worker connection.

        If user already has a worker, the old one is replaced.

        Args:
            user_id: Telegram user ID
            websocket: WebSocket connection
            machine_name: Hostname of worker machine

        Returns:
            True if registration successful
        """
        # Remove existing worker for this user (MVP: 1 worker per user)
        if user_id in self._user_workers:
            old_ws = self._user_workers[user_id]
            if old_ws in self._workers:
                del self._workers[old_ws]
            logger.debug(f"Replaced existing worker for user {user_id}")

        # Create new worker connection
        worker = ConnectedWorker(
            user_id=user_id,
            websocket=websocket,
            machine_name=machine_name,
            connected_at=datetime.now(),
        )

        # Store in both mappings
        self._workers[websocket] = worker
        self._user_workers[user_id] = websocket

        logger.info(f"Worker registered: user={user_id}, machine={machine_name}")

        # Trigger callback (handle both sync and async)
        if self._on_connect_callback:
            import asyncio
            import inspect
            if inspect.iscoroutinefunction(self._on_connect_callback):
                # Schedule async callback
                asyncio.create_task(self._on_connect_callback(user_id, machine_name))
            else:
                # Call sync callback
                self._on_connect_callback(user_id, machine_name)

        return True

    def unregister_worker(self, websocket: Any) -> bool:
        """Unregister a worker connection.

        Args:
            websocket: WebSocket connection to unregister

        Returns:
            True if worker was unregistered, False if not found
        """
        worker = self._workers.get(websocket)
        if worker is None:
            return False

        user_id = worker.user_id

        # Remove from both mappings
        del self._workers[websocket]
        if user_id in self._user_workers:
            if self._user_workers[user_id] == websocket:
                del self._user_workers[user_id]

        logger.info(f"Worker unregistered: user={user_id}")

        # Trigger callback (handle both sync and async)
        if self._on_disconnect_callback:
            import asyncio
            import inspect
            if inspect.iscoroutinefunction(self._on_disconnect_callback):
                # Schedule async callback
                asyncio.create_task(self._on_disconnect_callback(user_id))
            else:
                # Call sync callback
                self._on_disconnect_callback(user_id)

        return True

    def get_worker_for_user(self, user_id: int) -> Optional[ConnectedWorker]:
        """Get the connected worker for a user.

        Args:
            user_id: Telegram user ID

        Returns:
            ConnectedWorker if user has one, None otherwise
        """
        websocket = self._user_workers.get(user_id)
        if websocket is None:
            return None
        return self._workers.get(websocket)

    def get_user_for_worker(self, websocket: Any) -> Optional[int]:
        """Get the user_id for a websocket connection.

        Args:
            websocket: WebSocket connection

        Returns:
            user_id if worker exists, None otherwise
        """
        worker = self._workers.get(websocket)
        return worker.user_id if worker else None

    def is_user_connected(self, user_id: int) -> bool:
        """Check if user has a connected worker.

        Args:
            user_id: Telegram user ID

        Returns:
            True if user has connected worker
        """
        return user_id in self._user_workers

    def get_machine_name(self, user_id: int) -> Optional[str]:
        """Get the machine name for a user's worker.

        Args:
            user_id: Telegram user ID

        Returns:
            Machine name if worker connected, None otherwise
        """
        worker = self.get_worker_for_user(user_id)
        return worker.machine_name if worker else None

    def set_worker_busy(self, user_id: int, job_id: str) -> bool:
        """Mark a worker as busy processing a job.

        Args:
            user_id: Telegram user ID
            job_id: ID of the job being processed

        Returns:
            True if status updated, False if worker not found
        """
        worker = self.get_worker_for_user(user_id)
        if worker is None:
            return False

        worker.is_busy = True
        worker.current_job_id = job_id
        logger.debug(f"Worker busy: user={user_id}, job={job_id}")
        return True

    def set_worker_idle(self, user_id: int) -> bool:
        """Mark a worker as idle (finished processing).

        Args:
            user_id: Telegram user ID

        Returns:
            True if status updated, False if worker not found
        """
        worker = self.get_worker_for_user(user_id)
        if worker is None:
            return False

        worker.is_busy = False
        worker.current_job_id = None
        logger.debug(f"Worker idle: user={user_id}")
        return True

    def is_worker_available(self, user_id: int) -> bool:
        """Check if user's worker is available for new jobs.

        Args:
            user_id: Telegram user ID

        Returns:
            True if worker exists and is not busy
        """
        worker = self.get_worker_for_user(user_id)
        if worker is None:
            return False
        return not worker.is_busy

    def get_all_workers(self) -> List[ConnectedWorker]:
        """Get all connected workers.

        Returns:
            List of all connected workers
        """
        return list(self._workers.values())

    def get_stats(self) -> Dict[str, int]:
        """Get worker statistics.

        Returns:
            Dictionary with worker counts
        """
        workers = list(self._workers.values())
        total = len(workers)
        busy = sum(1 for w in workers if w.is_busy)
        idle = total - busy

        return {
            "total_workers": total,
            "busy_workers": busy,
            "idle_workers": idle,
        }

    async def disconnect_user(self, user_id: int) -> bool:
        """Disconnect a user's worker.

        Args:
            user_id: Telegram user ID

        Returns:
            True if disconnected, False if no worker
        """
        websocket = self._user_workers.get(user_id)
        if websocket is None:
            return False

        # Close the websocket
        try:
            await websocket.close()
        except Exception as e:
            logger.warning(f"Error closing websocket for user {user_id}: {e}")

        # Unregister
        self.unregister_worker(websocket)
        return True

    async def send_to_worker(self, user_id: int, message: Dict[str, Any]) -> bool:
        """Send a message to a user's worker.

        Args:
            user_id: Telegram user ID
            message: Message dict to send (will be JSON encoded)

        Returns:
            True if sent, False if no worker or error
        """
        worker = self.get_worker_for_user(user_id)
        if worker is None:
            return False

        try:
            await worker.websocket.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Error sending to worker for user {user_id}: {e}")
            return False
