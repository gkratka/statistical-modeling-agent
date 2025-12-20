"""Job queue and protocol for local worker system.

This module manages job creation, queuing, dispatching, and result handling
for the local worker architecture.

Features:
    - Job creation with unique IDs
    - Job queuing and dispatching to workers
    - Progress tracking and result handling
    - Timeout monitoring for stuck jobs
    - Message protocol validation
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class JobType(str, Enum):
    """Job types supported by workers."""

    TRAIN = "train"
    PREDICT = "predict"
    LIST_MODELS = "list_models"
    FILE_INFO = "file_info"
    SAVE_FILE = "save_file"
    DELETE_MODEL = "delete_model"
    SET_MODEL_NAME = "set_model_name"


class JobStatus(str, Enum):
    """Job execution status."""

    QUEUED = "queued"  # Created but not dispatched
    DISPATCHED = "dispatched"  # Sent to worker
    IN_PROGRESS = "in_progress"  # Worker reported progress
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Failed with error
    TIMEOUT = "timeout"  # Exceeded timeout


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class Job:
    """Represents a job in the queue.

    Attributes:
        job_id: Unique job identifier
        user_id: Telegram user ID that owns this job
        job_type: Type of job (train, predict, list_models)
        params: Job parameters
        created_at: When job was created
        status: Current job status
        progress: Progress percentage (0-100)
        result: Result data (when completed)
        error: Error message (when failed)
        dispatched_at: When job was dispatched to worker
        completed_at: When job finished (success or failure)
    """

    job_id: str
    user_id: int
    job_type: JobType
    params: Dict[str, Any]
    created_at: datetime
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    dispatched_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ============================================================================
# Message Protocol
# ============================================================================


def create_job_message(
    job_id: str, job_type: JobType, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a job message for sending to worker.

    Message format (bot → worker):
    {
        "type": "job",
        "job_id": "job_xyz",
        "action": "train" | "predict" | "list_models",
        "params": {...}
    }

    Args:
        job_id: Unique job identifier
        job_type: Type of job
        params: Job parameters

    Returns:
        Job message dict
    """
    return {
        "type": "job",
        "job_id": job_id,
        "action": job_type.value,
        "params": params,
    }


def validate_progress_message(message: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate progress message from worker.

    Expected format (worker → bot):
    {
        "type": "progress",
        "job_id": "job_xyz",
        "status": "training",
        "progress": 50,  # 0-100
        "message": "Training in progress..."
    }

    Args:
        message: Progress message to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["type", "job_id", "progress", "message"]

    # Check required fields
    for field in required_fields:
        if field not in message:
            return False, f"Progress message missing required field: {field}"

    # Validate type
    if message["type"] != "progress":
        return False, f"Invalid message type: {message['type']}"

    # Validate progress range
    progress = message.get("progress")
    if not isinstance(progress, (int, float)) or progress < 0 or progress > 100:
        return False, f"Progress must be 0-100, got: {progress}"

    return True, None


def validate_result_message(message: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate result message from worker.

    Expected format (worker → bot):
    Success:
    {
        "type": "result",
        "job_id": "job_xyz",
        "success": true,
        "data": {...}
    }

    Failure:
    {
        "type": "result",
        "job_id": "job_xyz",
        "success": false,
        "error": "Error message"
    }

    Args:
        message: Result message to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["type", "job_id", "success"]

    # Check required fields
    for field in required_fields:
        if field not in message:
            return False, f"Result message missing required field: {field}"

    # Validate type
    if message["type"] != "result":
        return False, f"Invalid message type: {message['type']}"

    # Validate success-specific requirements
    success = message.get("success")
    if success is True:
        if "data" not in message:
            return False, "Successful result must include 'data' field"
    elif success is False:
        if "error" not in message:
            return False, "Failed result must include 'error' field"

    return True, None


# ============================================================================
# Job Queue
# ============================================================================


class JobQueue:
    """Manages job creation, queuing, and lifecycle.

    Handles job creation, dispatching to workers, progress tracking,
    result handling, and timeout monitoring.

    Attributes:
        default_timeout: Default job timeout in seconds
    """

    def __init__(self, default_timeout: float = 3600.0):
        """Initialize job queue.

        Args:
            default_timeout: Default timeout for jobs in seconds (default: 3600 = 1 hour)
        """
        self.default_timeout = default_timeout
        self._jobs: Dict[str, Job] = {}  # job_id -> Job
        self._user_jobs: Dict[int, List[str]] = {}  # user_id -> [job_ids]
        self._timeout_tasks: Dict[str, asyncio.Task] = {}  # job_id -> timeout task
        self._worker_manager = None  # Set via set_worker_manager()

    async def cleanup(self) -> None:
        """Cleanup all pending timeout tasks."""
        for task in self._timeout_tasks.values():
            if not task.done():
                task.cancel()

        # Wait for all tasks to finish
        if self._timeout_tasks:
            await asyncio.gather(*self._timeout_tasks.values(), return_exceptions=True)
        self._timeout_tasks.clear()

    def set_worker_manager(self, worker_manager: Any) -> None:
        """Set the worker manager for dispatching jobs.

        Args:
            worker_manager: WorkerManager instance
        """
        self._worker_manager = worker_manager

    def _generate_job_id(self) -> str:
        """Generate unique job ID.

        Returns:
            Unique job ID string
        """
        return f"job_{uuid.uuid4().hex[:12]}"

    async def create_job(
        self,
        user_id: int,
        job_type: JobType,
        params: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> str:
        """Create and queue a new job.

        If a worker is connected for the user, the job is automatically dispatched.

        Args:
            user_id: Telegram user ID
            job_type: Type of job
            params: Job parameters
            timeout: Job timeout in seconds (uses default if None)

        Returns:
            Job ID
        """
        job_id = self._generate_job_id()
        job = Job(
            job_id=job_id,
            user_id=user_id,
            job_type=job_type,
            params=params,
            created_at=datetime.now(),
        )

        # Store job
        self._jobs[job_id] = job
        self._user_jobs.setdefault(user_id, []).append(job_id)

        logger.info(
            f"Job created: {job_id} (user={user_id}, type={job_type.value})"
        )

        # Auto-dispatch if worker connected
        if self._worker_manager and self._worker_manager.is_user_connected(user_id):
            asyncio.create_task(self.dispatch_job(job_id, timeout))

        return job_id

    async def dispatch_job(
        self, job_id: str, timeout: Optional[float] = None
    ) -> bool:
        """Dispatch a job to its user's worker.

        Args:
            job_id: Job ID to dispatch
            timeout: Job timeout in seconds (uses default if None)

        Returns:
            True if dispatched successfully, False otherwise
        """
        job = self._jobs.get(job_id)
        if job is None:
            logger.error(f"Cannot dispatch job {job_id}: job not found")
            return False

        if not self._worker_manager:
            logger.error(f"Cannot dispatch job {job_id}: no worker manager")
            return False

        # Create job message
        message = create_job_message(job_id, job.job_type, job.params)

        try:
            # Send to worker
            success = await self._worker_manager.send_to_worker(job.user_id, message)
            if not success:
                logger.error(f"Failed to send job {job_id} to worker")
                return False

            # Update job status
            job.status = JobStatus.DISPATCHED
            job.dispatched_at = datetime.now()

            # Mark worker as busy
            self._worker_manager.set_worker_busy(job.user_id, job_id)

            # Start timeout monitor
            timeout_seconds = timeout if timeout is not None else self.default_timeout
            self._start_timeout_monitor(job_id, timeout_seconds)

            logger.info(f"Job dispatched: {job_id} (timeout={timeout_seconds}s)")
            return True

        except Exception as e:
            logger.error(f"Error dispatching job {job_id}: {e}")
            return False

    async def handle_progress(self, user_id: int, message: Dict[str, Any]) -> None:
        """Handle progress message from worker.

        Args:
            user_id: User ID of the worker
            message: Progress message
        """
        # Validate message
        is_valid, error = validate_progress_message(message)
        if not is_valid:
            logger.warning(f"Invalid progress message from user {user_id}: {error}")
            return

        job_id = message["job_id"]
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning(f"Progress for unknown job: {job_id}")
            return

        # Update job
        job.status = JobStatus.IN_PROGRESS
        job.progress = int(message["progress"])

        logger.debug(
            f"Job progress: {job_id} ({job.progress}% - {message.get('message', '')})"
        )

    async def handle_result(self, user_id: int, message: Dict[str, Any]) -> None:
        """Handle result message from worker.

        Args:
            user_id: User ID of the worker
            message: Result message
        """
        job_id = message.get("job_id", "unknown")
        print(f"📥 JobQueue.handle_result called: job={job_id}, user={user_id}", flush=True)
        logger.info(f"JobQueue.handle_result: job={job_id}, user={user_id}")

        # Validate message
        is_valid, error = validate_result_message(message)
        if not is_valid:
            print(f"❌ Invalid result message: {error}", flush=True)
            logger.warning(f"Invalid result message from user {user_id}: {error}")
            return

        job_id = message["job_id"]
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning(f"Result for unknown job: {job_id}")
            return

        # Cancel timeout
        self._cancel_timeout(job_id)

        # Update job
        job.completed_at = datetime.now()
        success = message["success"]

        if success:
            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.result = message.get("data", {})
            print(f"✅ Job {job_id} marked COMPLETED (result keys: {list(job.result.keys()) if job.result else 'none'})", flush=True)
            logger.info(f"Job completed: {job_id}")
        else:
            job.status = JobStatus.FAILED
            job.error = message.get("error", "Unknown error")
            print(f"❌ Job {job_id} marked FAILED: {job.error}", flush=True)
            logger.warning(f"Job failed: {job_id} - {job.error}")

        # Mark worker as idle
        if self._worker_manager:
            self._worker_manager.set_worker_idle(user_id)

    def _start_timeout_monitor(self, job_id: str, timeout: float) -> None:
        """Start timeout monitoring for a job.

        Args:
            job_id: Job ID to monitor
            timeout: Timeout in seconds
        """
        task = asyncio.create_task(self._timeout_monitor(job_id, timeout))
        self._timeout_tasks[job_id] = task

    async def _timeout_monitor(self, job_id: str, timeout: float) -> None:
        """Monitor job timeout.

        Args:
            job_id: Job ID to monitor
            timeout: Timeout in seconds
        """
        try:
            await asyncio.sleep(timeout)

            # Timeout reached
            job = self._jobs.get(job_id)
            if job is None:
                return

            # Only timeout if job is not completed
            if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED]:
                job.status = JobStatus.TIMEOUT
                job.completed_at = datetime.now()
                logger.warning(f"Job timeout: {job_id} (timeout={timeout}s)")

                # Mark worker as idle
                if self._worker_manager:
                    self._worker_manager.set_worker_idle(job.user_id)

        except asyncio.CancelledError:
            # Timeout was cancelled (job completed)
            logger.debug(f"Timeout cancelled for job: {job_id}")

    def _cancel_timeout(self, job_id: str) -> None:
        """Cancel timeout monitoring for a job.

        Args:
            job_id: Job ID
        """
        task = self._timeout_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            del self._timeout_tasks[job_id]

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job if found, None otherwise
        """
        return self._jobs.get(job_id)

    def get_user_jobs(self, user_id: int) -> List[Job]:
        """Get all jobs for a user.

        Args:
            user_id: User ID

        Returns:
            List of jobs for the user
        """
        job_ids = self._user_jobs.get(user_id, [])
        return [self._jobs[jid] for jid in job_ids if jid in self._jobs]

    def get_active_job_for_user(self, user_id: int) -> Optional[Job]:
        """Get active job for a user (if any).

        Args:
            user_id: User ID

        Returns:
            Active job if exists, None otherwise
        """
        jobs = self.get_user_jobs(user_id)
        active_statuses = [
            JobStatus.QUEUED,
            JobStatus.DISPATCHED,
            JobStatus.IN_PROGRESS,
        ]
        for job in jobs:
            if job.status in active_statuses:
                return job
        return None

    def is_user_busy(self, user_id: int) -> bool:
        """Check if user has an active job.

        Args:
            user_id: User ID

        Returns:
            True if user has active job, False otherwise
        """
        return self.get_active_job_for_user(user_id) is not None

    def get_stats(self) -> Dict[str, int]:
        """Get job queue statistics.

        Returns:
            Dictionary with job counts by status
        """
        stats = {
            "total_jobs": len(self._jobs),
            "queued": 0,
            "dispatched": 0,
            "in_progress": 0,
            "completed": 0,
            "failed": 0,
            "timeout": 0,
        }

        for job in self._jobs.values():
            if job.status == JobStatus.QUEUED:
                stats["queued"] += 1
            elif job.status == JobStatus.DISPATCHED:
                stats["dispatched"] += 1
            elif job.status == JobStatus.IN_PROGRESS:
                stats["in_progress"] += 1
            elif job.status == JobStatus.COMPLETED:
                stats["completed"] += 1
            elif job.status == JobStatus.FAILED:
                stats["failed"] += 1
            elif job.status == JobStatus.TIMEOUT:
                stats["timeout"] += 1

        return stats
