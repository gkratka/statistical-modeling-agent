"""Unit tests for job queue and protocol."""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from src.worker.job_queue import (
    JobQueue,
    Job,
    JobStatus,
    JobType,
    create_job_message,
    validate_progress_message,
    validate_result_message,
)


class TestJobType:
    """Tests for JobType enum."""

    def test_job_types(self):
        """Test all job types are defined."""
        assert JobType.TRAIN == "train"
        assert JobType.PREDICT == "predict"
        assert JobType.LIST_MODELS == "list_models"


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_job_statuses(self):
        """Test all job statuses are defined."""
        assert JobStatus.QUEUED == "queued"
        assert JobStatus.DISPATCHED == "dispatched"
        assert JobStatus.IN_PROGRESS == "in_progress"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.TIMEOUT == "timeout"


class TestJob:
    """Tests for Job dataclass."""

    def test_job_creation(self):
        """Test creating a job."""
        job = Job(
            job_id="job-123",
            user_id=12345,
            job_type=JobType.TRAIN,
            params={"model": "xgboost"},
            created_at=datetime.now(),
        )
        assert job.job_id == "job-123"
        assert job.user_id == 12345
        assert job.job_type == JobType.TRAIN
        assert job.params == {"model": "xgboost"}
        assert job.status == JobStatus.QUEUED
        assert job.progress == 0
        assert job.result is None
        assert job.error is None

    def test_job_default_values(self):
        """Test job default values."""
        job = Job(
            job_id="job-123",
            user_id=12345,
            job_type=JobType.TRAIN,
            params={},
            created_at=datetime.now(),
        )
        assert job.status == JobStatus.QUEUED
        assert job.progress == 0
        assert job.result is None
        assert job.error is None
        assert job.dispatched_at is None
        assert job.completed_at is None

    def test_job_with_custom_status(self):
        """Test job with custom status."""
        job = Job(
            job_id="job-123",
            user_id=12345,
            job_type=JobType.TRAIN,
            params={},
            created_at=datetime.now(),
            status=JobStatus.COMPLETED,
            progress=100,
            result={"accuracy": 0.95},
        )
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 100
        assert job.result == {"accuracy": 0.95}


class TestMessageSchemas:
    """Tests for message schema creation and validation."""

    def test_create_job_message_train(self):
        """Test creating train job message."""
        message = create_job_message(
            job_id="job-123",
            job_type=JobType.TRAIN,
            params={
                "file_path": "/data/train.csv",
                "target": "price",
                "model": "xgboost",
            },
        )

        assert message["type"] == "job"
        assert message["job_id"] == "job-123"
        assert message["action"] == "train"
        assert message["params"]["file_path"] == "/data/train.csv"
        assert message["params"]["target"] == "price"
        assert message["params"]["model"] == "xgboost"

    def test_create_job_message_predict(self):
        """Test creating predict job message."""
        message = create_job_message(
            job_id="job-456",
            job_type=JobType.PREDICT,
            params={
                "file_path": "/data/test.csv",
                "model_id": "model-123",
            },
        )

        assert message["type"] == "job"
        assert message["job_id"] == "job-456"
        assert message["action"] == "predict"
        assert message["params"]["file_path"] == "/data/test.csv"
        assert message["params"]["model_id"] == "model-123"

    def test_create_job_message_list_models(self):
        """Test creating list_models job message."""
        message = create_job_message(
            job_id="job-789",
            job_type=JobType.LIST_MODELS,
            params={},
        )

        assert message["type"] == "job"
        assert message["job_id"] == "job-789"
        assert message["action"] == "list_models"
        assert message["params"] == {}

    def test_validate_progress_message_valid(self):
        """Test validating valid progress message."""
        message = {
            "type": "progress",
            "job_id": "job-123",
            "status": "training",
            "progress": 50,
            "message": "Training in progress...",
        }

        is_valid, error = validate_progress_message(message)
        assert is_valid is True
        assert error is None

    def test_validate_progress_message_missing_field(self):
        """Test validating progress message with missing field."""
        message = {
            "type": "progress",
            "job_id": "job-123",
            "progress": 50,
            # Missing 'message' field
        }

        is_valid, error = validate_progress_message(message)
        assert is_valid is False
        assert "missing required field" in error.lower()

    def test_validate_progress_message_invalid_progress(self):
        """Test validating progress message with invalid progress."""
        message = {
            "type": "progress",
            "job_id": "job-123",
            "status": "training",
            "progress": 150,  # Invalid: > 100
            "message": "Training...",
        }

        is_valid, error = validate_progress_message(message)
        assert is_valid is False
        assert "progress must be 0-100" in error.lower()

    def test_validate_result_message_success(self):
        """Test validating successful result message."""
        message = {
            "type": "result",
            "job_id": "job-123",
            "success": True,
            "data": {"accuracy": 0.95, "model_id": "model-123"},
        }

        is_valid, error = validate_result_message(message)
        assert is_valid is True
        assert error is None

    def test_validate_result_message_failure(self):
        """Test validating failed result message."""
        message = {
            "type": "result",
            "job_id": "job-123",
            "success": False,
            "error": "File not found",
        }

        is_valid, error = validate_result_message(message)
        assert is_valid is True
        assert error is None

    def test_validate_result_message_missing_data_and_error(self):
        """Test validating result message missing both data and error."""
        message = {
            "type": "result",
            "job_id": "job-123",
            "success": True,
            # Missing 'data' (success=True requires data)
        }

        is_valid, error = validate_result_message(message)
        assert is_valid is False
        assert "must include 'data'" in error.lower() or "requires 'data'" in error.lower()


class TestJobQueue:
    """Tests for JobQueue."""

    @pytest.fixture
    def job_queue(self):
        """Create a job queue for testing."""
        return JobQueue(default_timeout=30.0)

    @pytest.fixture
    def mock_worker_manager(self):
        """Create a mock worker manager."""
        manager = MagicMock()
        manager.send_to_worker = AsyncMock(return_value=True)
        manager.is_user_connected = MagicMock(return_value=True)
        manager.set_worker_busy = MagicMock(return_value=True)
        manager.set_worker_idle = MagicMock(return_value=True)
        return manager

    def test_init(self, job_queue):
        """Test job queue initialization."""
        assert job_queue.default_timeout == 30.0
        assert len(job_queue._jobs) == 0
        assert len(job_queue._user_jobs) == 0

    def test_generate_job_id(self, job_queue):
        """Test job ID generation."""
        job_id1 = job_queue._generate_job_id()
        job_id2 = job_queue._generate_job_id()

        assert job_id1.startswith("job_")
        assert job_id2.startswith("job_")
        assert job_id1 != job_id2

    @pytest.mark.asyncio
    async def test_create_job(self, job_queue, mock_worker_manager):
        """Test creating and queuing a job."""
        job_queue.set_worker_manager(mock_worker_manager)

        job_id = await job_queue.create_job(
            user_id=12345,
            job_type=JobType.TRAIN,
            params={"model": "xgboost"},
        )

        assert job_id is not None
        assert job_id.startswith("job_")

        # Check job is stored
        job = job_queue.get_job(job_id)
        assert job is not None
        assert job.user_id == 12345
        assert job.job_type == JobType.TRAIN
        assert job.status == JobStatus.QUEUED

    @pytest.mark.asyncio
    async def test_create_job_auto_dispatch(self, job_queue, mock_worker_manager):
        """Test job auto-dispatch when worker connected."""
        job_queue.set_worker_manager(mock_worker_manager)

        job_id = await job_queue.create_job(
            user_id=12345,
            job_type=JobType.TRAIN,
            params={"model": "xgboost"},
        )

        # Should be dispatched immediately
        await asyncio.sleep(0.1)  # Give async task time to run

        job = job_queue.get_job(job_id)
        assert job.status == JobStatus.DISPATCHED
        assert job.dispatched_at is not None

        # Verify message sent to worker
        mock_worker_manager.send_to_worker.assert_called_once()
        call_args = mock_worker_manager.send_to_worker.call_args
        assert call_args[0][0] == 12345  # user_id
        message = call_args[0][1]
        assert message["type"] == "job"
        assert message["job_id"] == job_id
        assert message["action"] == "train"

        # Cleanup
        await job_queue.cleanup()

    @pytest.mark.asyncio
    async def test_create_job_no_worker(self, job_queue):
        """Test creating job when no worker connected."""
        # No worker manager set
        job_id = await job_queue.create_job(
            user_id=12345,
            job_type=JobType.TRAIN,
            params={"model": "xgboost"},
        )

        # Should remain queued
        job = job_queue.get_job(job_id)
        assert job.status == JobStatus.QUEUED
        assert job.dispatched_at is None

    @pytest.mark.asyncio
    async def test_dispatch_job(self, job_queue, mock_worker_manager):
        """Test dispatching a queued job."""
        job_queue.set_worker_manager(mock_worker_manager)

        # Create job without auto-dispatch
        job_id = job_queue._generate_job_id()
        job = Job(
            job_id=job_id,
            user_id=12345,
            job_type=JobType.TRAIN,
            params={"model": "xgboost"},
            created_at=datetime.now(),
        )
        job_queue._jobs[job_id] = job

        # Dispatch
        success = await job_queue.dispatch_job(job_id)
        assert success is True

        # Check status updated
        assert job.status == JobStatus.DISPATCHED
        assert job.dispatched_at is not None

        # Verify worker manager called
        mock_worker_manager.send_to_worker.assert_called_once()
        mock_worker_manager.set_worker_busy.assert_called_once_with(12345, job_id)

        # Cleanup
        await job_queue.cleanup()

    @pytest.mark.asyncio
    async def test_dispatch_job_not_found(self, job_queue, mock_worker_manager):
        """Test dispatching non-existent job."""
        job_queue.set_worker_manager(mock_worker_manager)

        success = await job_queue.dispatch_job("nonexistent-job")
        assert success is False

    @pytest.mark.asyncio
    async def test_handle_progress(self, job_queue):
        """Test handling progress message."""
        # Create job
        job_id = job_queue._generate_job_id()
        job = Job(
            job_id=job_id,
            user_id=12345,
            job_type=JobType.TRAIN,
            params={},
            created_at=datetime.now(),
            status=JobStatus.DISPATCHED,
        )
        job_queue._jobs[job_id] = job

        # Handle progress
        progress_message = {
            "type": "progress",
            "job_id": job_id,
            "status": "training",
            "progress": 50,
            "message": "Training in progress...",
        }

        await job_queue.handle_progress(12345, progress_message)

        # Check job updated
        assert job.status == JobStatus.IN_PROGRESS
        assert job.progress == 50

    @pytest.mark.asyncio
    async def test_handle_progress_invalid_message(self, job_queue):
        """Test handling invalid progress message."""
        # Create job
        job_id = job_queue._generate_job_id()
        job = Job(
            job_id=job_id,
            user_id=12345,
            job_type=JobType.TRAIN,
            params={},
            created_at=datetime.now(),
        )
        job_queue._jobs[job_id] = job

        # Invalid progress message (missing fields)
        progress_message = {
            "type": "progress",
            "job_id": job_id,
            "progress": 150,  # Invalid
        }

        # Should not crash, just log error
        await job_queue.handle_progress(12345, progress_message)

        # Job should not be updated
        assert job.progress == 0

    @pytest.mark.asyncio
    async def test_handle_result_success(self, job_queue, mock_worker_manager):
        """Test handling successful result message."""
        job_queue.set_worker_manager(mock_worker_manager)

        # Create job
        job_id = job_queue._generate_job_id()
        job = Job(
            job_id=job_id,
            user_id=12345,
            job_type=JobType.TRAIN,
            params={},
            created_at=datetime.now(),
            status=JobStatus.IN_PROGRESS,
        )
        job_queue._jobs[job_id] = job
        job_queue._user_jobs.setdefault(12345, []).append(job_id)

        # Handle result
        result_message = {
            "type": "result",
            "job_id": job_id,
            "success": True,
            "data": {"accuracy": 0.95, "model_id": "model-123"},
        }

        await job_queue.handle_result(12345, result_message)

        # Check job completed
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 100
        assert job.result == {"accuracy": 0.95, "model_id": "model-123"}
        assert job.error is None
        assert job.completed_at is not None

        # Worker should be set to idle
        mock_worker_manager.set_worker_idle.assert_called_once_with(12345)

    @pytest.mark.asyncio
    async def test_handle_result_failure(self, job_queue, mock_worker_manager):
        """Test handling failed result message."""
        job_queue.set_worker_manager(mock_worker_manager)

        # Create job
        job_id = job_queue._generate_job_id()
        job = Job(
            job_id=job_id,
            user_id=12345,
            job_type=JobType.TRAIN,
            params={},
            created_at=datetime.now(),
            status=JobStatus.IN_PROGRESS,
        )
        job_queue._jobs[job_id] = job
        job_queue._user_jobs.setdefault(12345, []).append(job_id)

        # Handle result
        result_message = {
            "type": "result",
            "job_id": job_id,
            "success": False,
            "error": "File not found: /data/train.csv",
        }

        await job_queue.handle_result(12345, result_message)

        # Check job failed
        assert job.status == JobStatus.FAILED
        assert job.error == "File not found: /data/train.csv"
        assert job.result is None
        assert job.completed_at is not None

        # Worker should be set to idle
        mock_worker_manager.set_worker_idle.assert_called_once_with(12345)

    @pytest.mark.asyncio
    async def test_handle_result_job_not_found(self, job_queue):
        """Test handling result for non-existent job."""
        result_message = {
            "type": "result",
            "job_id": "nonexistent-job",
            "success": True,
            "data": {},
        }

        # Should not crash
        await job_queue.handle_result(12345, result_message)

    @pytest.mark.asyncio
    async def test_job_timeout(self, job_queue, mock_worker_manager):
        """Test job timeout handling."""
        job_queue.set_worker_manager(mock_worker_manager)

        # Create job with very short timeout
        job_id = job_queue._generate_job_id()
        job = Job(
            job_id=job_id,
            user_id=12345,
            job_type=JobType.TRAIN,
            params={},
            created_at=datetime.now(),
            status=JobStatus.DISPATCHED,
            dispatched_at=datetime.now(),
        )
        job_queue._jobs[job_id] = job
        job_queue._user_jobs.setdefault(12345, []).append(job_id)

        # Start timeout monitor with very short timeout
        job_queue._start_timeout_monitor(job_id, timeout=0.1)

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Job should be timed out
        assert job.status == JobStatus.TIMEOUT
        assert job.completed_at is not None

        # Worker should be set to idle
        mock_worker_manager.set_worker_idle.assert_called_with(12345)

        # Cleanup
        await job_queue.cleanup()

    @pytest.mark.asyncio
    async def test_timeout_cancelled_on_completion(self, job_queue):
        """Test timeout is cancelled when job completes."""
        # Create job
        job_id = job_queue._generate_job_id()
        job = Job(
            job_id=job_id,
            user_id=12345,
            job_type=JobType.TRAIN,
            params={},
            created_at=datetime.now(),
            status=JobStatus.DISPATCHED,
        )
        job_queue._jobs[job_id] = job

        # Start timeout
        job_queue._start_timeout_monitor(job_id, timeout=1.0)

        # Complete job before timeout
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()

        # Wait a bit
        await asyncio.sleep(0.1)

        # Job should still be completed, not timeout
        assert job.status == JobStatus.COMPLETED

        # Cleanup
        await job_queue.cleanup()

    def test_get_job(self, job_queue):
        """Test getting a job by ID."""
        job_id = job_queue._generate_job_id()
        job = Job(
            job_id=job_id,
            user_id=12345,
            job_type=JobType.TRAIN,
            params={},
            created_at=datetime.now(),
        )
        job_queue._jobs[job_id] = job

        retrieved = job_queue.get_job(job_id)
        assert retrieved is job

        # Non-existent job
        assert job_queue.get_job("nonexistent") is None

    def test_get_user_jobs(self, job_queue):
        """Test getting all jobs for a user."""
        user_id = 12345

        # Create multiple jobs
        job_ids = []
        for i in range(3):
            job_id = job_queue._generate_job_id()
            job = Job(
                job_id=job_id,
                user_id=user_id,
                job_type=JobType.TRAIN,
                params={},
                created_at=datetime.now(),
            )
            job_queue._jobs[job_id] = job
            job_queue._user_jobs.setdefault(user_id, []).append(job_id)
            job_ids.append(job_id)

        # Get user jobs
        jobs = job_queue.get_user_jobs(user_id)
        assert len(jobs) == 3
        assert all(j.user_id == user_id for j in jobs)
        assert set(j.job_id for j in jobs) == set(job_ids)

    def test_get_user_jobs_empty(self, job_queue):
        """Test getting jobs for user with no jobs."""
        jobs = job_queue.get_user_jobs(99999)
        assert jobs == []

    def test_get_active_job_for_user(self, job_queue):
        """Test getting active job for user."""
        user_id = 12345

        # Create in-progress job
        job_id = job_queue._generate_job_id()
        job = Job(
            job_id=job_id,
            user_id=user_id,
            job_type=JobType.TRAIN,
            params={},
            created_at=datetime.now(),
            status=JobStatus.IN_PROGRESS,
        )
        job_queue._jobs[job_id] = job
        job_queue._user_jobs.setdefault(user_id, []).append(job_id)

        active = job_queue.get_active_job_for_user(user_id)
        assert active is job

    def test_get_active_job_none(self, job_queue):
        """Test getting active job when none active."""
        user_id = 12345

        # Create completed job
        job_id = job_queue._generate_job_id()
        job = Job(
            job_id=job_id,
            user_id=user_id,
            job_type=JobType.TRAIN,
            params={},
            created_at=datetime.now(),
            status=JobStatus.COMPLETED,
        )
        job_queue._jobs[job_id] = job
        job_queue._user_jobs.setdefault(user_id, []).append(job_id)

        active = job_queue.get_active_job_for_user(user_id)
        assert active is None

    def test_is_user_busy(self, job_queue):
        """Test checking if user has active job."""
        user_id = 12345

        # No jobs
        assert job_queue.is_user_busy(user_id) is False

        # Add in-progress job
        job_id = job_queue._generate_job_id()
        job = Job(
            job_id=job_id,
            user_id=user_id,
            job_type=JobType.TRAIN,
            params={},
            created_at=datetime.now(),
            status=JobStatus.IN_PROGRESS,
        )
        job_queue._jobs[job_id] = job
        job_queue._user_jobs.setdefault(user_id, []).append(job_id)

        assert job_queue.is_user_busy(user_id) is True

    def test_get_stats(self, job_queue):
        """Test getting job statistics."""
        # Create various jobs
        statuses = [
            JobStatus.QUEUED,
            JobStatus.QUEUED,
            JobStatus.DISPATCHED,
            JobStatus.IN_PROGRESS,
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.TIMEOUT,
        ]

        for i, status in enumerate(statuses):
            job_id = job_queue._generate_job_id()
            job = Job(
                job_id=job_id,
                user_id=12345,
                job_type=JobType.TRAIN,
                params={},
                created_at=datetime.now(),
                status=status,
            )
            job_queue._jobs[job_id] = job

        stats = job_queue.get_stats()
        assert stats["total_jobs"] == 7
        assert stats["queued"] == 2
        assert stats["dispatched"] == 1
        assert stats["in_progress"] == 1
        assert stats["completed"] == 1
        assert stats["failed"] == 1
        assert stats["timeout"] == 1
