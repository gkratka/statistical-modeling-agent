"""
Tests for RunPodPodManager.

This module tests the RunPod pod manager including GPU selection,
pod launch, monitoring, log streaming, and termination.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 6.6: RunPod Pod Manager Tests)
"""

import sys
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio

# Mock runpod module before importing RunPodPodManager
sys.modules['runpod'] = MagicMock()

from src.cloud.runpod_pod_manager import RunPodPodManager
from src.cloud.runpod_config import RunPodConfig
import runpod


@pytest.fixture
def runpod_config():
    """Create valid RunPod configuration for testing."""
    return RunPodConfig(
        runpod_api_key="test-api-key",
        storage_endpoint="https://storage.runpod.io",
        network_volume_id="vol-test123",
        default_gpu_type="NVIDIA RTX A5000",
        cloud_type="COMMUNITY",
        storage_access_key="test-access-key",
        storage_secret_key="test-secret-key",
        data_prefix="datasets",
        models_prefix="models",
        results_prefix="results"
    )


@pytest.fixture
def pod_manager(runpod_config):
    """Create RunPodPodManager instance."""
    with patch('runpod.api_key'):
        manager = RunPodPodManager(runpod_config)
        return manager


class TestRunPodPodManagerInitialization:
    """Test RunPodPodManager initialization."""

    @patch('runpod.api_key')
    def test_init_sets_api_key(self, mock_api_key, runpod_config):
        """Initialization should set runpod.api_key."""
        manager = RunPodPodManager(runpod_config)

        assert manager.config == runpod_config
        assert runpod.api_key == runpod_config.runpod_api_key


class TestGPUSelection:
    """Test GPU type selection logic."""

    def test_select_compute_type_small_dataset(self, pod_manager):
        """Small dataset (<1GB) should select RTX A5000."""
        gpu_type = pod_manager.select_compute_type(
            dataset_size_mb=500,
            model_type='linear'
        )
        assert gpu_type == 'NVIDIA RTX A5000'

    def test_select_compute_type_medium_dataset(self, pod_manager):
        """Medium dataset (1-5GB) should select RTX A40."""
        gpu_type = pod_manager.select_compute_type(
            dataset_size_mb=2048,  # 2GB
            model_type='random_forest'
        )
        assert gpu_type == 'NVIDIA RTX A40'

    def test_select_compute_type_large_dataset(self, pod_manager):
        """Large dataset (>5GB) should select A100 40GB."""
        gpu_type = pod_manager.select_compute_type(
            dataset_size_mb=6144,  # 6GB
            model_type='random_forest'
        )
        assert gpu_type == 'NVIDIA A100 PCIe 40GB'

    def test_select_compute_type_neural_network(self, pod_manager):
        """Neural networks should always select A100."""
        gpu_type = pod_manager.select_compute_type(
            dataset_size_mb=500,
            model_type='mlp_regression'
        )
        assert gpu_type == 'NVIDIA A100 PCIe 40GB'

        gpu_type = pod_manager.select_compute_type(
            dataset_size_mb=500,
            model_type='mlp_classification'
        )
        assert gpu_type == 'NVIDIA A100 PCIe 40GB'

        gpu_type = pod_manager.select_compute_type(
            dataset_size_mb=500,
            model_type='neural_network'
        )
        assert gpu_type == 'NVIDIA A100 PCIe 40GB'


class TestPodLaunch:
    """Test pod launch functionality."""

    @patch('runpod.create_pod')
    def test_launch_training_creates_pod(self, mock_create_pod, pod_manager):
        """launch_training should create pod with correct configuration."""
        mock_create_pod.return_value = {'id': 'pod-xyz123'}

        config = {
            'gpu_type': 'NVIDIA RTX A5000',
            'dataset_key': 'datasets/user_123/data.csv',
            'model_id': 'model_456',
            'user_id': 123,
            'model_type': 'random_forest',
            'target_column': 'price',
            'feature_columns': ['sqft', 'bedrooms', 'bathrooms'],
            'hyperparameters': {'n_estimators': 100}
        }

        result = pod_manager.launch_training(config)

        # Verify pod creation called
        mock_create_pod.assert_called_once()
        call_kwargs = mock_create_pod.call_args[1]

        assert call_kwargs['name'] == 'training_123_model_456'
        assert call_kwargs['gpu_type_id'] == 'NVIDIA RTX A5000'
        assert call_kwargs['cloud_type'] == 'COMMUNITY'
        assert 'DATASET_KEY' in call_kwargs['env']
        assert call_kwargs['env']['MODEL_ID'] == 'model_456'

        # Verify result
        assert result['pod_id'] == 'pod-xyz123'
        assert result['gpu_type'] == 'NVIDIA RTX A5000'
        assert result['status'] == 'launching'
        assert 'launch_time' in result

    def test_launch_training_missing_required_keys(self, pod_manager):
        """launch_training should raise ValueError for missing required keys."""
        incomplete_config = {
            'gpu_type': 'NVIDIA RTX A5000',
            # Missing dataset_key, model_id, etc.
        }

        with pytest.raises(ValueError) as exc_info:
            pod_manager.launch_training(incomplete_config)

        assert "Missing required config key" in str(exc_info.value)

    @patch('runpod.create_pod')
    def test_launch_training_with_custom_docker_image(self, mock_create_pod, pod_manager):
        """launch_training should use custom Docker image if provided."""
        mock_create_pod.return_value = {'id': 'pod-xyz123'}

        config = {
            'gpu_type': 'NVIDIA RTX A5000',
            'dataset_key': 'datasets/user_123/data.csv',
            'model_id': 'model_456',
            'user_id': 123,
            'model_type': 'random_forest',
            'target_column': 'price',
            'feature_columns': ['sqft'],
            'docker_image': 'myuser/custom-ml:latest',
            'volume_size_gb': 100
        }

        result = pod_manager.launch_training(config)

        call_kwargs = mock_create_pod.call_args[1]
        assert call_kwargs['image_name'] == 'myuser/custom-ml:latest'
        assert call_kwargs['volume_in_gb'] == 100

    @patch('runpod.create_pod')
    def test_launch_training_pod_creation_failure(self, mock_create_pod, pod_manager):
        """launch_training should raise RuntimeError on pod creation failure."""
        mock_create_pod.side_effect = Exception("RunPod API error")

        config = {
            'gpu_type': 'NVIDIA RTX A5000',
            'dataset_key': 'datasets/user_123/data.csv',
            'model_id': 'model_456',
            'user_id': 123,
            'model_type': 'random_forest',
            'target_column': 'price',
            'feature_columns': ['sqft']
        }

        with pytest.raises(RuntimeError) as exc_info:
            pod_manager.launch_training(config)

        assert "Failed to create RunPod pod" in str(exc_info.value)


class TestPodMonitoring:
    """Test pod monitoring functionality."""

    @patch('runpod.get_pod')
    def test_monitor_training_returns_status(self, mock_get_pod, pod_manager):
        """monitor_training should return pod status."""
        mock_get_pod.return_value = {
            'id': 'pod-xyz123',
            'desiredStatus': 'RUNNING',
            'runtime': 120,
            'gpuUtilization': 85,
            'machineId': 'machine-abc'
        }

        status = pod_manager.monitor_training('pod-xyz123')

        assert status['pod_id'] == 'pod-xyz123'
        assert status['status'] == 'RUNNING'
        assert status['runtime_seconds'] == 120
        assert status['gpu_utilization'] == 85
        assert status['machine_id'] == 'machine-abc'

    @patch('runpod.get_pod')
    def test_monitor_training_pod_not_found(self, mock_get_pod, pod_manager):
        """monitor_training should handle pod not found gracefully."""
        mock_get_pod.side_effect = Exception("Pod not found")

        status = pod_manager.monitor_training('pod-nonexistent')

        assert status['pod_id'] == 'pod-nonexistent'
        assert status['status'] == 'UNKNOWN'
        assert 'error' in status


class TestPodLogs:
    """Test pod log retrieval."""

    @patch('runpod.get_pod_logs')
    def test_get_pod_logs(self, mock_get_logs, pod_manager):
        """get_pod_logs should return log lines."""
        mock_get_logs.return_value = "Line 1\nLine 2\nLine 3"

        logs = pod_manager.get_pod_logs('pod-xyz123', lines=100)

        assert len(logs) == 3
        assert logs[0] == "Line 1"
        assert logs[2] == "Line 3"
        mock_get_logs.assert_called_once_with('pod-xyz123', tail=100)

    @patch('runpod.get_pod_logs')
    def test_get_pod_logs_empty(self, mock_get_logs, pod_manager):
        """get_pod_logs should handle empty logs."""
        mock_get_logs.return_value = None

        logs = pod_manager.get_pod_logs('pod-xyz123')

        assert logs == []

    @patch('runpod.get_pod_logs')
    def test_get_pod_logs_error(self, mock_get_logs, pod_manager):
        """get_pod_logs should handle errors gracefully."""
        mock_get_logs.side_effect = Exception("API error")

        logs = pod_manager.get_pod_logs('pod-xyz123')

        assert len(logs) == 1
        assert "Error retrieving logs" in logs[0]


class TestPodTermination:
    """Test pod termination."""

    @patch('runpod.terminate_pod')
    def test_terminate_pod(self, mock_terminate, pod_manager):
        """terminate_pod should terminate the pod."""
        terminated_id = pod_manager.terminate_pod('pod-xyz123')

        assert terminated_id == 'pod-xyz123'
        mock_terminate.assert_called_once_with('pod-xyz123')

    @patch('runpod.terminate_pod')
    def test_terminate_pod_failure(self, mock_terminate, pod_manager):
        """terminate_pod should raise RuntimeError on failure."""
        mock_terminate.side_effect = Exception("Termination failed")

        with pytest.raises(RuntimeError) as exc_info:
            pod_manager.terminate_pod('pod-xyz123')

        assert "Failed to terminate pod" in str(exc_info.value)


class TestTrainingTimeEstimation:
    """Test training time estimation."""

    def test_estimate_training_time_linear_models(self, pod_manager):
        """Linear models should use 0.1 sec/MB heuristic."""
        time_sec = pod_manager.estimate_training_time(100, 'linear')
        assert time_sec == 10  # 100 * 0.1

        time_sec = pod_manager.estimate_training_time(100, 'ridge')
        assert time_sec == 10

    def test_estimate_training_time_tree_models(self, pod_manager):
        """Tree models should use 0.5 sec/MB heuristic."""
        time_sec = pod_manager.estimate_training_time(100, 'random_forest')
        assert time_sec == 50  # 100 * 0.5

        time_sec = pod_manager.estimate_training_time(100, 'gradient_boosting')
        assert time_sec == 50

        time_sec = pod_manager.estimate_training_time(100, 'xgboost')
        assert time_sec == 50

    def test_estimate_training_time_neural_networks(self, pod_manager):
        """Neural networks should use 2 sec/MB heuristic."""
        time_sec = pod_manager.estimate_training_time(100, 'mlp_regression')
        assert time_sec == 200  # 100 * 2.0

        time_sec = pod_manager.estimate_training_time(100, 'mlp_classification')
        assert time_sec == 200


class TestAsyncLogStreaming:
    """Test async log streaming."""

    @pytest.mark.asyncio
    @patch('runpod.get_pod')
    @patch('runpod.get_pod_logs')
    async def test_poll_training_logs_yields_new_lines(
        self, mock_get_logs, mock_get_pod, pod_manager
    ):
        """poll_training_logs should yield new log lines as they appear."""
        # Simulate log progression
        mock_get_pod.side_effect = [
            {'desiredStatus': 'RUNNING'},
            {'desiredStatus': 'RUNNING'},
            {'desiredStatus': 'EXITED'}
        ]

        mock_get_logs.side_effect = [
            "Line 1\nLine 2",
            "Line 1\nLine 2\nLine 3\nLine 4",
            "Line 1\nLine 2\nLine 3\nLine 4"
        ]

        logs = []
        async for line in pod_manager.poll_training_logs('pod-xyz123', poll_interval_seconds=0.01):
            logs.append(line)

        # Should yield only new lines
        assert len(logs) == 4
        assert logs[0] == "Line 1"
        assert logs[3] == "Line 4"

    @pytest.mark.asyncio
    @patch('runpod.get_pod')
    @patch('runpod.get_pod_logs')
    async def test_poll_training_logs_stops_on_completion(
        self, mock_get_logs, mock_get_pod, pod_manager
    ):
        """poll_training_logs should stop when pod completes."""
        mock_get_pod.return_value = {'desiredStatus': 'EXITED'}
        mock_get_logs.return_value = "Training complete"

        logs = []
        async for line in pod_manager.poll_training_logs('pod-xyz123', poll_interval_seconds=0.01):
            logs.append(line)

        assert len(logs) == 1
        assert logs[0] == "Training complete"


class TestWaitForCompletion:
    """Test wait for completion and auto-termination."""

    @patch('runpod.get_pod')
    @patch('runpod.terminate_pod')
    @patch('time.sleep')
    def test_wait_for_completion_success(
        self, mock_sleep, mock_terminate, mock_get_pod, pod_manager
    ):
        """wait_for_completion_and_terminate should wait for EXITED status."""
        # Simulate pod progression: RUNNING -> RUNNING -> EXITED
        mock_get_pod.side_effect = [
            {'desiredStatus': 'RUNNING', 'runtime': 30},
            {'desiredStatus': 'RUNNING', 'runtime': 60},
            {'desiredStatus': 'EXITED', 'runtime': 90}
        ]

        result = pod_manager.wait_for_completion_and_terminate('pod-xyz123')

        assert result['success'] is True
        assert result['pod_id'] == 'pod-xyz123'
        assert 'runtime_seconds' in result
        mock_terminate.assert_called_once_with('pod-xyz123')

    @patch('runpod.get_pod')
    @patch('runpod.terminate_pod')
    @patch('time.sleep')
    def test_wait_for_completion_failed_pod(
        self, mock_sleep, mock_terminate, mock_get_pod, pod_manager
    ):
        """wait_for_completion_and_terminate should handle failed pods."""
        mock_get_pod.return_value = {'desiredStatus': 'FAILED'}

        result = pod_manager.wait_for_completion_and_terminate('pod-xyz123')

        assert result['success'] is False
        assert 'Pod failed during training' in result['error']
        mock_terminate.assert_called_once_with('pod-xyz123')

    @patch('runpod.get_pod')
    @patch('runpod.terminate_pod')
    @patch('time.time')
    @patch('time.sleep')
    def test_wait_for_completion_timeout(
        self, mock_sleep, mock_time, mock_terminate, mock_get_pod, pod_manager
    ):
        """wait_for_completion_and_terminate should timeout after max_wait_seconds."""
        mock_get_pod.return_value = {'desiredStatus': 'RUNNING'}

        # Simulate timeout (start at 0, then exceed max_wait_seconds)
        mock_time.side_effect = [0, 10, 20, 7210]  # Exceeds 7200

        result = pod_manager.wait_for_completion_and_terminate(
            'pod-xyz123',
            max_wait_seconds=7200
        )

        assert result['success'] is False
        assert 'timeout' in result['error'].lower()
        mock_terminate.assert_called_once_with('pod-xyz123')

    @patch('runpod.get_pod')
    @patch('runpod.terminate_pod')
    @patch('time.sleep')
    def test_wait_for_completion_terminate_failure_handled(
        self, mock_sleep, mock_terminate, mock_get_pod, pod_manager
    ):
        """wait_for_completion_and_terminate should handle termination failures gracefully."""
        mock_get_pod.return_value = {'desiredStatus': 'EXITED'}
        mock_terminate.side_effect = Exception("Termination failed")

        # Should not raise exception, just complete
        result = pod_manager.wait_for_completion_and_terminate('pod-xyz123')

        assert result['success'] is True
