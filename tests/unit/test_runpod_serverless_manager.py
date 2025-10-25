"""
Tests for RunPodServerlessManager.

This module tests the RunPod serverless manager including endpoint invocation,
job status checking, and endpoint creation.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 5.6: RunPod Serverless Tests)
"""

import sys
import pytest
from unittest.mock import Mock, patch, MagicMock

# Mock runpod module before importing RunPodServerlessManager
sys.modules['runpod'] = MagicMock()

from src.cloud.runpod_serverless_manager import RunPodServerlessManager
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
        results_prefix="results",
        serverless_endpoint_id="endpoint-abc123"
    )


@pytest.fixture
def serverless_manager(runpod_config):
    """Create RunPodServerlessManager instance."""
    with patch('runpod.api_key'):
        manager = RunPodServerlessManager(runpod_config)
        return manager


class TestRunPodServerlessManagerInitialization:
    """Test RunPodServerlessManager initialization."""

    @patch('runpod.api_key')
    def test_init_sets_api_key(self, mock_api_key, runpod_config):
        """Initialization should set runpod.api_key."""
        manager = RunPodServerlessManager(runpod_config)

        assert manager.config == runpod_config
        assert runpod.api_key == runpod_config.runpod_api_key


class TestRunPodSynchronousPrediction:
    """Test synchronous prediction invocation."""

    @patch('runpod.Endpoint')
    def test_invoke_prediction_success(self, mock_endpoint_class, serverless_manager):
        """invoke_prediction should call endpoint.run_sync with correct parameters."""
        # Mock endpoint instance
        mock_endpoint = Mock()
        mock_endpoint.run_sync.return_value = {
            'success': True,
            'num_predictions': 1000,
            'output_key': 'predictions/user_12345/results.csv'
        }
        mock_endpoint_class.return_value = mock_endpoint

        # Invoke prediction
        result = serverless_manager.invoke_prediction(
            model_uri="models/user_12345/model_abc",
            data_uri="datasets/user_12345/test_data.csv",
            output_uri="predictions/user_12345/results.csv"
        )

        # Verify endpoint was created with correct ID
        mock_endpoint_class.assert_called_once_with("endpoint-abc123")

        # Verify run_sync was called with correct payload
        expected_input = {
            "input": {
                "model_key": "models/user_12345/model_abc",
                "data_key": "datasets/user_12345/test_data.csv",
                "output_key": "predictions/user_12345/results.csv",
                "volume_id": "vol-test123",
                "prediction_column_name": "prediction"
            }
        }
        mock_endpoint.run_sync.assert_called_once()
        call_args = mock_endpoint.run_sync.call_args
        assert call_args[0][0] == expected_input
        assert call_args[1]['timeout'] == 300

        # Verify result
        assert result['success'] is True
        assert result['num_predictions'] == 1000

    @patch('runpod.Endpoint')
    def test_invoke_prediction_with_feature_columns(self, mock_endpoint_class, serverless_manager):
        """invoke_prediction should include feature_columns in payload."""
        mock_endpoint = Mock()
        mock_endpoint.run_sync.return_value = {'success': True}
        mock_endpoint_class.return_value = mock_endpoint

        result = serverless_manager.invoke_prediction(
            model_uri="models/user_12345/model_abc",
            data_uri="datasets/user_12345/test_data.csv",
            output_uri="predictions/user_12345/results.csv",
            feature_columns=["col1", "col2", "col3"]
        )

        # Check feature_columns in payload
        call_args = mock_endpoint.run_sync.call_args[0][0]
        assert call_args["input"]["feature_columns"] == ["col1", "col2", "col3"]

    @patch('runpod.Endpoint')
    def test_invoke_prediction_custom_column_name(self, mock_endpoint_class, serverless_manager):
        """invoke_prediction should support custom prediction column name."""
        mock_endpoint = Mock()
        mock_endpoint.run_sync.return_value = {'success': True}
        mock_endpoint_class.return_value = mock_endpoint

        result = serverless_manager.invoke_prediction(
            model_uri="models/user_12345/model_abc",
            data_uri="datasets/user_12345/test_data.csv",
            output_uri="predictions/user_12345/results.csv",
            prediction_column_name="predicted_value"
        )

        # Check custom column name
        call_args = mock_endpoint.run_sync.call_args[0][0]
        assert call_args["input"]["prediction_column_name"] == "predicted_value"

    def test_invoke_prediction_no_endpoint_id_raises_error(self):
        """invoke_prediction should raise ValueError if no endpoint_id available."""
        # Config without endpoint_id
        config = RunPodConfig(
            runpod_api_key="test-key",
            storage_endpoint="https://storage.runpod.io",
            network_volume_id="vol-123",
            default_gpu_type="NVIDIA RTX A5000",
            cloud_type="COMMUNITY",
            storage_access_key="test-key",
            storage_secret_key="test-secret",
            data_prefix="datasets",
            models_prefix="models",
            results_prefix="results"
            # No endpoint_id
        )

        with patch('runpod.api_key'):
            manager = RunPodServerlessManager(config)

        with pytest.raises(ValueError) as exc_info:
            manager.invoke_prediction(
                model_uri="models/test",
                data_uri="datasets/test",
                output_uri="predictions/test"
            )

        assert "endpoint_id" in str(exc_info.value).lower()


class TestRunPodAsynchronousPrediction:
    """Test asynchronous prediction invocation."""

    @patch('runpod.Endpoint')
    def test_invoke_async_returns_job_id(self, mock_endpoint_class, serverless_manager):
        """invoke_async should return job ID."""
        mock_endpoint = Mock()
        mock_endpoint.run.return_value = {'id': 'job-xyz123'}
        mock_endpoint_class.return_value = mock_endpoint

        job_id = serverless_manager.invoke_async(
            model_uri="models/user_12345/model_abc",
            data_uri="datasets/user_12345/large_data.csv",
            output_uri="predictions/user_12345/results.csv"
        )

        assert job_id == "job-xyz123"
        mock_endpoint.run.assert_called_once()

    @patch('runpod.Endpoint')
    def test_check_job_status(self, mock_endpoint_class, serverless_manager):
        """check_job_status should query endpoint status."""
        mock_endpoint = Mock()
        mock_endpoint.status.return_value = {
            'status': 'COMPLETED',
            'output': {'success': True, 'num_predictions': 5000}
        }
        mock_endpoint_class.return_value = mock_endpoint

        status = serverless_manager.check_job_status("endpoint-abc123", "job-xyz123")

        assert status['status'] == 'COMPLETED'
        assert status['output']['num_predictions'] == 5000
        mock_endpoint.status.assert_called_once_with("job-xyz123")

    @patch('runpod.Endpoint')
    @patch('time.sleep')
    def test_wait_for_completion_success(self, mock_sleep, mock_endpoint_class, serverless_manager):
        """wait_for_completion should poll until job completes."""
        mock_endpoint = Mock()

        # Simulate job progressing from IN_PROGRESS to COMPLETED
        mock_endpoint.status.side_effect = [
            {'status': 'IN_QUEUE'},
            {'status': 'IN_PROGRESS'},
            {'status': 'COMPLETED', 'output': {'success': True}}
        ]
        mock_endpoint_class.return_value = mock_endpoint

        result = serverless_manager.wait_for_completion("endpoint-abc123", "job-xyz123", poll_interval=1)

        assert result['status'] == 'COMPLETED'
        assert mock_endpoint.status.call_count == 3

    @patch('runpod.Endpoint')
    @patch('time.sleep')
    @patch('time.time')
    def test_wait_for_completion_timeout(self, mock_time, mock_sleep, mock_endpoint_class, serverless_manager):
        """wait_for_completion should raise TimeoutError if job doesn't complete."""
        mock_endpoint = Mock()
        mock_endpoint.status.return_value = {'status': 'IN_PROGRESS'}
        mock_endpoint_class.return_value = mock_endpoint

        # Simulate timeout
        mock_time.side_effect = [0, 100, 200, 310]  # Exceeds 300s timeout

        with pytest.raises(TimeoutError) as exc_info:
            serverless_manager.wait_for_completion("endpoint-abc123", "job-xyz123", timeout=300)

        assert "did not complete within 300 seconds" in str(exc_info.value)

    @patch('runpod.Endpoint')
    @patch('time.sleep')
    def test_wait_for_completion_failed_job(self, mock_sleep, mock_endpoint_class, serverless_manager):
        """wait_for_completion should raise RuntimeError if job fails."""
        mock_endpoint = Mock()
        mock_endpoint.status.return_value = {
            'status': 'FAILED',
            'error': 'GPU out of memory'
        }
        mock_endpoint_class.return_value = mock_endpoint

        with pytest.raises(RuntimeError) as exc_info:
            serverless_manager.wait_for_completion("endpoint-abc123", "job-xyz123")

        assert "GPU out of memory" in str(exc_info.value)


class TestRunPodEndpointManagement:
    """Test endpoint creation and listing."""

    @patch('requests.post')
    def test_create_endpoint(self, mock_post, serverless_manager):
        """create_endpoint should call GraphQL API with correct parameters."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': {
                'createEndpoint': {
                    'id': 'endpoint-new123',
                    'name': 'test-endpoint',
                    'idleTimeout': 30,
                    'scaleType': 'AUTOSCALE'
                }
            }
        }
        mock_post.return_value = mock_response

        result = serverless_manager.create_endpoint(
            name="test-endpoint",
            docker_image="username/ml-prediction:latest",
            gpu_type="NVIDIA RTX A40",
            max_workers=5
        )

        assert result['id'] == "endpoint-new123"
        assert result['name'] == "test-endpoint"

        # Verify GraphQL request
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert 'json' in call_kwargs
        assert 'query' in call_kwargs['json']

    @patch('requests.post')
    def test_list_endpoints(self, mock_post, serverless_manager):
        """list_endpoints should retrieve all endpoints."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': {
                'myself': {
                    'endpoints': [
                        {'id': 'endpoint-1', 'name': 'predictions', 'activeWorkers': 0},
                        {'id': 'endpoint-2', 'name': 'training', 'activeWorkers': 1}
                    ]
                }
            }
        }
        mock_post.return_value = mock_response

        endpoints = serverless_manager.list_endpoints()

        assert len(endpoints) == 2
        assert endpoints[0]['id'] == "endpoint-1"
        assert endpoints[1]['name'] == "training"
