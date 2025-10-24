"""
Unit tests for CostTracker class.

Tests cost estimation, calculation, and tracking for AWS services:
- EC2 spot instance cost estimation and calculation
- Lambda function cost estimation and calculation
- S3 storage cost calculation
- User spending tracking and cost limit enforcement
- Cost logging to JSON file

Following TDD: Tests written FIRST before implementation.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 6.0: CostTracker with TDD)
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, Mock, patch, mock_open

import pytest

from src.cloud.aws_config import CloudConfig
from src.cloud.cost_tracker import CostTracker
from src.cloud.exceptions import CostTrackingError


@pytest.fixture
def mock_config() -> CloudConfig:
    """Create mock CloudConfig for testing."""
    config = Mock(spec=CloudConfig)
    config.aws_region = "us-east-1"
    config.aws_access_key_id = "test_key"
    config.aws_secret_access_key = "test_secret"
    config.s3_bucket = "test-bucket"
    config.s3_data_prefix = "datasets"
    config.s3_models_prefix = "models"
    config.s3_results_prefix = "results"
    config.ec2_instance_type = "m5.large"
    config.ec2_ami_id = "ami-12345"
    config.ec2_key_name = "test-key"
    config.ec2_security_group = "sg-12345"
    config.lambda_function_name = "test-function"
    config.lambda_memory_mb = 3008
    config.lambda_timeout_seconds = 300
    config.max_training_cost_dollars = 10.0
    config.max_prediction_cost_dollars = 1.0
    config.cost_warning_threshold = 0.8
    return config


@pytest.fixture
def cost_tracker(mock_config: CloudConfig) -> CostTracker:
    """Create CostTracker instance for testing."""
    return CostTracker(config=mock_config)


@pytest.fixture
def temp_log_dir():
    """Create temporary directory for cost logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestCostTrackerInitialization:
    """Test CostTracker initialization and pricing constants."""

    def test_init_with_config(self, mock_config: CloudConfig):
        """Test initialization with CloudConfig."""
        tracker = CostTracker(config=mock_config)

        assert tracker._config == mock_config
        assert tracker._log_path is not None

    def test_pricing_constants_exist(self, cost_tracker: CostTracker):
        """Test that pricing constants are defined."""
        # EC2 spot pricing
        assert hasattr(cost_tracker, 'EC2_SPOT_PRICING')
        assert isinstance(cost_tracker.EC2_SPOT_PRICING, dict)
        assert 't3.medium' in cost_tracker.EC2_SPOT_PRICING
        assert 'm5.large' in cost_tracker.EC2_SPOT_PRICING
        assert 'm5.xlarge' in cost_tracker.EC2_SPOT_PRICING
        assert 'p3.2xlarge' in cost_tracker.EC2_SPOT_PRICING

        # Lambda pricing
        assert hasattr(cost_tracker, 'LAMBDA_PRICE_PER_GB_SECOND')
        assert cost_tracker.LAMBDA_PRICE_PER_GB_SECOND == 0.0000166667

        assert hasattr(cost_tracker, 'LAMBDA_PRICE_PER_REQUEST')
        assert cost_tracker.LAMBDA_PRICE_PER_REQUEST == 0.0000002

        # S3 pricing
        assert hasattr(cost_tracker, 'S3_STORAGE_PRICE_PER_GB_MONTH')
        assert cost_tracker.S3_STORAGE_PRICE_PER_GB_MONTH == 0.023

    def test_ec2_pricing_values(self, cost_tracker: CostTracker):
        """Test EC2 spot pricing values."""
        assert cost_tracker.EC2_SPOT_PRICING['t3.medium'] == 0.01
        assert cost_tracker.EC2_SPOT_PRICING['m5.large'] == 0.05
        assert cost_tracker.EC2_SPOT_PRICING['m5.xlarge'] == 0.10
        assert cost_tracker.EC2_SPOT_PRICING['p3.2xlarge'] == 1.20


class TestTrainingCostEstimation:
    """Test EC2 training cost estimation."""

    def test_estimate_training_cost_t3_medium(self, cost_tracker: CostTracker):
        """Test cost estimation for t3.medium instance."""
        cost = cost_tracker.estimate_training_cost(
            instance_type='t3.medium',
            estimated_time_minutes=60
        )

        # $0.01/hour * 1 hour = $0.01
        assert cost == pytest.approx(0.01, abs=0.001)

    def test_estimate_training_cost_m5_large(self, cost_tracker: CostTracker):
        """Test cost estimation for m5.large instance."""
        cost = cost_tracker.estimate_training_cost(
            instance_type='m5.large',
            estimated_time_minutes=120
        )

        # $0.05/hour * 2 hours = $0.10
        assert cost == pytest.approx(0.10, abs=0.001)

    def test_estimate_training_cost_p3_2xlarge(self, cost_tracker: CostTracker):
        """Test cost estimation for p3.2xlarge instance."""
        cost = cost_tracker.estimate_training_cost(
            instance_type='p3.2xlarge',
            estimated_time_minutes=30
        )

        # $1.20/hour * 0.5 hours = $0.60
        assert cost == pytest.approx(0.60, abs=0.001)

    def test_estimate_training_cost_fractional_minutes(self, cost_tracker: CostTracker):
        """Test cost estimation with fractional minutes."""
        cost = cost_tracker.estimate_training_cost(
            instance_type='m5.xlarge',
            estimated_time_minutes=45
        )

        # $0.10/hour * 0.75 hours = $0.075
        assert cost == pytest.approx(0.075, abs=0.001)

    def test_estimate_training_cost_unknown_instance_type(self, cost_tracker: CostTracker):
        """Test cost estimation with unknown instance type raises error."""
        with pytest.raises(CostTrackingError) as exc_info:
            cost_tracker.estimate_training_cost(
                instance_type='unknown.type',
                estimated_time_minutes=60
            )

        assert "Unknown instance type" in str(exc_info.value)
        assert "unknown.type" in str(exc_info.value)

    def test_estimate_training_cost_zero_minutes(self, cost_tracker: CostTracker):
        """Test cost estimation with zero minutes returns zero."""
        cost = cost_tracker.estimate_training_cost(
            instance_type='m5.large',
            estimated_time_minutes=0
        )

        assert cost == 0.0


class TestTrainingCostCalculation:
    """Test actual EC2 training cost calculation."""

    @patch('boto3.client')
    def test_calculate_training_cost_success(
        self,
        mock_boto_client: MagicMock,
        cost_tracker: CostTracker
    ):
        """Test actual training cost calculation."""
        # Mock EC2 describe_instances response
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2

        # Instance ran for 2 hours
        launch_time = datetime(2025, 10, 24, 10, 0, 0)
        current_time = datetime(2025, 10, 24, 12, 0, 0)

        mock_ec2.describe_instances.return_value = {
            'Reservations': [{
                'Instances': [{
                    'InstanceId': 'i-12345',
                    'LaunchTime': launch_time,
                    'State': {'Name': 'running'}
                }]
            }]
        }

        with patch('src.cloud.cost_tracker.datetime') as mock_datetime:
            mock_datetime.now.return_value = current_time

            cost = cost_tracker.calculate_training_cost(
                instance_id='i-12345',
                instance_type='m5.large'
            )

        # $0.05/hour * 2 hours = $0.10
        assert cost == pytest.approx(0.10, abs=0.001)

    @patch('boto3.client')
    def test_calculate_training_cost_terminated_instance(
        self,
        mock_boto_client: MagicMock,
        cost_tracker: CostTracker
    ):
        """Test cost calculation for terminated instance."""
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2

        # Instance ran for 1.5 hours
        launch_time = datetime(2025, 10, 24, 10, 0, 0)
        termination_time = datetime(2025, 10, 24, 11, 30, 0)

        mock_ec2.describe_instances.return_value = {
            'Reservations': [{
                'Instances': [{
                    'InstanceId': 'i-12345',
                    'LaunchTime': launch_time,
                    'State': {'Name': 'terminated'},
                    'StateTransitionReason': f'User initiated ({termination_time.isoformat()})'
                }]
            }]
        }

        cost = cost_tracker.calculate_training_cost(
            instance_id='i-12345',
            instance_type='m5.xlarge'
        )

        # $0.10/hour * 1.5 hours = $0.15
        assert cost == pytest.approx(0.15, abs=0.001)

    @patch('boto3.client')
    def test_calculate_training_cost_instance_not_found(
        self,
        mock_boto_client: MagicMock,
        cost_tracker: CostTracker
    ):
        """Test cost calculation when instance not found."""
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2

        mock_ec2.describe_instances.return_value = {
            'Reservations': []
        }

        with pytest.raises(CostTrackingError) as exc_info:
            cost_tracker.calculate_training_cost(
                instance_id='i-nonexistent',
                instance_type='m5.large'
            )

        assert "Instance not found" in str(exc_info.value)


class TestPredictionCostEstimation:
    """Test Lambda prediction cost estimation."""

    def test_estimate_prediction_cost_basic(self, cost_tracker: CostTracker):
        """Test basic prediction cost estimation."""
        cost = cost_tracker.estimate_prediction_cost(
            num_rows=1000,
            lambda_memory_mb=3008,
            estimated_time_seconds=60
        )

        # Lambda compute: (3008 MB / 1024) GB * 60 sec * $0.0000166667 = ~$0.00293
        # Lambda requests: 1 request * $0.0000002 = $0.0000002
        # Total: ~$0.00293
        assert cost > 0.002
        assert cost < 0.004

    def test_estimate_prediction_cost_large_dataset(self, cost_tracker: CostTracker):
        """Test prediction cost for large dataset."""
        cost = cost_tracker.estimate_prediction_cost(
            num_rows=100000,
            lambda_memory_mb=3008,
            estimated_time_seconds=300
        )

        # Should be higher than basic case
        assert cost > 0.01

    def test_estimate_prediction_cost_small_memory(self, cost_tracker: CostTracker):
        """Test prediction cost with small memory allocation."""
        cost = cost_tracker.estimate_prediction_cost(
            num_rows=100,
            lambda_memory_mb=128,
            estimated_time_seconds=10
        )

        # Should be very small
        assert cost < 0.001

    def test_estimate_prediction_cost_zero_time(self, cost_tracker: CostTracker):
        """Test prediction cost with zero time."""
        cost = cost_tracker.estimate_prediction_cost(
            num_rows=1000,
            lambda_memory_mb=3008,
            estimated_time_seconds=0
        )

        # Should only include request cost
        assert cost == pytest.approx(0.0000002, abs=0.0000001)


class TestPredictionCostCalculation:
    """Test actual Lambda prediction cost calculation."""

    def test_calculate_prediction_cost_basic(self, cost_tracker: CostTracker):
        """Test actual prediction cost calculation."""
        cost = cost_tracker.calculate_prediction_cost(
            execution_time_ms=5000,  # 5 seconds
            lambda_memory_mb=3008
        )

        # Lambda compute: (3008 MB / 1024) GB * 5 sec * $0.0000166667 = ~$0.000244
        # Lambda requests: 1 request * $0.0000002 = $0.0000002
        # Total: ~$0.000244
        assert cost == pytest.approx(0.000244, abs=0.00001)

    def test_calculate_prediction_cost_high_memory(self, cost_tracker: CostTracker):
        """Test prediction cost with high memory."""
        cost = cost_tracker.calculate_prediction_cost(
            execution_time_ms=10000,  # 10 seconds
            lambda_memory_mb=10240  # 10GB
        )

        # Should be higher than basic case
        assert cost > 0.001

    def test_calculate_prediction_cost_fast_execution(self, cost_tracker: CostTracker):
        """Test prediction cost with fast execution."""
        cost = cost_tracker.calculate_prediction_cost(
            execution_time_ms=100,  # 100ms
            lambda_memory_mb=3008
        )

        # Should be very small
        assert cost < 0.0001


class TestS3StorageCost:
    """Test S3 storage cost calculation."""

    def test_calculate_s3_storage_cost_empty(self, cost_tracker: CostTracker):
        """Test S3 storage cost when user has no files."""
        mock_s3_manager = Mock()
        mock_s3_manager.list_user_datasets.return_value = []
        mock_s3_manager.list_user_models.return_value = []

        result = cost_tracker.calculate_s3_storage_cost(
            user_id=12345,
            s3_manager=mock_s3_manager
        )

        assert result['total_storage_gb'] == 0.0
        assert result['monthly_cost'] == 0.0
        assert result['datasets_gb'] == 0.0
        assert result['models_gb'] == 0.0

    def test_calculate_s3_storage_cost_with_datasets(self, cost_tracker: CostTracker):
        """Test S3 storage cost with datasets."""
        mock_s3_manager = Mock()
        mock_s3_manager.list_user_datasets.return_value = [
            {'size_mb': 100.0},  # 100 MB
            {'size_mb': 200.0},  # 200 MB
        ]
        mock_s3_manager.list_user_models.return_value = []

        result = cost_tracker.calculate_s3_storage_cost(
            user_id=12345,
            s3_manager=mock_s3_manager
        )

        # 300 MB = 0.29296875 GB
        expected_gb = 300.0 / 1024
        expected_cost = expected_gb * 0.023

        assert result['total_storage_gb'] == pytest.approx(expected_gb, abs=0.001)
        assert result['monthly_cost'] == pytest.approx(expected_cost, abs=0.0001)
        assert result['datasets_gb'] == pytest.approx(expected_gb, abs=0.001)
        assert result['models_gb'] == 0.0

    def test_calculate_s3_storage_cost_with_models(self, cost_tracker: CostTracker):
        """Test S3 storage cost with models."""
        mock_s3_manager = Mock()
        mock_s3_manager.list_user_datasets.return_value = []
        mock_s3_manager.list_user_models.return_value = [
            {'size_mb': 500.0},  # 500 MB
            {'size_mb': 1024.0},  # 1 GB
        ]

        result = cost_tracker.calculate_s3_storage_cost(
            user_id=12345,
            s3_manager=mock_s3_manager
        )

        # 1524 MB = 1.48828125 GB
        expected_gb = 1524.0 / 1024
        expected_cost = expected_gb * 0.023

        assert result['total_storage_gb'] == pytest.approx(expected_gb, abs=0.001)
        assert result['monthly_cost'] == pytest.approx(expected_cost, abs=0.0001)
        assert result['datasets_gb'] == 0.0
        assert result['models_gb'] == pytest.approx(expected_gb, abs=0.001)

    def test_calculate_s3_storage_cost_mixed(self, cost_tracker: CostTracker):
        """Test S3 storage cost with both datasets and models."""
        mock_s3_manager = Mock()
        mock_s3_manager.list_user_datasets.return_value = [
            {'size_mb': 100.0},
        ]
        mock_s3_manager.list_user_models.return_value = [
            {'size_mb': 500.0},
        ]

        result = cost_tracker.calculate_s3_storage_cost(
            user_id=12345,
            s3_manager=mock_s3_manager
        )

        # 600 MB = 0.5859375 GB
        expected_total_gb = 600.0 / 1024
        expected_datasets_gb = 100.0 / 1024
        expected_models_gb = 500.0 / 1024
        expected_cost = expected_total_gb * 0.023

        assert result['total_storage_gb'] == pytest.approx(expected_total_gb, abs=0.001)
        assert result['monthly_cost'] == pytest.approx(expected_cost, abs=0.0001)
        assert result['datasets_gb'] == pytest.approx(expected_datasets_gb, abs=0.001)
        assert result['models_gb'] == pytest.approx(expected_models_gb, abs=0.001)


class TestUserTotalSpend:
    """Test user total spend tracking."""

    def test_get_user_total_spend_no_logs(self, cost_tracker: CostTracker, temp_log_dir: Path):
        """Test total spend when no logs exist."""
        cost_tracker._log_path = temp_log_dir / "cloud_costs.json"

        total = cost_tracker.get_user_total_spend(user_id=12345)

        assert total == 0.0

    def test_get_user_total_spend_with_costs(self, cost_tracker: CostTracker, temp_log_dir: Path):
        """Test total spend calculation from logs."""
        log_path = temp_log_dir / "cloud_costs.json"
        cost_tracker._log_path = log_path

        # Write sample log entries
        log_entries = [
            {
                'timestamp': '2025-10-24T10:00:00',
                'user_id': 12345,
                'service': 'ec2',
                'operation': 'training',
                'cost': 0.50
            },
            {
                'timestamp': '2025-10-24T11:00:00',
                'user_id': 12345,
                'service': 'lambda',
                'operation': 'prediction',
                'cost': 0.01
            },
            {
                'timestamp': '2025-10-24T12:00:00',
                'user_id': 99999,  # Different user
                'service': 'ec2',
                'operation': 'training',
                'cost': 1.00
            },
            {
                'timestamp': '2025-10-24T13:00:00',
                'user_id': 12345,
                'service': 's3',
                'operation': 'storage',
                'cost': 0.05
            }
        ]

        with open(log_path, 'w') as f:
            for entry in log_entries:
                f.write(json.dumps(entry) + '\n')

        total = cost_tracker.get_user_total_spend(user_id=12345)

        # Sum: 0.50 + 0.01 + 0.05 = 0.56
        assert total == pytest.approx(0.56, abs=0.001)

    def test_get_user_total_spend_different_user(self, cost_tracker: CostTracker, temp_log_dir: Path):
        """Test total spend for different user."""
        log_path = temp_log_dir / "cloud_costs.json"
        cost_tracker._log_path = log_path

        log_entries = [
            {
                'timestamp': '2025-10-24T10:00:00',
                'user_id': 12345,
                'service': 'ec2',
                'operation': 'training',
                'cost': 0.50
            },
            {
                'timestamp': '2025-10-24T11:00:00',
                'user_id': 99999,
                'service': 'ec2',
                'operation': 'training',
                'cost': 2.00
            }
        ]

        with open(log_path, 'w') as f:
            for entry in log_entries:
                f.write(json.dumps(entry) + '\n')

        total = cost_tracker.get_user_total_spend(user_id=99999)

        assert total == pytest.approx(2.00, abs=0.001)


class TestCostLimitEnforcement:
    """Test cost limit checking."""

    def test_check_cost_limit_under_limit(self, cost_tracker: CostTracker, temp_log_dir: Path):
        """Test cost limit check when under limit."""
        cost_tracker._log_path = temp_log_dir / "cloud_costs.json"

        # No existing costs
        result = cost_tracker.check_cost_limit(
            user_id=12345,
            operation_cost=0.50,
            limit_type='training'
        )

        assert result is True

    def test_check_cost_limit_exceeds_limit(self, cost_tracker: CostTracker, temp_log_dir: Path):
        """Test cost limit check when exceeding limit."""
        log_path = temp_log_dir / "cloud_costs.json"
        cost_tracker._log_path = log_path

        # User already spent $9.50
        log_entries = [
            {
                'timestamp': '2025-10-24T10:00:00',
                'user_id': 12345,
                'service': 'ec2',
                'operation': 'training',
                'cost': 9.50
            }
        ]

        with open(log_path, 'w') as f:
            for entry in log_entries:
                f.write(json.dumps(entry) + '\n')

        # Trying to spend another $1.00 would exceed $10 limit
        result = cost_tracker.check_cost_limit(
            user_id=12345,
            operation_cost=1.00,
            limit_type='training'
        )

        assert result is False

    def test_check_cost_limit_at_exact_limit(self, cost_tracker: CostTracker, temp_log_dir: Path):
        """Test cost limit check at exact limit."""
        log_path = temp_log_dir / "cloud_costs.json"
        cost_tracker._log_path = log_path

        # User already spent $9.00
        log_entries = [
            {
                'timestamp': '2025-10-24T10:00:00',
                'user_id': 12345,
                'service': 'ec2',
                'operation': 'training',
                'cost': 9.00
            }
        ]

        with open(log_path, 'w') as f:
            for entry in log_entries:
                f.write(json.dumps(entry) + '\n')

        # Trying to spend exactly $1.00 reaches $10 limit (should be allowed)
        result = cost_tracker.check_cost_limit(
            user_id=12345,
            operation_cost=1.00,
            limit_type='training'
        )

        assert result is True

    def test_check_cost_limit_prediction_type(self, cost_tracker: CostTracker, temp_log_dir: Path):
        """Test cost limit check for prediction type."""
        cost_tracker._log_path = temp_log_dir / "cloud_costs.json"

        # No existing costs
        result = cost_tracker.check_cost_limit(
            user_id=12345,
            operation_cost=0.50,
            limit_type='prediction'
        )

        # Max prediction cost is $1.00
        assert result is True

    def test_check_cost_limit_invalid_type(self, cost_tracker: CostTracker):
        """Test cost limit check with invalid limit type."""
        with pytest.raises(CostTrackingError) as exc_info:
            cost_tracker.check_cost_limit(
                user_id=12345,
                operation_cost=0.50,
                limit_type='invalid'
            )

        assert "Invalid limit type" in str(exc_info.value)


class TestCostLogging:
    """Test cost logging to JSON file."""

    def test_log_cost_creates_file(self, cost_tracker: CostTracker, temp_log_dir: Path):
        """Test that logging creates log file."""
        log_path = temp_log_dir / "cloud_costs.json"
        cost_tracker._log_path = log_path

        cost_tracker._log_cost(
            service='ec2',
            operation='training',
            user_id=12345,
            cost=0.50,
            instance_type='m5.large',
            duration_minutes=60
        )

        assert log_path.exists()

    def test_log_cost_content(self, cost_tracker: CostTracker, temp_log_dir: Path):
        """Test logged cost entry content."""
        log_path = temp_log_dir / "cloud_costs.json"
        cost_tracker._log_path = log_path

        cost_tracker._log_cost(
            service='ec2',
            operation='training',
            user_id=12345,
            cost=0.50,
            instance_type='m5.large',
            duration_minutes=60
        )

        with open(log_path) as f:
            entry = json.loads(f.readline())

        assert entry['service'] == 'ec2'
        assert entry['operation'] == 'training'
        assert entry['user_id'] == 12345
        assert entry['cost'] == 0.50
        assert entry['instance_type'] == 'm5.large'
        assert entry['duration_minutes'] == 60
        assert 'timestamp' in entry

    def test_log_cost_appends(self, cost_tracker: CostTracker, temp_log_dir: Path):
        """Test that logging appends to existing file."""
        log_path = temp_log_dir / "cloud_costs.json"
        cost_tracker._log_path = log_path

        # First entry
        cost_tracker._log_cost(
            service='ec2',
            operation='training',
            user_id=12345,
            cost=0.50
        )

        # Second entry
        cost_tracker._log_cost(
            service='lambda',
            operation='prediction',
            user_id=12345,
            cost=0.01
        )

        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])

        assert entry1['service'] == 'ec2'
        assert entry2['service'] == 'lambda'

    def test_log_cost_creates_parent_directory(self, cost_tracker: CostTracker, temp_log_dir: Path):
        """Test that logging creates parent directory if needed."""
        log_path = temp_log_dir / "logs" / "cloud_costs.json"
        cost_tracker._log_path = log_path

        cost_tracker._log_cost(
            service='s3',
            operation='storage',
            user_id=12345,
            cost=0.05
        )

        assert log_path.parent.exists()
        assert log_path.exists()


class TestGetInstanceRuntime:
    """Test EC2 instance runtime calculation."""

    @patch('boto3.client')
    def test_get_instance_runtime_running(
        self,
        mock_boto_client: MagicMock,
        cost_tracker: CostTracker
    ):
        """Test runtime calculation for running instance."""
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2

        # Instance launched 2 hours ago
        launch_time = datetime(2025, 10, 24, 10, 0, 0)
        current_time = datetime(2025, 10, 24, 12, 0, 0)

        mock_ec2.describe_instances.return_value = {
            'Reservations': [{
                'Instances': [{
                    'InstanceId': 'i-12345',
                    'LaunchTime': launch_time,
                    'State': {'Name': 'running'}
                }]
            }]
        }

        with patch('src.cloud.cost_tracker.datetime') as mock_datetime:
            mock_datetime.now.return_value = current_time

            runtime = cost_tracker._get_instance_runtime(instance_id='i-12345')

        # 2 hours = 120 minutes
        assert runtime == pytest.approx(120.0, abs=0.1)

    @patch('boto3.client')
    def test_get_instance_runtime_terminated(
        self,
        mock_boto_client: MagicMock,
        cost_tracker: CostTracker
    ):
        """Test runtime calculation for terminated instance."""
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2

        # Instance ran for 90 minutes
        launch_time = datetime(2025, 10, 24, 10, 0, 0)
        termination_time = datetime(2025, 10, 24, 11, 30, 0)

        mock_ec2.describe_instances.return_value = {
            'Reservations': [{
                'Instances': [{
                    'InstanceId': 'i-12345',
                    'LaunchTime': launch_time,
                    'State': {'Name': 'terminated'},
                    'StateTransitionReason': f'User initiated ({termination_time.isoformat()})'
                }]
            }]
        }

        runtime = cost_tracker._get_instance_runtime(instance_id='i-12345')

        # 90 minutes
        assert runtime == pytest.approx(90.0, abs=0.1)
