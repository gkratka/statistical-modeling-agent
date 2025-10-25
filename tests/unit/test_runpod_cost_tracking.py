"""
Tests for RunPod cost tracking in CostTracker.

Tests RunPod GPU pricing, serverless pricing, storage costs, and per-second
billing calculations.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 3.6: RunPod Cost Tracking Tests)
"""

import pytest
from unittest.mock import Mock

from src.cloud.cost_tracker import CostTracker
from src.cloud.runpod_config import RunPodConfig
from src.cloud.exceptions import CostTrackingError


@pytest.fixture
def runpod_config():
    """Create RunPod configuration for testing."""
    return RunPodConfig(
        runpod_api_key="test-api-key",
        network_volume_id="vol-123",
        storage_access_key="AKIAIOSFODNN7EXAMPLE",
        storage_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        max_training_cost_dollars=10.0,
        max_prediction_cost_dollars=1.0
    )


@pytest.fixture
def cost_tracker(runpod_config):
    """Create CostTracker instance (works with both AWS and RunPod)."""
    # CostTracker expects CloudConfig, but we'll test RunPod methods directly
    from src.cloud.aws_config import CloudConfig
    aws_config = CloudConfig(
        aws_region="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        s3_bucket="test-bucket",
        s3_data_prefix="datasets",
        s3_models_prefix="models",
        s3_results_prefix="results",
        ec2_instance_type="m5.large",
        ec2_ami_id="ami-123",
        ec2_key_name="test-key",
        ec2_security_group="sg-123",
        lambda_function_name="test-function",
        max_training_cost_dollars=10.0,
        max_prediction_cost_dollars=1.0
    )
    return CostTracker(aws_config)


class TestRunPodPricingConstants:
    """Test RunPod pricing constants are defined correctly."""

    def test_gpu_pricing_constants_exist(self, cost_tracker):
        """RunPod GPU pricing should be defined."""
        assert hasattr(cost_tracker, 'RUNPOD_GPU_PRICING')
        assert isinstance(cost_tracker.RUNPOD_GPU_PRICING, dict)
        assert len(cost_tracker.RUNPOD_GPU_PRICING) > 0

    def test_gpu_pricing_has_common_types(self, cost_tracker):
        """Common GPU types should be in pricing."""
        pricing = cost_tracker.RUNPOD_GPU_PRICING
        assert 'NVIDIA RTX A5000' in pricing
        assert 'NVIDIA RTX A40' in pricing
        assert 'NVIDIA A100 PCIe 40GB' in pricing

    def test_serverless_pricing_constants_exist(self, cost_tracker):
        """RunPod serverless pricing should be defined."""
        assert hasattr(cost_tracker, 'RUNPOD_SERVERLESS_PRICING')
        assert isinstance(cost_tracker.RUNPOD_SERVERLESS_PRICING, dict)

    def test_serverless_pricing_per_second(self, cost_tracker):
        """Serverless pricing should be per-second (much smaller than hourly)."""
        pricing = cost_tracker.RUNPOD_SERVERLESS_PRICING
        for gpu_type, rate in pricing.items():
            assert rate < 0.001, f"{gpu_type} rate should be per-second (very small)"

    def test_storage_pricing_constants_exist(self, cost_tracker):
        """RunPod storage pricing should be defined."""
        assert hasattr(cost_tracker, 'RUNPOD_STORAGE_PRICE_PER_GB_MONTH')
        assert hasattr(cost_tracker, 'RUNPOD_STORAGE_PRICE_PER_GB_MONTH_RUNNING')
        assert cost_tracker.RUNPOD_STORAGE_PRICE_PER_GB_MONTH == 0.07
        assert cost_tracker.RUNPOD_STORAGE_PRICE_PER_GB_MONTH_RUNNING == 0.10


class TestRunPodTrainingCostEstimation:
    """Test RunPod GPU training cost estimation (Task 3.2)."""

    def test_estimate_runpod_training_cost_30_minutes(self, cost_tracker):
        """Estimate cost for 30-minute training on RTX A5000."""
        # RTX A5000 is $0.29/hour, 30 minutes = 1800 seconds
        cost = cost_tracker.estimate_runpod_training_cost('NVIDIA RTX A5000', 1800)

        expected = 0.29 * (1800 / 3600)  # $0.29 * 0.5 hours = $0.145
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_estimate_runpod_training_cost_1_hour(self, cost_tracker):
        """Estimate cost for 1-hour training on RTX A40."""
        # RTX A40 is $0.39/hour
        cost = cost_tracker.estimate_runpod_training_cost('NVIDIA RTX A40', 3600)

        assert cost == pytest.approx(0.39, rel=1e-6)

    def test_estimate_runpod_training_cost_2_hours(self, cost_tracker):
        """Estimate cost for 2-hour training on A100."""
        # A100 40GB is $0.79/hour, 2 hours = 7200 seconds
        cost = cost_tracker.estimate_runpod_training_cost('NVIDIA A100 PCIe 40GB', 7200)

        expected = 0.79 * 2  # $1.58
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_estimate_runpod_training_cost_invalid_gpu_raises_error(self, cost_tracker):
        """Unknown GPU type should raise CostTrackingError."""
        with pytest.raises(CostTrackingError) as exc_info:
            cost_tracker.estimate_runpod_training_cost('NVIDIA RTX 9090', 1800)

        assert "Unknown GPU type" in str(exc_info.value)
        assert "NVIDIA RTX A5000" in str(exc_info.value)

    def test_estimate_runpod_training_cost_short_job(self, cost_tracker):
        """Estimate cost for short 5-minute job."""
        # RTX A5000: 5 minutes = 300 seconds
        cost = cost_tracker.estimate_runpod_training_cost('NVIDIA RTX A5000', 300)

        expected = 0.29 * (300 / 3600)  # $0.29 * (1/12) = ~$0.024
        assert cost == pytest.approx(expected, rel=1e-6)


class TestRunPodPredictionCostEstimation:
    """Test RunPod serverless prediction cost estimation (Task 3.3)."""

    def test_estimate_runpod_prediction_cost_default(self, cost_tracker):
        """Estimate prediction cost with default GPU and time."""
        # Default: RTX A5000, 10 seconds
        cost = cost_tracker.estimate_runpod_prediction_cost(1000)

        per_second_rate = 0.29 / 3600
        expected = per_second_rate * 10
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_estimate_runpod_prediction_cost_custom_time(self, cost_tracker):
        """Estimate prediction cost with custom execution time."""
        # RTX A40, 30 seconds
        cost = cost_tracker.estimate_runpod_prediction_cost(
            num_rows=5000,
            gpu_type='NVIDIA RTX A40',
            estimated_time_seconds=30
        )

        per_second_rate = 0.39 / 3600
        expected = per_second_rate * 30
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_estimate_runpod_prediction_cost_fast_inference(self, cost_tracker):
        """Estimate cost for fast 2-second inference."""
        cost = cost_tracker.estimate_runpod_prediction_cost(
            num_rows=100,
            gpu_type='NVIDIA RTX A5000',
            estimated_time_seconds=2
        )

        per_second_rate = 0.29 / 3600
        expected = per_second_rate * 2
        assert cost == pytest.approx(expected, rel=1e-6)
        assert cost < 0.001, "Fast inference should cost less than $0.001"

    def test_estimate_runpod_prediction_cost_invalid_gpu_raises_error(self, cost_tracker):
        """Unknown serverless GPU type should raise CostTrackingError."""
        with pytest.raises(CostTrackingError) as exc_info:
            cost_tracker.estimate_runpod_prediction_cost(
                1000,
                gpu_type='NVIDIA H100 PCIe',  # Not in serverless pricing
                estimated_time_seconds=10
            )

        assert "Unknown serverless GPU type" in str(exc_info.value)


class TestRunPodStorageCostCalculation:
    """Test RunPod network volume storage cost calculation (Task 3.4)."""

    @pytest.fixture
    def mock_storage_manager(self):
        """Create mock RunPod storage manager."""
        mock_mgr = Mock()

        # Mock datasets: 2GB total
        mock_mgr.list_user_datasets.return_value = [
            {'name': 'data1.csv', 'size_bytes': 1 * 1024**3},  # 1GB
            {'name': 'data2.csv', 'size_bytes': 1 * 1024**3},  # 1GB
        ]

        # Mock models: 3GB total
        mock_mgr.list_user_models.return_value = [
            {'name': 'model1', 'size_bytes': 1.5 * 1024**3},  # 1.5GB
            {'name': 'model2', 'size_bytes': 1.5 * 1024**3},  # 1.5GB
        ]

        return mock_mgr

    def test_calculate_runpod_storage_cost_stopped_volume(self, cost_tracker, mock_storage_manager):
        """Calculate storage cost for stopped volume."""
        result = cost_tracker.calculate_runpod_storage_cost(
            user_id=12345,
            storage_manager=mock_storage_manager,
            is_running=False
        )

        # Total: 5GB (2GB datasets + 3GB models)
        assert result['total_storage_gb'] == pytest.approx(5.0, rel=1e-2)
        assert result['datasets_gb'] == pytest.approx(2.0, rel=1e-2)
        assert result['models_gb'] == pytest.approx(3.0, rel=1e-2)

        # Stopped: $0.07/GB/month * 5GB = $0.35/month
        assert result['monthly_cost'] == pytest.approx(0.35, rel=1e-6)
        assert result['price_per_gb'] == 0.07

    def test_calculate_runpod_storage_cost_running_volume(self, cost_tracker, mock_storage_manager):
        """Calculate storage cost for running volume (higher rate)."""
        result = cost_tracker.calculate_runpod_storage_cost(
            user_id=12345,
            storage_manager=mock_storage_manager,
            is_running=True
        )

        # Running: $0.10/GB/month * 5GB = $0.50/month
        assert result['monthly_cost'] == pytest.approx(0.50, rel=1e-6)
        assert result['price_per_gb'] == 0.10

    def test_calculate_runpod_storage_cost_empty_storage(self, cost_tracker):
        """Calculate cost for user with no data."""
        mock_mgr = Mock()
        mock_mgr.list_user_datasets.return_value = []
        mock_mgr.list_user_models.return_value = []

        result = cost_tracker.calculate_runpod_storage_cost(
            user_id=99999,
            storage_manager=mock_mgr
        )

        assert result['total_storage_gb'] == 0.0
        assert result['monthly_cost'] == 0.0


class TestRunPodCostComparison:
    """Test cost comparisons between AWS and RunPod."""

    def test_runpod_gpu_cheaper_than_aws_p3(self, cost_tracker):
        """RunPod GPUs should be cheaper than AWS P3 instances."""
        # AWS P3.2xlarge (1 V100): $1.20/hour
        # RunPod A100 (better GPU): $0.79/hour

        # 1 hour training
        aws_cost = cost_tracker.estimate_training_cost('p3.2xlarge', 60)
        runpod_cost = cost_tracker.estimate_runpod_training_cost('NVIDIA A100 PCIe 40GB', 3600)

        assert runpod_cost < aws_cost, "RunPod A100 should be cheaper than AWS P3"

    def test_runpod_per_second_billing_advantage(self, cost_tracker):
        """RunPod per-second billing provides precise cost calculation."""
        # Both estimate based on actual runtime, but RunPod bills per-second
        # while AWS bills per-hour minimum in practice

        # 30-minute job comparison
        aws_cost_30min = cost_tracker.estimate_training_cost('m5.xlarge', 30)
        runpod_cost_30min = cost_tracker.estimate_runpod_training_cost('NVIDIA RTX A5000', 1800)

        # Verify calculations are precise for both
        assert aws_cost_30min == pytest.approx(0.10 * 0.5, rel=1e-6)  # 0.5 hours
        assert runpod_cost_30min == pytest.approx(0.29 * 0.5, rel=1e-6)  # 0.5 hours

        # RunPod A5000 is more expensive per hour ($0.29 vs $0.10) but faster
        # The advantage is GPU performance, not hourly cost


class TestRunPodCostLogging:
    """Test cost logging works for RunPod services (Task 3.5)."""

    def test_log_cost_accepts_runpod_service_names(self, cost_tracker, tmp_path):
        """Cost logger should accept RunPod service names."""
        # Set log path to temp directory
        cost_tracker._log_path = tmp_path / "costs.json"

        # Log RunPod GPU training cost
        cost_tracker._log_cost(
            service='runpod_gpu',
            operation='training',
            user_id=12345,
            cost=0.145,
            gpu_type='NVIDIA RTX A5000',
            duration_seconds=1800
        )

        # Verify log file was created
        assert cost_tracker._log_path.exists()

        # Read log entry
        with open(cost_tracker._log_path) as f:
            import json
            entry = json.loads(f.read().strip())

        assert entry['service'] == 'runpod_gpu'
        assert entry['operation'] == 'training'
        assert entry['gpu_type'] == 'NVIDIA RTX A5000'
        assert entry['cost'] == 0.145

    def test_log_cost_runpod_serverless(self, cost_tracker, tmp_path):
        """Cost logger should work for RunPod serverless."""
        cost_tracker._log_path = tmp_path / "costs.json"

        cost_tracker._log_cost(
            service='runpod_serverless',
            operation='prediction',
            user_id=12345,
            cost=0.0008,
            gpu_type='NVIDIA RTX A5000',
            num_predictions=1000
        )

        with open(cost_tracker._log_path) as f:
            import json
            entry = json.loads(f.read().strip())

        assert entry['service'] == 'runpod_serverless'
        assert entry['operation'] == 'prediction'
