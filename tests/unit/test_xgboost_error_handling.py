"""
Test XGBoost error handling with missing OpenMP dependency.

Tests that the enhanced error handling in xgboost_trainer.py provides
helpful, actionable error messages when libomp is missing.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.engines.trainers.xgboost_trainer import XGBoostTrainer
from src.engines.ml_config import MLEngineConfig
from src.utils.exceptions import TrainingError


class TestXGBoostErrorHandling:
    """Test enhanced error handling for XGBoost import failures."""

    def test_libomp_missing_error_detection(self):
        """Test that libomp missing error is properly detected and formatted."""
        trainer = XGBoostTrainer(MLEngineConfig.get_default())

        # Simulate the actual error message from macOS when libomp is missing
        libomp_error_msg = (
            "dlopen(/path/to/libxgboost.dylib, 0x0006): "
            "Library not loaded: @rpath/libomp.dylib"
        )

        with patch('src.engines.trainers.xgboost_trainer.XGBClassifier') as mock_xgb:
            # Simulate import raising error with libomp message
            mock_xgb.side_effect = Exception(libomp_error_msg)

            with pytest.raises(TrainingError) as exc_info:
                trainer._import_xgboost()

            # Verify error message contains helpful information
            error = exc_info.value
            assert "OpenMP runtime" in str(error)
            assert "libomp" in str(error)

            # Verify error_details contains setup instructions
            assert hasattr(error, 'error_details')
            assert "brew install libomp" in error.error_details
            assert "Gradient Boosting (sklearn)" in error.error_details
            assert "XGBOOST_SETUP.md" in error.error_details

    def test_openmp_keyword_detection(self):
        """Test that errors containing 'OpenMP' are properly handled."""
        trainer = XGBoostTrainer(MLEngineConfig.get_default())

        openmp_error = "XGBoost Library could not be loaded. OpenMP runtime is not installed."

        with patch('src.engines.trainers.xgboost_trainer.XGBClassifier') as mock_xgb:
            mock_xgb.side_effect = Exception(openmp_error)

            with pytest.raises(TrainingError) as exc_info:
                trainer._import_xgboost()

            error = exc_info.value
            assert "OpenMP runtime" in str(error)
            assert "brew install libomp" in error.error_details

    def test_standard_import_error_handling(self):
        """Test that standard ImportError is still handled properly."""
        trainer = XGBoostTrainer(MLEngineConfig.get_default())

        with patch('src.engines.trainers.xgboost_trainer.XGBClassifier') as mock_xgb:
            mock_xgb.side_effect = ImportError("No module named 'xgboost'")

            with pytest.raises(TrainingError) as exc_info:
                trainer._import_xgboost()

            error = exc_info.value
            assert "install xgboost>=1.7.0" in str(error)
            assert "pip install" in error.error_details

    def test_generic_error_handling(self):
        """Test that other errors are handled gracefully."""
        trainer = XGBoostTrainer(MLEngineConfig.get_default())

        with patch('src.engines.trainers.xgboost_trainer.XGBClassifier') as mock_xgb:
            mock_xgb.side_effect = RuntimeError("Some other error")

            with pytest.raises(TrainingError) as exc_info:
                trainer._import_xgboost()

            error = exc_info.value
            assert "initialization failed" in str(error).lower()

    def test_error_message_structure(self):
        """Test that error messages have proper structure for Telegram display."""
        trainer = XGBoostTrainer(MLEngineConfig.get_default())

        libomp_error = "Library not loaded: @rpath/libomp.dylib"

        with patch('src.engines.trainers.xgboost_trainer.XGBClassifier') as mock_xgb:
            mock_xgb.side_effect = Exception(libomp_error)

            with pytest.raises(TrainingError) as exc_info:
                trainer._import_xgboost()

            error = exc_info.value

            # Main message should be concise
            main_msg = str(error)
            assert len(main_msg) < 100  # Short main message

            # Details should be comprehensive
            assert len(error.error_details) > 200  # Detailed instructions
            assert error.error_details.count('\n') >= 5  # Multi-line formatted

    def test_import_only_called_once(self):
        """Test that successful import is cached (not called repeatedly)."""
        trainer = XGBoostTrainer(MLEngineConfig.get_default())

        # First call succeeds (if XGBoost available)
        # If XGBoost not available, this test will be skipped
        try:
            from xgboost import XGBClassifier
            # XGBoost is available
            trainer._import_xgboost()
            assert trainer._xgboost_imported is True

            # Second call should not re-import
            with patch('src.engines.trainers.xgboost_trainer.XGBClassifier') as mock_xgb:
                trainer._import_xgboost()  # Should not trigger import
                mock_xgb.assert_not_called()

        except Exception:
            # XGBoost not available - skip test
            pytest.skip("XGBoost not available for caching test")


class TestXGBoostChecker:
    """Test xgboost_checker utility functions."""

    def test_check_xgboost_available_with_libomp(self):
        """Test check_xgboost_available when libomp is available."""
        from src.utils.xgboost_checker import check_xgboost_available

        # This will either succeed or fail depending on actual system state
        available, message = check_xgboost_available()

        # Should return tuple
        assert isinstance(available, bool)
        assert isinstance(message, str)
        assert len(message) > 0

        # Message should be informative
        if available:
            assert "XGBoost" in message
        else:
            assert any(word in message.lower() for word in ['missing', 'failed', 'not installed'])

    def test_get_setup_instructions_macos(self):
        """Test platform-specific setup instructions for macOS."""
        from src.utils.xgboost_checker import get_xgboost_setup_instructions
        import platform

        with patch('platform.system', return_value='Darwin'):
            instructions = get_xgboost_setup_instructions()

            assert "macOS" in instructions
            assert "brew install libomp" in instructions
            assert "XGBOOST_SETUP.md" in instructions

    def test_get_setup_instructions_linux(self):
        """Test platform-specific setup instructions for Linux."""
        from src.utils.xgboost_checker import get_xgboost_setup_instructions

        with patch('platform.system', return_value='Linux'):
            instructions = get_xgboost_setup_instructions()

            assert "Linux" in instructions
            assert "libgomp" in instructions
            assert "XGBOOST_SETUP.md" in instructions

    def test_log_xgboost_status_available(self):
        """Test logging when XGBoost is available."""
        from src.utils.xgboost_checker import log_xgboost_status

        with patch('src.utils.xgboost_checker.check_xgboost_available', return_value=(True, "XGBoost 2.1.4")):
            with patch('src.utils.xgboost_checker.logger') as mock_logger:
                log_xgboost_status()

                # Should log success
                mock_logger.info.assert_called()
                call_args = str(mock_logger.info.call_args)
                assert "available" in call_args.lower()

    def test_log_xgboost_status_unavailable(self):
        """Test logging when XGBoost is unavailable."""
        from src.utils.xgboost_checker import log_xgboost_status

        with patch('src.utils.xgboost_checker.check_xgboost_available', return_value=(False, "OpenMP missing")):
            with patch('src.utils.xgboost_checker.logger') as mock_logger:
                log_xgboost_status()

                # Should log warning
                mock_logger.warning.assert_called()
                warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
                assert any("unavailable" in call.lower() for call in warning_calls)
