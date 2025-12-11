"""
Tests for critical security fixes.

These tests verify:
1. No hardcoded passwords in code
2. Resource limits are enforced
3. Model files are signed with HMAC
"""

import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


class TestPasswordSecurity:
    """Tests for Issue 1: Hardcoded Password removal."""

    def test_password_requires_env_var(self):
        """Password validator raises error without FILE_PATH_PASSWORD env var."""
        # Clear the environment variable
        env_backup = os.environ.pop('FILE_PATH_PASSWORD', None)

        try:
            # Import fresh to test initialization
            import importlib
            import src.utils.password_validator as pv_module
            importlib.reload(pv_module)

            # Should raise error when no env var and no default
            with pytest.raises(ValueError, match="FILE_PATH_PASSWORD.*required"):
                pv_module.PasswordValidator()
        finally:
            # Restore environment
            if env_backup:
                os.environ['FILE_PATH_PASSWORD'] = env_backup

    def test_no_default_password_constant(self):
        """No DEFAULT_PASSWORD constant exists in password_validator.py."""
        from src.utils import password_validator

        # Check that DEFAULT_PASSWORD doesn't exist or is None
        assert not hasattr(password_validator.PasswordValidator, 'DEFAULT_PASSWORD') or \
               password_validator.PasswordValidator.DEFAULT_PASSWORD is None, \
               "DEFAULT_PASSWORD constant should be removed"

    def test_no_hardcoded_senha123_in_code(self):
        """No hardcoded 'senha123' exists in Python source files."""
        src_dir = Path(__file__).parent.parent.parent / "src"

        hardcoded_found = []
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text()
            # Check for hardcoded password
            if 'senha123' in content.lower():
                # Exclude test files
                if 'test_' not in py_file.name:
                    hardcoded_found.append(str(py_file))

        assert not hardcoded_found, \
            f"Found hardcoded password in: {hardcoded_found}"

    def test_no_password_in_config_yaml(self):
        """No password value in config.yaml."""
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        content = config_path.read_text()

        # Should not have senha123 anywhere
        assert 'senha123' not in content.lower(), \
            "Found hardcoded password 'senha123' in config.yaml"

        # Password line should be removed or only reference env var
        lines = content.split('\n')
        for line in lines:
            if 'password:' in line.lower() and 'file_path_password' not in line.lower():
                # Skip if it's a comment about password
                if not line.strip().startswith('#'):
                    assert False, f"Found password configuration that's not env var reference: {line}"


class TestResourceLimits:
    """Tests for Issue 2: Resource limits must be enabled."""

    def test_resource_limits_code_not_commented(self):
        """Resource limits code should not be commented out in executor.py."""
        executor_path = Path(__file__).parent.parent.parent / "src" / "execution" / "executor.py"
        content = executor_path.read_text()

        # Check that sandbox is conditionally applied (using config.sandbox_enabled)
        assert 'config.sandbox_enabled' in content, \
               "Sandbox should be controlled via config.sandbox_enabled"

        # Check that create_sandbox_env is called when enabled
        assert 'create_sandbox_env()' in content, \
               "Sandbox environment should be used"

        # Check that set_resource_limits is called when enabled
        assert 'set_resource_limits' in content and 'preexec_fn' in content, \
               "Resource limits should be applied via preexec_fn"

    def test_sandbox_enabled_by_default(self):
        """Sandbox should be enabled by default in config."""
        from src.execution.config import SandboxConfig

        config = SandboxConfig()

        # Sandbox should be enabled by default
        assert hasattr(config, 'sandbox_enabled'), \
            "SandboxConfig should have sandbox_enabled attribute"
        assert config.sandbox_enabled is True, \
            "sandbox_enabled should be True by default"

    def test_memory_limit_default(self):
        """Memory limit should default to 2048 MB."""
        from src.execution.config import SandboxConfig

        config = SandboxConfig()
        assert config.memory_limit == 2048, \
            f"Memory limit should default to 2048, got {config.memory_limit}"

    def test_timeout_default(self):
        """Timeout should default to 30 seconds."""
        from src.execution.config import SandboxConfig

        config = SandboxConfig()
        assert config.timeout == 30, \
            f"Timeout should default to 30, got {config.timeout}"


class TestModelSigning:
    """Tests for Issue 3: Model signing with HMAC."""

    def test_model_signing_module_exists(self):
        """Model signing utility module should exist."""
        try:
            from src.utils import model_signing
            assert hasattr(model_signing, 'sign_file')
            assert hasattr(model_signing, 'verify_file')
        except ImportError:
            pytest.fail("src/utils/model_signing.py module should exist")

    def test_signing_requires_key_env_var(self):
        """Model signing should require MODEL_SIGNING_KEY env var."""
        # Clear the environment variable
        env_backup = os.environ.pop('MODEL_SIGNING_KEY', None)

        try:
            from src.utils.model_signing import get_signing_key

            with pytest.raises(ValueError, match="MODEL_SIGNING_KEY.*required"):
                get_signing_key()
        except ImportError:
            pytest.fail("model_signing module should exist with get_signing_key function")
        finally:
            if env_backup:
                os.environ['MODEL_SIGNING_KEY'] = env_backup

    def test_sign_and_verify_file(self):
        """sign_file creates signature, verify_file validates it."""
        # Set test key
        os.environ['MODEL_SIGNING_KEY'] = 'test_key_for_signing_12345'

        try:
            from src.utils.model_signing import sign_file, verify_file

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
                f.write(b"test model data")
                test_file = Path(f.name)

            try:
                # Sign the file
                sign_file(test_file)

                # Signature file should exist
                sig_file = test_file.with_suffix(test_file.suffix + '.sig')
                assert sig_file.exists(), "Signature file should be created"

                # Verification should pass
                assert verify_file(test_file) is True, "Verification should pass for valid signature"
            finally:
                test_file.unlink(missing_ok=True)
                sig_file.unlink(missing_ok=True)
        finally:
            os.environ.pop('MODEL_SIGNING_KEY', None)

    def test_verify_rejects_unsigned_file(self):
        """verify_file should reject files without signature."""
        os.environ['MODEL_SIGNING_KEY'] = 'test_key_for_signing_12345'

        try:
            from src.utils.model_signing import verify_file
            from src.utils.exceptions import SecurityViolationError

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
                f.write(b"test model data")
                test_file = Path(f.name)

            try:
                # Verification should fail for unsigned file
                with pytest.raises(SecurityViolationError) as exc_info:
                    verify_file(test_file)
                assert "unsigned" in str(exc_info.value).lower() or "signature" in str(exc_info.value).lower()
            finally:
                test_file.unlink(missing_ok=True)
        finally:
            os.environ.pop('MODEL_SIGNING_KEY', None)

    def test_verify_rejects_tampered_file(self):
        """verify_file should reject files with invalid signature."""
        os.environ['MODEL_SIGNING_KEY'] = 'test_key_for_signing_12345'

        try:
            from src.utils.model_signing import sign_file, verify_file
            from src.utils.exceptions import SecurityViolationError

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
                f.write(b"test model data")
                test_file = Path(f.name)

            try:
                # Sign the file
                sign_file(test_file)

                # Tamper with the file
                with open(test_file, 'wb') as f:
                    f.write(b"tampered model data")

                # Verification should fail
                with pytest.raises(SecurityViolationError) as exc_info:
                    verify_file(test_file)
                error_msg = str(exc_info.value).lower()
                assert "invalid" in error_msg or "tampered" in error_msg or "signature" in error_msg
            finally:
                test_file.unlink(missing_ok=True)
                sig_file = test_file.with_suffix(test_file.suffix + '.sig')
                sig_file.unlink(missing_ok=True)
        finally:
            os.environ.pop('MODEL_SIGNING_KEY', None)

    def test_model_manager_signs_on_save(self):
        """ModelManager.save_model should sign model files."""
        os.environ['MODEL_SIGNING_KEY'] = 'test_key_for_signing_12345'

        try:
            from src.engines.model_manager import ModelManager
            from src.engines.ml_config import MLEngineConfig
            from sklearn.linear_model import LinearRegression

            with tempfile.TemporaryDirectory() as temp_dir:
                config = MLEngineConfig.get_default()
                config.models_dir = temp_dir
                manager = ModelManager(config)

                # Create a real sklearn model (MagicMock can't be pickled)
                model = LinearRegression()

                # Save model
                manager.save_model(
                    user_id=12345,
                    model_id="test_model",
                    model=model,
                    metadata={"test": "data"}
                )

                # Check signature file exists
                model_dir = Path(temp_dir) / "user_12345" / "test_model"
                model_pkl = model_dir / "model.pkl"
                sig_file = model_pkl.with_suffix('.pkl.sig')

                assert sig_file.exists(), "Model signature file should be created on save"
        finally:
            os.environ.pop('MODEL_SIGNING_KEY', None)

    def test_model_manager_verifies_on_load(self):
        """ModelManager.load_model should verify signature before loading."""
        os.environ['MODEL_SIGNING_KEY'] = 'test_key_for_signing_12345'

        try:
            from src.engines.model_manager import ModelManager
            from src.engines.ml_config import MLEngineConfig
            from src.utils.exceptions import SecurityViolationError
            import joblib

            with tempfile.TemporaryDirectory() as temp_dir:
                config = MLEngineConfig.get_default()
                config.models_dir = temp_dir
                manager = ModelManager(config)

                # Create model directory manually (simulating external/old model)
                model_dir = Path(temp_dir) / "user_12345" / "unsigned_model"
                model_dir.mkdir(parents=True)

                # Save model without signature
                model_path = model_dir / "model.pkl"
                joblib.dump({"test": "model"}, model_path)

                # Save metadata
                import json
                metadata_path = model_dir / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump({
                        "model_id": "unsigned_model",
                        "user_id": 12345,
                        "model_format": "sklearn"
                    }, f)

                # Loading should fail due to missing signature
                with pytest.raises(SecurityViolationError) as exc_info:
                    manager.load_model(user_id=12345, model_id="unsigned_model")
                error_msg = str(exc_info.value).lower()
                assert "unsigned" in error_msg or "signature" in error_msg
        finally:
            os.environ.pop('MODEL_SIGNING_KEY', None)


class TestNoRegressions:
    """Verify existing functionality still works after security fixes."""

    def test_password_validator_with_env_var_works(self):
        """PasswordValidator works when FILE_PATH_PASSWORD is set."""
        os.environ['FILE_PATH_PASSWORD'] = 'test_password_123'

        try:
            import importlib
            import src.utils.password_validator as pv_module
            importlib.reload(pv_module)

            validator = pv_module.PasswordValidator()

            # Valid password should work
            is_valid, error = validator.validate_password(
                user_id=12345,
                password_input='test_password_123',
                path='/some/path'
            )
            assert is_valid is True
            assert error is None

            # Invalid password should fail
            is_valid, error = validator.validate_password(
                user_id=12346,
                password_input='wrong_password',
                path='/some/path'
            )
            assert is_valid is False
            assert error is not None
        finally:
            os.environ.pop('FILE_PATH_PASSWORD', None)

    def test_sandbox_config_creation(self):
        """SandboxConfig can still be created with custom values."""
        from src.execution.config import SandboxConfig

        config = SandboxConfig(
            timeout=60,
            memory_limit=4096,
            cpu_limit=120,
            allow_network=True
        )

        assert config.timeout == 60
        assert config.memory_limit == 4096
        assert config.cpu_limit == 120
        assert config.allow_network is True
