"""Tests for local_path_messages.py i18n integration."""

import pytest
from src.bot.messages.local_path_messages import LocalPathMessages
from src.utils.i18n_manager import I18nManager


class TestLocalPathMessagesI18n:
    """Test i18n integration for LocalPathMessages."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize i18n before each test."""
        I18nManager.initialize("./locales", "en")

    # =========================================================================
    # Data Source Selection
    # =========================================================================

    def test_data_source_selection_prompt_english(self):
        """Test data source selection prompt in English."""
        result = LocalPathMessages.data_source_selection_prompt()
        assert "ML Training Workflow" in result
        assert "How would you like to provide your training data?" in result
        assert "Upload File" in result
        assert "Use Local Path" in result
        assert "Max size: 10MB" in result
        assert "No size limits" in result

    def test_data_source_selection_prompt_portuguese(self):
        """Test data source selection prompt in Portuguese."""
        I18nManager.set_locale("pt")
        result = LocalPathMessages.data_source_selection_prompt(locale="pt")
        assert "Treinamento de ML" in result
        assert "Como você gostaria de fornecer seus dados de treinamento?" in result

    # =========================================================================
    # Telegram Upload
    # =========================================================================

    def test_telegram_upload_prompt_english(self):
        """Test Telegram upload prompt in English."""
        result = LocalPathMessages.telegram_upload_prompt()
        assert "Telegram Upload" in result
        assert "Please upload your file" in result
        assert "CSV, Excel, Parquet" in result
        assert "10MB" in result

    def test_telegram_upload_prompt_portuguese(self):
        """Test Telegram upload prompt in Portuguese."""
        I18nManager.set_locale("pt")
        result = LocalPathMessages.telegram_upload_prompt(locale="pt")
        assert "Upload via Telegram" in result
        assert "Por favor, envie seu arquivo" in result

    # =========================================================================
    # Local Path Input
    # =========================================================================

    def test_file_path_input_prompt_english(self):
        """Test file path input prompt in English."""
        allowed_dirs = ["/home/user/data", "/Users/username/datasets"]
        result = LocalPathMessages.file_path_input_prompt(allowed_dirs)
        assert "Local File Path" in result
        assert "Allowed directories" in result
        assert "/home/user/data" in result
        assert "CSV, Excel" in result
        assert "Type or paste your file path" in result

    def test_file_path_input_prompt_portuguese(self):
        """Test file path input prompt in Portuguese."""
        I18nManager.set_locale("pt")
        allowed_dirs = ["/home/user/data"]
        result = LocalPathMessages.file_path_input_prompt(allowed_dirs, locale="pt")
        assert "Caminho do Arquivo Local" in result
        assert "Diretórios permitidos" in result

    def test_file_path_input_prompt_truncates_long_list(self):
        """Test that allowed directories list truncates after 5."""
        allowed_dirs = [f"/dir{i}" for i in range(10)]
        result = LocalPathMessages.file_path_input_prompt(allowed_dirs)
        assert "... and 5 more" in result

    # =========================================================================
    # Loading Messages
    # =========================================================================

    def test_loading_data_message_english(self):
        """Test loading data message in English."""
        result = LocalPathMessages.loading_data_message()
        assert "Loading data" in result
        assert "Validating, loading, and analyzing schema" in result

    def test_loading_data_message_portuguese(self):
        """Test loading data message in Portuguese."""
        I18nManager.set_locale("pt")
        result = LocalPathMessages.loading_data_message(locale="pt")
        assert "Carregando dados" in result

    # =========================================================================
    # Schema Confirmation
    # =========================================================================

    def test_schema_confirmation_prompt_english(self):
        """Test schema confirmation prompt in English."""
        summary = "Dataset: housing.csv\nRows: 100 | Columns: 5"
        result = LocalPathMessages.schema_confirmation_prompt(
            summary=summary,
            suggested_target="price",
            suggested_features=["sqft", "bedrooms", "bathrooms"],
            task_type="regression"
        )
        assert "Dataset: housing.csv" in result
        assert "Auto-Detected:**" in result  # Format includes emoji
        assert "Regression" in result
        assert "Target:** `price`" in result
        assert "Features:** `sqft`, `bedrooms`, `bathrooms`" in result
        assert "Proceed?" in result

    def test_schema_confirmation_prompt_portuguese(self):
        """Test schema confirmation prompt in Portuguese."""
        I18nManager.set_locale("pt")
        summary = "Dataset: housing.csv\nRows: 100 | Columns: 5"
        result = LocalPathMessages.schema_confirmation_prompt(
            summary=summary,
            suggested_target="price",
            suggested_features=["sqft", "bedrooms"],
            task_type="regression",
            locale="pt"
        )
        assert "Detectado Automaticamente" in result
        assert "Alvo:" in result

    def test_schema_confirmation_prompt_truncates_features(self):
        """Test that feature list truncates after 5."""
        features = [f"feature_{i}" for i in range(10)]
        result = LocalPathMessages.schema_confirmation_prompt(
            summary="Test",
            suggested_target="target",
            suggested_features=features,
            task_type="classification"
        )
        assert "(+5)" in result or "... (+5)" in result

    def test_schema_confirmation_prompt_no_suggestions(self):
        """Test schema prompt without auto-detected suggestions."""
        result = LocalPathMessages.schema_confirmation_prompt(
            summary="Dataset loaded",
            suggested_target=None,
            suggested_features=[],
            task_type=None
        )
        assert "Proceed?" in result
        assert "Auto-Detected" not in result

    # =========================================================================
    # Schema Acceptance/Rejection
    # =========================================================================

    def test_schema_accepted_message_with_target_english(self):
        """Test schema accepted message with target in English."""
        result = LocalPathMessages.schema_accepted_message("price")
        assert "Schema Accepted" in result
        assert "Using: `price`" in result
        assert "Proceeding to target selection" in result

    def test_schema_accepted_message_with_target_portuguese(self):
        """Test schema accepted message with target in Portuguese."""
        I18nManager.set_locale("pt")
        result = LocalPathMessages.schema_accepted_message("price", locale="pt")
        assert "Esquema Aceito" in result

    def test_schema_accepted_message_without_target(self):
        """Test schema accepted message without target."""
        result = LocalPathMessages.schema_accepted_message(None)
        assert "Schema Accepted" in result
        assert "Proceeding to target selection" in result

    def test_schema_rejected_message_english(self):
        """Test schema rejected message in English."""
        result = LocalPathMessages.schema_rejected_message()
        assert "Schema Rejected" in result
        assert "Please provide a different file path" in result

    def test_schema_rejected_message_portuguese(self):
        """Test schema rejected message in Portuguese."""
        I18nManager.set_locale("pt")
        result = LocalPathMessages.schema_rejected_message(locale="pt")
        assert "Esquema Rejeitado" in result

    # =========================================================================
    # Load Strategy Selection
    # =========================================================================

    def test_load_option_prompt_english(self):
        """Test load strategy prompt in English."""
        result = LocalPathMessages.load_option_prompt("/path/to/file.csv", 50.5)
        assert "Path Validated" in result
        assert "50.50 MB" in result
        assert "Choose Loading Strategy" in result
        assert "Load Now" in result
        assert "Defer Loading" in result

    def test_load_option_prompt_portuguese(self):
        """Test load strategy prompt in Portuguese."""
        I18nManager.set_locale("pt")
        result = LocalPathMessages.load_option_prompt("/path/to/file.csv", 50.5, locale="pt")
        assert "Caminho Validado" in result
        assert "Escolha a Estratégia de Carregamento" in result

    # =========================================================================
    # Manual Schema Input
    # =========================================================================

    def test_schema_input_prompt_english(self):
        """Test manual schema input prompt in English."""
        result = LocalPathMessages.schema_input_prompt()
        assert "Manual Schema Input" in result
        assert "3 formats supported" in result
        assert "Format 1 - Key-Value" in result
        assert "Format 2 - JSON" in result
        assert "Format 3 - Simple List" in result

    def test_schema_input_prompt_portuguese(self):
        """Test manual schema input prompt in Portuguese."""
        I18nManager.set_locale("pt")
        result = LocalPathMessages.schema_input_prompt(locale="pt")
        assert "Entrada Manual de Esquema" in result

    def test_schema_accepted_deferred_english(self):
        """Test deferred schema acceptance in English."""
        result = LocalPathMessages.schema_accepted_deferred("price", 3)
        assert "Schema Accepted" in result
        assert "Target: `price`" in result
        assert "Features: 3 features" in result
        assert "Data will load at training time" in result

    def test_schema_accepted_deferred_portuguese(self):
        """Test deferred schema acceptance in Portuguese."""
        I18nManager.set_locale("pt")
        result = LocalPathMessages.schema_accepted_deferred("price", 3, locale="pt")
        assert "Esquema Aceito" in result

    def test_schema_accepted_deferred_singular(self):
        """Test deferred schema with 1 feature uses singular."""
        result = LocalPathMessages.schema_accepted_deferred("target", 1)
        assert "1 feature" in result

    def test_schema_parse_error_english(self):
        """Test schema parse error in English."""
        result = LocalPathMessages.schema_parse_error("Invalid JSON format")
        assert "Schema Parse Error" in result
        assert "Invalid JSON format" in result
        assert "Please try again" in result

    def test_schema_parse_error_portuguese(self):
        """Test schema parse error in Portuguese."""
        I18nManager.set_locale("pt")
        result = LocalPathMessages.schema_parse_error("Formato JSON inválido", locale="pt")
        assert "Erro de Análise de Esquema" in result

    # =========================================================================
    # Error Messages
    # =========================================================================

    def test_format_path_error_not_found(self):
        """Test file not found error message."""
        result = LocalPathMessages.format_path_error("not_found", "/path/to/missing.csv")
        assert "File Not Found" in result
        assert "/path/to/missing.csv" in result
        assert "Check: Path correct?" in result

    def test_format_path_error_not_in_whitelist(self):
        """Test whitelist violation error message."""
        allowed = ["/home/user", "/data"]
        result = LocalPathMessages.format_path_error(
            "not_in_whitelist",
            "/forbidden/path.csv",
            allowed_dirs=allowed
        )
        assert "Access Denied" in result
        assert "/home/user" in result

    def test_format_path_error_path_traversal(self):
        """Test path traversal security error."""
        result = LocalPathMessages.format_path_error("path_traversal", "../../../etc/passwd")
        assert "suspicious patterns" in result
        assert "../../../etc/passwd" in result

    def test_format_path_error_too_large(self):
        """Test file too large error."""
        result = LocalPathMessages.format_path_error(
            "too_large",
            "/path/to/huge.csv",
            size_mb=2000.5,
            max_size_mb=1000
        )
        assert "File Too Large" in result
        assert "2000.5MB" in result
        assert "1000MB" in result

    def test_format_path_error_invalid_extension(self):
        """Test invalid file extension error."""
        result = LocalPathMessages.format_path_error(
            "invalid_extension",
            "/path/to/file.txt",
            allowed_extensions=[".csv", ".xlsx"]
        )
        assert "Unsupported Format" in result
        assert ".csv, .xlsx" in result

    def test_format_path_error_empty(self):
        """Test empty file error."""
        result = LocalPathMessages.format_path_error("empty", "/path/to/empty.csv")
        assert "0 bytes" in result
        assert "/path/to/empty.csv" in result

    def test_format_path_error_loading_error(self):
        """Test data loading error."""
        result = LocalPathMessages.format_path_error(
            "loading_error",
            "/path/to/corrupt.csv",
            error_details="Parsing error at line 10"
        )
        assert "Loading Error" in result
        assert "Parsing error at line 10" in result

    def test_format_path_error_feature_disabled(self):
        """Test feature disabled error."""
        result = LocalPathMessages.format_path_error("feature_disabled", "/any/path.csv")
        assert "Feature Disabled" in result
        assert "Use Telegram upload" in result

    def test_format_path_error_unexpected(self):
        """Test unexpected error fallback."""
        result = LocalPathMessages.format_path_error("unknown_error_type", "/path/to/file.csv")
        assert "Unexpected Error" in result

    # =========================================================================
    # XGBoost Messages
    # =========================================================================

    def test_xgboost_description_english(self):
        """Test XGBoost description in English."""
        result = LocalPathMessages.xgboost_description()
        assert "XGBoost (Gradient Boosting)" in result
        assert "Best for:" in result
        assert "Advantages:" in result
        assert "When to use:" in result

    def test_xgboost_description_portuguese(self):
        """Test XGBoost description in Portuguese."""
        I18nManager.set_locale("pt")
        result = LocalPathMessages.xgboost_description(locale="pt")
        assert "XGBoost" in result

    def test_xgboost_hyperparameter_help_english(self):
        """Test XGBoost hyperparameter help in English."""
        result = LocalPathMessages.xgboost_hyperparameter_help()
        assert "XGBoost Hyperparameters" in result
        assert "n_estimators" in result
        assert "max_depth" in result
        assert "learning_rate" in result

    def test_xgboost_setup_required_english(self):
        """Test XGBoost setup required message in English."""
        result = LocalPathMessages.xgboost_setup_required()
        assert "XGBoost Setup Required" in result
        assert "brew install libomp" in result
        assert "XGBOOST_SETUP.md" in result

    # =========================================================================
    # Password Protection Messages
    # =========================================================================

    def test_password_prompt_english(self):
        """Test password prompt in English."""
        result = LocalPathMessages.password_prompt(
            "/outside/whitelist/file.csv",
            "/outside/whitelist"
        )
        assert "Password Required" in result
        assert "/outside/whitelist/file.csv" in result
        assert "Security Note" in result

    def test_password_prompt_portuguese(self):
        """Test password prompt in Portuguese."""
        I18nManager.set_locale("pt")
        result = LocalPathMessages.password_prompt(
            "/outside/whitelist/file.csv",
            "/outside/whitelist",
            locale="pt"
        )
        assert "Senha Necessária" in result

    def test_password_success_english(self):
        """Test password success message in English."""
        result = LocalPathMessages.password_success("/granted/directory")
        assert "Access Granted" in result
        assert "/granted/directory" in result

    def test_password_success_portuguese(self):
        """Test password success message in Portuguese."""
        I18nManager.set_locale("pt")
        result = LocalPathMessages.password_success("/granted/directory", locale="pt")
        assert "Acesso Concedido" in result

    def test_password_failure_english(self):
        """Test password failure message in English."""
        result = LocalPathMessages.password_failure("Incorrect password")
        assert "Incorrect password" in result

    def test_password_lockout_english(self):
        """Test password lockout message in English."""
        result = LocalPathMessages.password_lockout(300)
        assert "Access Locked" in result
        assert "300 seconds" in result

    def test_password_lockout_portuguese(self):
        """Test password lockout message in Portuguese."""
        I18nManager.set_locale("pt")
        result = LocalPathMessages.password_lockout(300, locale="pt")
        assert "Acesso Bloqueado" in result

    def test_password_timeout_english(self):
        """Test password timeout message in English."""
        result = LocalPathMessages.password_timeout()
        assert "Session Timeout" in result
        assert "5 minute limit" in result

    def test_password_timeout_portuguese(self):
        """Test password timeout message in Portuguese."""
        I18nManager.set_locale("pt")
        result = LocalPathMessages.password_timeout(locale="pt")
        assert "Tempo Esgotado" in result
