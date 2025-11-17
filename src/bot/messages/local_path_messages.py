"""User-facing messages for local path workflow."""

from typing import List, Optional
from telegram import InlineKeyboardButton
from src.utils.i18n_manager import I18nManager


class LocalPathMessages:
    """Consolidated message templates for local path workflow."""

    @staticmethod
    def data_source_selection_prompt(locale: Optional[str] = None) -> str:
        return I18nManager.t('workflows.ml_training.data_source_prompt', locale=locale)

    @staticmethod
    def file_path_input_prompt(allowed_dirs: List[str], locale: Optional[str] = None) -> str:
        dirs = "\n".join(f"• `{d}`" for d in allowed_dirs[:5])
        if len(allowed_dirs) > 5:
            dirs += f"\n• {I18nManager.t('common.and_more', locale=locale, count=len(allowed_dirs) - 5)}"

        return (
            f"{I18nManager.t('workflows.ml_training.local_file_path_header', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.allowed_directories', locale=locale)}\n{dirs}\n\n"
            f"{I18nManager.t('workflows.ml_training.formats_label', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.examples_label', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.example_path_1', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.example_path_2', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.type_path_prompt', locale=locale)}"
        )

    @staticmethod
    def loading_data_message(locale: Optional[str] = None) -> str:
        return (
            f"{I18nManager.t('workflows.ml_training.loading_data', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.loading_status', locale=locale)}"
        )

    @staticmethod
    def schema_confirmation_prompt(
        summary: str,
        suggested_target: Optional[str],
        suggested_features: List[str],
        task_type: Optional[str],
        locale: Optional[str] = None
    ) -> str:
        msg = f"{summary}\n\n"

        if task_type and suggested_target:
            msg += (
                f"{I18nManager.t('workflows.ml_training.auto_detected', locale=locale, task_type=task_type.title())}\n"
                f"{I18nManager.t('workflows.ml_training.target_label', locale=locale, target=suggested_target)}\n"
            )
            if suggested_features:
                feats = ", ".join(f"`{f}`" for f in suggested_features[:5])
                if len(suggested_features) > 5:
                    feats += f" ... (+{len(suggested_features) - 5})"
                msg += f"{I18nManager.t('workflows.ml_training.features_label', locale=locale, features=feats)}\n\n"

        msg += I18nManager.t('workflows.ml_training.proceed_question', locale=locale)
        return msg

    @staticmethod
    def schema_accepted_message(suggested_target: Optional[str], locale: Optional[str] = None) -> str:
        if suggested_target:
            return (
                f"{I18nManager.t('workflows.ml_training.schema_accepted', locale=locale)}\n\n"
                f"{I18nManager.t('workflows.ml_training.using_target', locale=locale, target=suggested_target)}\n\n"
                f"{I18nManager.t('workflows.ml_training.proceeding_to_selection', locale=locale)}"
            )
        return (
            f"{I18nManager.t('workflows.ml_training.schema_accepted', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.proceeding_to_selection', locale=locale)}"
        )

    @staticmethod
    def schema_rejected_message(locale: Optional[str] = None) -> str:
        return (
            f"{I18nManager.t('workflows.ml_training.schema_rejected', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.provide_different_file', locale=locale)}"
        )

    @staticmethod
    def telegram_upload_prompt(locale: Optional[str] = None) -> str:
        return I18nManager.t('workflows.ml_training.awaiting_file', locale=locale)

    @staticmethod
    def load_option_prompt(file_path: str, size_mb: float, locale: Optional[str] = None) -> str:
        """Prompt for load strategy selection (immediate or deferred)."""
        return (
            f"{I18nManager.t('workflows.ml_training.path_validated', locale=locale, file_path=file_path)}\n"
            f"{I18nManager.t('workflows.ml_training.size_label', locale=locale, size_mb=f'{size_mb:.2f}')}\n\n"
            f"{I18nManager.t('workflows.ml_training.choose_loading_strategy', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.load_now_option', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.load_now_features', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.defer_loading_option', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.defer_loading_features', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.select_strategy', locale=locale)}"
        )

    @staticmethod
    def schema_input_prompt(locale: Optional[str] = None) -> str:
        """Prompt for manual schema input (deferred loading)."""
        return (
            f"{I18nManager.t('workflows.ml_training.manual_schema_header', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.schema_formats_intro', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.format_1_label', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.format_1_example', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.format_2_label', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.format_2_example', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.format_3_label', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.format_3_example', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.type_schema_prompt', locale=locale)}"
        )

    @staticmethod
    def schema_accepted_deferred(target: str, n_features: int, locale: Optional[str] = None) -> str:
        """Message when manual schema is accepted (deferred loading)."""
        features_word = "feature" if n_features == 1 else "features"
        return (
            f"{I18nManager.t('workflows.ml_training.schema_accepted_deferred', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.target_deferred', locale=locale, target=target)}\n"
            f"{I18nManager.t('workflows.ml_training.features_count', locale=locale, n_features=n_features, features_word=features_word)}\n\n"
            f"{I18nManager.t('workflows.ml_training.data_load_later', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.proceeding_to_model', locale=locale)}"
        )

    @staticmethod
    def schema_parse_error(error_msg: str, locale: Optional[str] = None) -> str:
        """Error message when schema parsing fails."""
        return (
            f"{I18nManager.t('workflows.ml_training.schema_parse_error', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.parse_error_details', locale=locale, error_msg=error_msg)}\n\n"
            f"{I18nManager.t('workflows.ml_training.try_again', locale=locale)}"
        )

    @staticmethod
    def format_path_error(
        error_type: str,
        path: str,
        allowed_dirs: List[str] = None,
        size_mb: float = None,
        max_size_mb: int = None,
        allowed_extensions: List[str] = None,
        error_details: str = None,
        locale: Optional[str] = None
    ) -> str:
        """Format error message based on type."""
        if error_type == "not_found":
            return (
                f"{I18nManager.t('workflows.ml_training.file_not_found', locale=locale)}\n\n`{path}`\n\n"
                f"{I18nManager.t('workflows.ml_training.check_path', locale=locale)}"
            )
        elif error_type == "not_in_whitelist":
            dirs = "\n".join(f"• `{d}`" for d in (allowed_dirs or [])[:5])
            if allowed_dirs and len(allowed_dirs) > 5:
                dirs += f"\n• {I18nManager.t('common.and_more', locale=locale, count=len(allowed_dirs) - 5)}"
            return (
                f"{I18nManager.t('workflows.ml_training.access_denied', locale=locale)}\n\n`{path}`\n\n"
                f"{I18nManager.t('workflows.ml_training.allowed_label', locale=locale)}\n{dirs}"
            )
        elif error_type == "path_traversal":
            return I18nManager.t('workflows.ml_training.suspicious_patterns', locale=locale, path=path)
        elif error_type == "too_large":
            size_val = size_mb if size_mb is not None else 0
            max_val = max_size_mb if max_size_mb is not None else 0
            return (
                f"{I18nManager.t('workflows.ml_training.file_too_large_error', locale=locale)}\n\n"
                f"{I18nManager.t('workflows.ml_training.size_limit', locale=locale, size_mb=f'{size_val:.1f}', max_size_mb=max_val)}\n\n"
                f"{I18nManager.t('workflows.ml_training.try_smaller', locale=locale)}"
            )
        elif error_type == "invalid_extension":
            exts = ', '.join(allowed_extensions or [])
            return (
                f"{I18nManager.t('workflows.ml_training.unsupported_format', locale=locale)}\n\n`{path}`\n\n"
                f"{I18nManager.t('workflows.ml_training.supported_label', locale=locale, allowed_extensions=exts)}"
            )
        elif error_type == "empty":
            return I18nManager.t('workflows.ml_training.zero_bytes', locale=locale, path=path)
        elif error_type == "loading_error":
            return (
                f"{I18nManager.t('workflows.ml_training.loading_error', locale=locale)}\n\n"
                f"`{path}`\n\n{error_details}"
            )
        elif error_type == "security_validation":
            if error_details:
                return error_details
            return (
                f"{I18nManager.t('workflows.ml_training.security_validation_error', locale=locale)}\n\n"
                f"`{path}`"
            )
        elif error_type == "feature_disabled":
            return (
                f"{I18nManager.t('workflows.ml_training.feature_disabled', locale=locale)}\n\n"
                f"{I18nManager.t('workflows.ml_training.use_telegram_upload', locale=locale)}"
            )
        else:  # unexpected
            msg = f"{I18nManager.t('workflows.ml_training.unexpected_error', locale=locale)}\n\n`{path}`\n\n"
            if error_details:
                msg += f"{error_details}\n\n"
            msg += I18nManager.t('workflows.ml_training.try_restart', locale=locale)
            return msg

    @staticmethod
    def xgboost_description(locale: Optional[str] = None) -> str:
        """Description of XGBoost models for users."""
        return (
            f"{I18nManager.t('workflows.ml_training.xgboost_header', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_best_for', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_performance', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_advantages', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_adv_accuracy', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_adv_importance', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_adv_missing', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_adv_fast', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_when_use', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_use_credit', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_use_churn', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_use_price', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_use_tabular', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_when_not', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_not_image', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_not_text', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_not_timeseries', locale=locale)}"
        )

    @staticmethod
    def xgboost_hyperparameter_help(locale: Optional[str] = None) -> str:
        """Help text for XGBoost hyperparameters."""
        return (
            f"{I18nManager.t('workflows.ml_training.xgboost_hyperparams_header', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_n_estimators', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_max_depth', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_learning_rate', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_subsample', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_colsample', locale=locale)}"
        )

    @staticmethod
    def xgboost_setup_required(locale: Optional[str] = None) -> str:
        """Error message when XGBoost is unavailable due to missing OpenMP."""
        return (
            f"{I18nManager.t('workflows.ml_training.xgboost_setup_required', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_needs_openmp', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_macos_fix', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_brew_install', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_alternative', locale=locale)}\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_alt_note', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.xgboost_docs', locale=locale)}"
        )

    # =========================================================================
    # Password Protection Messages (Phase 6: Password Implementation)
    # =========================================================================

    @staticmethod
    def password_prompt(original_path: str, resolved_dir: str, locale: Optional[str] = None) -> str:
        """Password prompt for non-whitelisted path.

        Args:
            original_path: Original path string from user
            resolved_dir: Resolved parent directory
            locale: Language locale

        Returns:
            Markdown-formatted password prompt
        """
        return (
            f"{I18nManager.t('workflows.ml_training.password_required', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.password_not_in_whitelist', locale=locale)}\n"
            f"`{original_path}`\n\n"
            f"{I18nManager.t('workflows.ml_training.password_resolved_to', locale=locale, resolved_dir=resolved_dir)}\n\n"
            f"{I18nManager.t('workflows.ml_training.password_enter_prompt', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.password_security_note', locale=locale)}"
        )

    @staticmethod
    def password_success(directory: str, locale: Optional[str] = None) -> str:
        """Message shown after successful password entry.

        Args:
            directory: Directory that was granted access
            locale: Language locale

        Returns:
            Markdown-formatted success message
        """
        return (
            f"{I18nManager.t('workflows.ml_training.password_access_granted', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.password_directory_added', locale=locale)}\n"
            f"`{directory}`\n\n"
            f"{I18nManager.t('workflows.ml_training.password_can_access', locale=locale)}"
        )

    @staticmethod
    def password_failure(error_message: str, locale: Optional[str] = None) -> str:
        """Message shown after failed password attempt.

        Args:
            error_message: Error message from PasswordValidator
            locale: Language locale

        Returns:
            Markdown-formatted error message
        """
        return f"❌ {error_message}"

    @staticmethod
    def password_lockout(wait_seconds: int, locale: Optional[str] = None) -> str:
        """Message shown when user is locked out.

        Args:
            wait_seconds: Seconds until lockout expires
            locale: Language locale

        Returns:
            Markdown-formatted lockout message
        """
        return (
            f"{I18nManager.t('workflows.ml_training.password_locked', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.password_too_many_attempts', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.password_wait', locale=locale, wait_seconds=wait_seconds)}"
        )

    @staticmethod
    def password_timeout(locale: Optional[str] = None) -> str:
        """Message shown when password prompt expires.

        Args:
            locale: Language locale

        Returns:
            Markdown-formatted timeout message
        """
        return (
            f"{I18nManager.t('workflows.ml_training.password_timeout', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.password_prompt_expired', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.ml_training.password_restart', locale=locale)}"
        )


# Back Button Utilities (Phase 2: Workflow Back Button)
def create_back_button() -> InlineKeyboardButton:
    """
    Create standardized back button for workflow navigation.

    Returns:
        InlineKeyboardButton with callback_data='workflow_back'
    """
    return InlineKeyboardButton("⬅️ Back", callback_data="workflow_back")


def add_back_button(keyboard: List[List[InlineKeyboardButton]]) -> None:
    """
    Add back button to keyboard layout as last row.

    Args:
        keyboard: Existing keyboard layout (modified in-place)

    Example:
        >>> keyboard = [[button1, button2]]
        >>> add_back_button(keyboard)
        >>> # keyboard is now: [[button1, button2], [back_button]]
    """
    keyboard.append([create_back_button()])
