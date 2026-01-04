"""User-facing messages for ML prediction workflow."""

from typing import List, Dict, Any, Optional, Union
from telegram import InlineKeyboardButton
from src.bot.utils.markdown_escape import escape_markdown_v1
from src.utils.i18n_manager import I18nManager


class PredictionMessages:
    """Consolidated message templates for ML prediction workflow."""

    # =============================================================================
    # Step 1: Workflow Start
    # =============================================================================

    @staticmethod
    def prediction_start_message(locale: Optional[str] = None) -> str:
        """Initial message when /predict command is invoked."""
        return I18nManager.t('workflows.prediction.start_message', locale=locale)

    # =============================================================================
    # Steps 2-3: Data Loading
    # =============================================================================

    @staticmethod
    def data_source_selection_prompt(locale: Optional[str] = None) -> str:
        """Prompt for data source selection."""
        return I18nManager.t('workflows.prediction.data_source_prompt', locale=locale)

    @staticmethod
    def file_path_input_prompt(allowed_dirs: List[str], locale: Optional[str] = None) -> str:
        """Prompt for local file path input."""
        dirs = "\n".join(f"• `{d}`" for d in allowed_dirs[:5])
        if len(allowed_dirs) > 5:
            dirs += f"\n• ... and {len(allowed_dirs) - 5} more"

        return I18nManager.t('workflows.prediction.file_path_prompt', locale=locale, allowed_dirs=dirs)

    @staticmethod
    def telegram_upload_prompt(locale: Optional[str] = None) -> str:
        """Prompt for Telegram file upload."""
        return I18nManager.t('workflows.prediction.telegram_upload_prompt', locale=locale)

    @staticmethod
    def loading_data_message(locale: Optional[str] = None) -> str:
        """Loading indicator message."""
        return I18nManager.t('workflows.prediction.loading_data', locale=locale)

    @staticmethod
    def loading_deferred_data_message(file_path: str, locale: Optional[str] = None) -> str:
        """Message shown when loading deferred data before prediction execution."""
        # No escaping needed - backticks in locale YAML prevent Markdown interpretation
        return I18nManager.t('workflows.prediction.loading_deferred', locale=locale, file_path=file_path)

    @staticmethod
    def schema_confirmation_prompt(
        summary: str,
        available_columns: List[str], locale: Optional[str] = None) -> str:
        """Prompt for schema confirmation showing dataset summary."""
        # Escape column names before wrapping in backticks
        escaped_cols = [escape_markdown_v1(c) for c in available_columns[:10]]
        cols = ", ".join(f"`{c}`" for c in escaped_cols)
        if len(available_columns) > 10:
            cols += f" ... (+{len(available_columns) - 10})"

        msg = (
            f"{summary}\n\n"
            f"**{I18nManager.t('workflows.prediction.feature_selection.columns_available', locale=locale)} ({len(available_columns)}):**\n{cols}\n\n"
            f"{I18nManager.t('workflows.ml_training.proceed_question', locale=locale)}"
        )
        return msg

    @staticmethod
    def schema_accepted_message(locale: Optional[str] = None) -> str:
        """Message when schema is accepted."""
        return I18nManager.t('workflows.prediction.schema_accepted', locale=locale)

    @staticmethod
    def schema_rejected_message(locale: Optional[str] = None) -> str:
        """Message when schema is rejected."""
        return I18nManager.t('workflows.prediction.schema_rejected', locale=locale)

    @staticmethod
    def deferred_loading_confirmed_message(locale: Optional[str] = None) -> str:
        """Message shown when user chooses to defer loading."""
        return I18nManager.t('workflows.prediction.deferred_confirmed', locale=locale)

    # =============================================================================
    # Steps 4-5: Feature Selection
    # =============================================================================

    @staticmethod
    def feature_selection_prompt(
        available_columns: List[str],
        dataset_shape: tuple, locale: Optional[str] = None) -> str:
        """Prompt for feature selection."""
        # Escape column names before wrapping in backticks
        escaped_cols = [escape_markdown_v1(c) for c in available_columns]
        cols = "\n".join(f"• `{c}`" for c in escaped_cols)
        rows, cols_count = dataset_shape

        return (
            f"{I18nManager.t('workflows.prediction.feature_selection.header', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.prediction.feature_selection.dataset_label', locale=locale)} {rows:,} rows × {cols_count} columns\n\n"
            f"{I18nManager.t('workflows.prediction.feature_selection.columns_available', locale=locale)}\n{cols}\n\n"
            f"{I18nManager.t('workflows.prediction.feature_selection.select_prompt', locale=locale)}\n"
            f"• {I18nManager.t('workflows.prediction.feature_selection.format_example', locale=locale)}\n"
            f"• {I18nManager.t('workflows.prediction.feature_selection.no_prefix_warning', locale=locale)}\n"
            "• Must match your model's training features\n\n"
            f"{I18nManager.t('workflows.prediction.feature_selection.example_label', locale=locale)} {I18nManager.t('workflows.prediction.feature_selection.example_text', locale=locale)}"
        )

    @staticmethod
    def feature_selection_prompt_no_preview(locale: Optional[str] = None) -> str:
        """Feature selection prompt when data not yet loaded (defer mode)."""
        return (
            f"{I18nManager.t('workflows.prediction.feature_selection.header_deferred', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.prediction.feature_selection.select_prompt', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.prediction.feature_selection.format_label', locale=locale)} {I18nManager.t('workflows.prediction.feature_selection.format_example', locale=locale)}\n"
            f"{I18nManager.t('workflows.prediction.feature_selection.example_label', locale=locale)} `Attribute1, Attribute2, Attribute3`\n\n"
            f"{I18nManager.t('workflows.prediction.feature_selection.note_label', locale=locale)} {I18nManager.t('workflows.prediction.feature_selection.note_text', locale=locale)}\n"
            f"{I18nManager.t('workflows.prediction.feature_selection.no_prefix_warning', locale=locale)}\n\n"
            f"**{I18nManager.t('workflows.prediction.feature_selection.type_prompt', locale=locale)}**"
        )

    @staticmethod
    def features_selected_message(selected_features: List[str], locale: Optional[str] = None) -> str:
        """Message when features are selected."""
        # Escape feature names before wrapping in backticks
        escaped_feats = [escape_markdown_v1(f) for f in selected_features[:10]]
        feats = ", ".join(f"`{f}`" for f in escaped_feats)
        if len(selected_features) > 10:
            feats += f" ... (+{len(selected_features) - 10})"

        return (
            f"{I18nManager.t('workflows.prediction.feature_selection.selected_success', locale=locale)} ({len(selected_features)})\n\n"
            f"{feats}\n\n"
            f"{I18nManager.t('workflows.prediction.feature_selection.loading_models', locale=locale)}"
        )

    @staticmethod
    def feature_validation_error(
        reason: str,
        details: Optional[Dict[str, List[str]]] = None, locale: Optional[str] = None) -> str:
        """Error message when feature validation fails."""
        msg = f"{I18nManager.t('workflows.prediction.feature_selection.error_header', locale=locale)}\n\n{reason}\n\n"

        # Detect if invalid features contain format prefixes
        format_prefix_detected = False
        if details and 'invalid' in details and details['invalid']:
            format_prefixes = ['features:', 'target:']
            for feat in details['invalid']:
                if any(prefix in feat.lower() for prefix in format_prefixes):
                    format_prefix_detected = True
                    break

        if details:
            if 'missing' in details and details['missing']:
                # Escape missing feature names
                escaped_missing = [escape_markdown_v1(f) for f in details['missing']]
                msg += f"**{I18nManager.t('workflows.prediction.feature_selection.error_missing_label', locale=locale)}** {', '.join(f'`{f}`' for f in escaped_missing)}\n"
            if 'invalid' in details and details['invalid']:
                # Escape invalid feature names
                escaped_invalid = [escape_markdown_v1(f) for f in details['invalid']]
                msg += f"**{I18nManager.t('workflows.prediction.feature_selection.error_invalid_label', locale=locale)}** {', '.join(f'`{f}`' for f in escaped_invalid)}\n"

        msg += f"\n{I18nManager.t('workflows.prediction.feature_selection.error_retry', locale=locale)}"

        # Add helpful tip if format prefix was detected
        if format_prefix_detected:
            msg += (
                f"\n\n{I18nManager.t('workflows.prediction.feature_selection.error_tip', locale=locale)} "
                f"{I18nManager.t('workflows.prediction.feature_selection.no_prefix_warning', locale=locale)}"
            )

        return msg

    # =============================================================================
    # Steps 6-7: Model Selection
    # =============================================================================

    @staticmethod
    def model_selection_prompt(
        models: List[Dict[str, Any]],
        selected_features: List[str], locale: Optional[str] = None) -> str:
        """Prompt for model selection with compatible models."""
        if not models:
            return (
                f"{I18nManager.t('workflows.prediction.feature_selection.no_compatible_models', locale=locale)}\n\n"
                f"No trained models match the {len(selected_features)} features you selected.\n\n"
                f"**{I18nManager.t('workflows.prediction.feature_selection.no_models_options', locale=locale)}**\n"
                f"• {I18nManager.t('workflows.prediction.feature_selection.no_models_back', locale=locale)}\n"
                f"• {I18nManager.t('workflows.prediction.feature_selection.no_models_train', locale=locale)}"
            )

        models_text = ""
        for i, model in enumerate(models[:10], 1):
            # Get display name (prepared by ml_engine.list_models with custom_name priority)
            display_name = model.get('display_name', model.get('model_type', 'Unknown'))
            task_type = model.get('task_type', 'Unknown')
            target = model.get('target_column', 'Unknown')
            accuracy = model.get('metrics', {}).get('accuracy') or model.get('metrics', {}).get('r2')

            # Get feature count
            feature_columns = model.get('feature_columns', [])
            feature_count = len(feature_columns) if feature_columns else None

            # Escape for markdown
            escaped_display_name = escape_markdown_v1(display_name)
            escaped_target = escape_markdown_v1(target)
            escaped_task_type = escape_markdown_v1(task_type)

            # Build model line with feature count
            models_text += f"{i}. **{escaped_display_name.title()}**"
            if feature_count is not None and feature_count > 0:
                feature_word = I18nManager.t(
                    'workflows.prediction.model_selection.feature_singular' if feature_count == 1
                    else 'workflows.prediction.model_selection.feature_plural',
                    locale=locale
                )
                models_text += f" ({feature_count} {feature_word})"
            models_text += f" | {escaped_task_type}\n"

            # Add details line - use i18n for Target label
            models_text += f"   {I18nManager.t('workflows.prediction.model_selection.target_label', locale=locale)} {escaped_target}"
            if accuracy:
                models_text += f" | {I18nManager.t('workflows.prediction.model_selection.accuracy_label', locale=locale)}: {accuracy:.2%}"
            models_text += "\n\n"

        if len(models) > 10:
            models_text += f"... {I18nManager.t('common.and_more', locale=locale, count=len(models) - 10)}\n\n"

        compatible_word = I18nManager.t(
            'workflows.prediction.model_selection.compatible_models' if len(models) == 1
            else 'workflows.prediction.model_selection.compatible_models_plural',
            locale=locale
        )
        return (
            f"{I18nManager.t('workflows.prediction.model_selection.header', locale=locale)} ({len(models)} {compatible_word})\n\n"
            f"{models_text}"
            f"**{I18nManager.t('workflows.prediction.model_selection.prompt', locale=locale)}**"
        )

    @staticmethod
    def model_selected_message(
        model_id: str,
        model_type: str,
        target_column: str, locale: Optional[str] = None) -> str:
        """Message when model is selected."""
        # Escape all dynamic content
        escaped_model_type = escape_markdown_v1(model_type)
        escaped_target = escape_markdown_v1(target_column)
        escaped_id = escape_markdown_v1(model_id)

        return (
            f"{I18nManager.t('workflows.prediction.model_selection.selected_success', locale=locale)}\n\n"
            f"**{I18nManager.t('workflows.prediction.model_selection.type_label', locale=locale)}** {escaped_model_type.title()}\n"
            f"**{I18nManager.t('workflows.prediction.model_selection.target_label', locale=locale)}** `{escaped_target}`\n"
            f"**{I18nManager.t('workflows.prediction.model_selection.id_label', locale=locale)}** `{escaped_id}`\n\n"
            f"{I18nManager.t('workflows.prediction.model_selection.preparing', locale=locale)}"
        )

    @staticmethod
    def no_models_available_error(locale: Optional[str] = None) -> str:
        """Error when no models are available for user."""
        return I18nManager.t('workflows.prediction.no_models_error', locale=locale)

    @staticmethod
    def model_feature_mismatch_error(
        model_features: List[str],
        selected_features: List[str], locale: Optional[str] = None) -> str:
        """Error when selected features don't match model requirements."""
        model_set = set(model_features)
        selected_set = set(selected_features)

        missing = sorted(model_set - selected_set)
        extra = sorted(selected_set - model_set)

        msg = (
            f"{I18nManager.t('workflows.prediction.model_selection.feature_mismatch_error', locale=locale)}\n\n"
            f"Model requires {len(model_features)} features, you selected {len(selected_features)}.\n\n"
        )

        if missing:
            msg += f"**{I18nManager.t('workflows.prediction.model_selection.missing_features', locale=locale)}** {', '.join(f'`{f}`' for f in missing)}\n"
        if extra:
            msg += f"**{I18nManager.t('workflows.prediction.model_selection.extra_features', locale=locale)}** {', '.join(f'`{f}`' for f in extra)}\n"

        msg += f"\n**{I18nManager.t('workflows.prediction.model_selection.required_features', locale=locale)}**\n"
        msg += ", ".join(f"`{f}`" for f in model_features)
        msg += f"\n\n{I18nManager.t('workflows.prediction.model_selection.mismatch_instructions', locale=locale)}"

        return msg

    # =============================================================================
    # Steps 8-9: Prediction Column Confirmation
    # =============================================================================

    @staticmethod
    def prediction_column_prompt(
        target_column: str,
        existing_columns: List[str], locale: Optional[str] = None) -> str:
        """Prompt for prediction column name confirmation."""
        # Escape target column name
        escaped_target = escape_markdown_v1(target_column)
        escaped_predicted = escape_markdown_v1(f"{target_column}_predicted")

        return (
            f"{I18nManager.t('workflows.prediction.prediction_column.header', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.prediction.prediction_column.prompt', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.prediction.prediction_column.default_label', locale=locale)} `{escaped_predicted}`\n\n"
            f"**{I18nManager.t('workflows.prediction.prediction_column.options_label', locale=locale)}**\n"
            "• Press ✅ to use default\n"
            "• Type custom name (e.g., `price_forecast`)\n\n"
            f"**{I18nManager.t('workflows.prediction.prediction_column.note_label', locale=locale)}** {I18nManager.t('workflows.prediction.prediction_column.note_text', locale=locale)}"
        )

    @staticmethod
    def column_name_confirmed_message(column_name: str, locale: Optional[str] = None) -> str:
        """Message when column name is confirmed."""
        return (
            f"{I18nManager.t('workflows.prediction.prediction_column.confirmed_success', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.prediction.prediction_column.confirmed_text', locale=locale)} `{column_name}`\n\n"
            f"{I18nManager.t('workflows.prediction.prediction_column.ready_message', locale=locale)}"
        )

    @staticmethod
    def column_name_conflict_error(
        column_name: str,
        existing_columns: List[str], locale: Optional[str] = None) -> str:
        """Error when column name conflicts with existing columns."""
        # Escape column names
        escaped_name = escape_markdown_v1(column_name)
        escaped_existing = [escape_markdown_v1(c) for c in existing_columns[:10]]

        return (
            f"{I18nManager.t('workflows.prediction.prediction_column.conflict_error', locale=locale)}\n\n"
            f"`{escaped_name}` {I18nManager.t('workflows.prediction.prediction_column.conflict_exists', locale=locale)}.\n\n"
            f"**{I18nManager.t('workflows.prediction.prediction_column.conflict_existing', locale=locale)}**\n"
            + ", ".join(f"`{c}`" for c in escaped_existing)
            + (f" ... (+{len(existing_columns) - 10})" if len(existing_columns) > 10 else "")
            + f"\n\n{I18nManager.t('workflows.prediction.prediction_column.conflict_retry', locale=locale)}"
        )

    # =============================================================================
    # Steps 10-11: Ready to Run
    # =============================================================================

    @staticmethod
    def ready_to_run_prompt(
        model_type: str,
        target_column: str,
        prediction_column: str,
        n_rows: Union[int, str],
        n_features: int, locale: Optional[str] = None) -> str:
        """Prompt showing ready state before execution."""
        # Conditional formatting for n_rows (handle defer mode where n_rows="Deferred")
        dataset_info = f"{n_rows:,} rows" if isinstance(n_rows, int) else n_rows

        return (
            f"{I18nManager.t('workflows.prediction.ready_to_run.header', locale=locale)}\n\n"
            f"**{I18nManager.t('workflows.prediction.ready_to_run.model_label', locale=locale)}** {model_type.title()}\n"
            f"**{I18nManager.t('workflows.prediction.ready_to_run.target_label', locale=locale)}** `{target_column}`\n"
            f"**{I18nManager.t('workflows.prediction.ready_to_run.output_column_label', locale=locale)}** `{prediction_column}`\n\n"
            f"**{I18nManager.t('workflows.prediction.ready_to_run.dataset_label', locale=locale)}** {dataset_info} × {n_features} features\n\n"
            f"**{I18nManager.t('workflows.prediction.ready_to_run.choose_action', locale=locale)}**\n"
            f"• {I18nManager.t('workflows.prediction.ready_to_run.run_instructions', locale=locale)}\n"
            f"• {I18nManager.t('workflows.prediction.ready_to_run.back_instructions', locale=locale)}\n\n"
            "Ready when you are!"
        )

    # =============================================================================
    # Steps 12-13: Execution and Results
    # =============================================================================

    @staticmethod
    def running_prediction_message(locale: Optional[str] = None) -> str:
        """Message during prediction execution."""
        return I18nManager.t('workflows.prediction.running', locale=locale)

    @staticmethod
    def prediction_success_message(
        model_type: str,
        n_predictions: int,
        prediction_column: str,
        execution_time: float,
        preview_data: str,
        statistics: Optional[Dict[str, Any]] = None, locale: Optional[str] = None) -> str:
        """Success message with prediction results and statistics."""
        msg = (
            f"{I18nManager.t('workflows.prediction.results.success_header', locale=locale)}\n\n"
            f"**{I18nManager.t('workflows.prediction.results.model_label', locale=locale)}** {model_type.title()}\n"
            f"**{I18nManager.t('workflows.prediction.results.predictions_label', locale=locale)}** {n_predictions:,} rows\n"
            f"**{I18nManager.t('workflows.prediction.results.time_label', locale=locale)}** {execution_time:.2f}s\n\n"
        )

        if statistics:
            # Escape prediction column name to prevent markdown parsing errors (e.g., "class_predicted" has underscore)
            escaped_pred_col = escape_markdown_v1(prediction_column)
            msg += f"**{I18nManager.t('workflows.prediction.results.statistics_header', locale=locale)} ({escaped_pred_col}):**\n"

            # Show class distribution for classification predictions
            if 'class_distribution' in statistics:
                class_dist = statistics['class_distribution']
                for class_val in sorted(class_dist.keys()):
                    info = class_dist[class_val]
                    msg += f"• Class {class_val}: {info['count']} ({info['percentage']}%)\n"
            else:
                # Show numeric statistics for regression
                if 'mean' in statistics:
                    msg += f"{I18nManager.t('workflows.prediction.results.stat_mean', locale=locale)} {statistics['mean']:.4f}\n"
                if 'std' in statistics:
                    msg += f"{I18nManager.t('workflows.prediction.results.stat_std', locale=locale)} {statistics['std']:.4f}\n"
                if 'min' in statistics:
                    msg += f"{I18nManager.t('workflows.prediction.results.stat_min', locale=locale)} {statistics['min']:.4f}\n"
                if 'max' in statistics:
                    msg += f"{I18nManager.t('workflows.prediction.results.stat_max', locale=locale)} {statistics['max']:.4f}\n"
                if 'median' in statistics:
                    msg += f"{I18nManager.t('workflows.prediction.results.stat_median', locale=locale)} {statistics['median']:.4f}\n"
            msg += "\n"

        # Escape preview data to prevent markdown parsing errors (column names may have special chars)
        escaped_preview = escape_markdown_v1(preview_data)
        msg += (
            f"**{I18nManager.t('workflows.prediction.results.preview_header', locale=locale)}**\n{escaped_preview}\n\n"
            f"{I18nManager.t('workflows.prediction.results.download_prompt', locale=locale)}"
        )

        return msg

    @staticmethod
    def prediction_error_message(error_details: str, locale: Optional[str] = None) -> str:
        """Error message when prediction fails."""
        return (
            f"{I18nManager.t('workflows.prediction.results.error_header', locale=locale)}\n\n"
            f"{error_details}\n\n"
            f"{I18nManager.t('workflows.prediction.results.error_issues_label', locale=locale)}\n"
            f"• {I18nManager.t('workflows.prediction.results.error_feature_mismatch', locale=locale)}\n"
            f"• {I18nManager.t('workflows.prediction.results.error_invalid_types', locale=locale)}\n"
            f"• {I18nManager.t('workflows.prediction.results.error_missing_values', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.prediction.results.error_back_instructions', locale=locale)}"
        )

    # =============================================================================
    # Workflow Completion
    # =============================================================================

    @staticmethod
    def workflow_complete_message(locale: Optional[str] = None) -> str:
        """Message when workflow is complete."""
        return I18nManager.t('workflows.prediction.workflow_complete', locale=locale)

    # =============================================================================
    # Error Messages
    # =============================================================================

    @staticmethod
    def unexpected_error(error_msg: str, locale: Optional[str] = None) -> str:
        """Generic unexpected error message."""
        # Escape error message to prevent markdown issues
        escaped_msg = escape_markdown_v1(error_msg)

        # Get translated error header (first line only)
        error_header = I18nManager.t('workflows.ml_training.unexpected_error', locale=locale)
        error_header_line = error_header.split('\n')[0]

        return (
            f"{error_header_line}\n\n"
            f"{escaped_msg}\n\n"
            "Try /predict to start over."
        )

    @staticmethod
    def file_loading_error(file_path: str, error_details: str, locale: Optional[str] = None) -> str:
        """Error when file loading fails."""
        # File paths don't need escaping when wrapped in backticks
        # Error details still need escaping (may contain user input)
        escaped_details = escape_markdown_v1(error_details)

        return (
            f"{I18nManager.t('workflows.ml_training.loading_error', locale=locale)}\n\n"
            f"`{file_path}`\n\n"
            f"{escaped_details}\n\n"
            f"{I18nManager.t('workflows.ml_training.check_path', locale=locale)}"
        )

    @staticmethod
    def validation_error(field: str, reason: str, locale: Optional[str] = None) -> str:
        """Generic validation error."""
        return (
            f"{I18nManager.t('workflows.ml_training.security_validation_error', locale=locale)}\n\n"
            f"**Field:** {field}\n"
            f"**Reason:** {reason}\n\n"
            f"{I18nManager.t('workflows.ml_training.try_again', locale=locale)}"
        )

    # =============================================================================
    # NEW: Local File Save Workflow Messages
    # =============================================================================

    @staticmethod
    def output_options_prompt(locale: Optional[str] = None) -> str:
        """Present output method choices after predictions complete."""
        return (
            f"{I18nManager.t('workflows.prediction.output_options.save_header', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.prediction.output_options.save_prompt', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.prediction.output_options.save_local_option', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.prediction.output_options.save_telegram_option', locale=locale)}\n\n"
            f"{I18nManager.t('workflows.prediction.output_options.done_option', locale=locale)}"
        )

    @staticmethod
    def save_path_input_prompt(allowed_dirs: List[str], locale: Optional[str] = None) -> str:
        """Prompt user for output directory path."""
        dirs = "\n".join(f"• `{d}`" for d in allowed_dirs[:5])
        if len(allowed_dirs) > 5:
            dirs += f"\n• ... {I18nManager.t('common.and_more', locale=locale, count=len(allowed_dirs) - 5)}"

        return (
            f"{I18nManager.t('workflows.prediction.output_options.path_header', locale=locale)}\n\n"
            f"**{I18nManager.t('workflows.prediction.output_options.path_allowed_label', locale=locale)}**\n{dirs}\n\n"
            f"**{I18nManager.t('workflows.prediction.output_options.path_examples_label', locale=locale)}**\n"
            "• `/Users/username/results`\n"
            "• `~/Documents/predictions`\n\n"
            f"{I18nManager.t('workflows.prediction.output_options.path_prompt', locale=locale)}"
        )

    @staticmethod
    def filename_confirmation_prompt(
        default_name: str,
        directory: str, locale: Optional[str] = None) -> str:
        """Show default filename with rename option."""
        escaped_default = escape_markdown_v1(default_name)
        escaped_dir = escape_markdown_v1(directory)

        return (
            f"{I18nManager.t('workflows.prediction.output_options.filename_header', locale=locale)}\n\n"
            f"**{I18nManager.t('workflows.prediction.output_options.filename_directory_label', locale=locale)}** `{escaped_dir}`\n"
            f"**{I18nManager.t('workflows.prediction.output_options.filename_default_label', locale=locale)}** `{escaped_default}`\n\n"
            f"**{I18nManager.t('workflows.prediction.output_options.filename_options_label', locale=locale)}**\n"
            "• ✅ Accept - Use default name\n"
            "• ✏️ Custom - Provide your own name\n\n"
            f"{I18nManager.t('workflows.prediction.output_options.filename_prompt', locale=locale)}"
        )

    @staticmethod
    def file_save_success_message(
        full_path: str,
        n_rows: int, locale: Optional[str] = None) -> str:
        """Confirm successful file save."""
        escaped_path = escape_markdown_v1(full_path)

        return (
            f"{I18nManager.t('workflows.prediction.output_options.save_success_header', locale=locale)}\n\n"
            f"**{I18nManager.t('workflows.prediction.output_options.save_success_location', locale=locale)}** `{escaped_path}`\n"
            f"**{I18nManager.t('workflows.prediction.output_options.save_success_rows', locale=locale)}** {n_rows:,}\n\n"
            f"{I18nManager.t('workflows.prediction.output_options.save_success_message', locale=locale)}"
        )

    @staticmethod
    def file_save_error_message(
        error_type: str,
        details: str, locale: Optional[str] = None) -> str:
        """Error messages for various save failures."""
        escaped_details = escape_markdown_v1(details)

        return (
            f"{I18nManager.t('workflows.prediction.output_options.save_error_header', locale=locale, error_type=error_type)}\n\n"
            f"{escaped_details}\n\n"
            f"**{I18nManager.t('workflows.prediction.output_options.save_error_options', locale=locale)}**\n"
            f"• {I18nManager.t('workflows.prediction.buttons.try_again', locale=locale)}\n"
            f"• {I18nManager.t('workflows.prediction.buttons.download_telegram', locale=locale)}\n"
            f"• {I18nManager.t('workflows.prediction.buttons.cancel_workflow', locale=locale)}"
        )


# =============================================================================
# Button Creation Utilities
# =============================================================================

def create_data_source_buttons(locale: Optional[str] = None) -> List[List[InlineKeyboardButton]]:
    """Create data source selection buttons."""
    from src.bot.messages.local_path_messages import create_back_button
    return [
        [
            InlineKeyboardButton(
                I18nManager.t('workflows.prediction.buttons.upload_file', locale=locale),
                callback_data="pred_upload"
            ),
            InlineKeyboardButton(
                I18nManager.t('workflows.prediction.buttons.local_path', locale=locale),
                callback_data="pred_local_path"
            )
        ],
        [InlineKeyboardButton(
            I18nManager.t('workflows.prediction.buttons.use_template', locale=locale),
            callback_data="use_pred_template"
        )],
        [create_back_button(locale=locale, callback_data="pred_back")]
    ]


def create_load_option_buttons(locale: Optional[str] = None) -> List[List[InlineKeyboardButton]]:
    """Create load option selection buttons for defer loading workflow."""
    from src.bot.messages.local_path_messages import add_back_button
    keyboard = [
        [InlineKeyboardButton(
            I18nManager.t('workflows.prediction.buttons.load_now', locale=locale),
            callback_data="pred_load_immediate"
        )],
        [InlineKeyboardButton(
            I18nManager.t('workflows.prediction.buttons.defer_loading', locale=locale),
            callback_data="pred_load_defer"
        )]
    ]
    add_back_button(keyboard, locale=locale, callback_data="pred_back")  # Add back button support
    return keyboard


def create_schema_confirmation_buttons(locale: Optional[str] = None) -> List[List[InlineKeyboardButton]]:
    """Create schema confirmation buttons."""
    from src.bot.messages.local_path_messages import create_back_button
    return [
        [
            InlineKeyboardButton(
                I18nManager.t('workflows.prediction.buttons.continue', locale=locale),
                callback_data="pred_schema_accept"
            ),
            InlineKeyboardButton(
                I18nManager.t('workflows.prediction.buttons.different_file', locale=locale),
                callback_data="pred_schema_reject"
            )
        ],
        [create_back_button(locale=locale, callback_data="pred_back")]
    ]


def create_column_confirmation_buttons(locale: Optional[str] = None) -> List[List[InlineKeyboardButton]]:
    """Create prediction column confirmation buttons."""
    from src.bot.messages.local_path_messages import create_back_button
    return [
        [InlineKeyboardButton(
            I18nManager.t('workflows.prediction.buttons.use_default', locale=locale),
            callback_data="pred_column_default"
        )],
        [create_back_button(locale=locale, callback_data="pred_back")]
    ]


def create_ready_to_run_buttons(locale: Optional[str] = None) -> List[List[InlineKeyboardButton]]:
    """Create ready to run buttons."""
    from src.bot.messages.local_path_messages import create_back_button
    return [
        [InlineKeyboardButton(
            I18nManager.t('workflows.prediction.buttons.run_model', locale=locale),
            callback_data="pred_run"
        )],
        [create_back_button(locale=locale, callback_data="pred_back")]
    ]


def create_model_selection_buttons(
    models: List[Dict[str, Any]],
    locale: Optional[str] = None
) -> List[List[InlineKeyboardButton]]:
    """Create model selection buttons using indices (up to 10 models)."""
    from src.bot.messages.local_path_messages import create_back_button

    buttons = []
    for i, model in enumerate(models[:10], 0):  # Start at 0 for index-based lookup
        # Get display name (prepared by ml_engine.list_models with custom_name priority)
        display_name = model.get('display_name', model.get('model_type', 'Unknown'))

        # Get feature count
        feature_columns = model.get('feature_columns', [])
        feature_count = len(feature_columns) if feature_columns else None

        # Build button text
        button_text = f"{i+1}. {display_name}"

        # Add feature count if available
        if feature_count is not None and feature_count > 0:
            # Use i18n for singular/plural feature word
            feature_word = I18nManager.t(
                'workflows.prediction.model_selection.feature_singular' if feature_count == 1
                else 'workflows.prediction.model_selection.feature_plural',
                locale=locale
            )
            button_text += f" ({feature_count} {feature_word})"

        button = InlineKeyboardButton(
            button_text,  # Enhanced display text
            callback_data=f"pred_model_{i}"  # Callback uses 0-based index
        )
        buttons.append([button])

    buttons.append([create_back_button(locale=locale, callback_data="pred_back")])
    # Add delete models button
    buttons.append([InlineKeyboardButton(
        I18nManager.t('prediction.delete_models.button', locale=locale),
        callback_data="pred_delete_start"
    )])
    return buttons


def create_delete_models_checkbox_buttons(
    models: List[Dict[str, Any]],
    selected_indices: set,
    locale: Optional[str] = None
) -> List[List[InlineKeyboardButton]]:
    """Create checkbox-style buttons for model deletion selection."""
    buttons = []
    selected_prefix = I18nManager.t('prediction.delete_models.selected_prefix', locale=locale)
    unselected_prefix = I18nManager.t('prediction.delete_models.unselected_prefix', locale=locale)

    for i, model in enumerate(models[:10]):
        # Get display name (prepared by ml_engine.list_models with custom_name priority)
        display_name = model.get('display_name', model.get('model_type', 'Unknown'))

        # Add checkbox prefix based on selection state
        prefix = selected_prefix if i in selected_indices else unselected_prefix
        button_text = f"{prefix}{display_name}"

        button = InlineKeyboardButton(
            button_text,
            callback_data=f"pred_delete_toggle_{i}"
        )
        buttons.append([button])

    # Add confirm and cancel buttons
    buttons.append([
        InlineKeyboardButton(
            I18nManager.t('prediction.delete_models.confirm_button', locale=locale),
            callback_data="pred_delete_confirm"
        ),
        InlineKeyboardButton(
            I18nManager.t('prediction.delete_models.cancel_button', locale=locale),
            callback_data="pred_delete_cancel"
        )
    ])
    return buttons


def create_path_error_recovery_buttons(locale: Optional[str] = None) -> List[List[InlineKeyboardButton]]:
    """
    Create recovery buttons for path validation errors.

    These buttons help users recover from path validation failures without
    losing their workflow progress.
    """
    return [
        [InlineKeyboardButton(
            I18nManager.t('workflows.prediction.buttons.try_again', locale=locale),
            callback_data="pred_retry_path"
        )],
        [InlineKeyboardButton(
            I18nManager.t('workflows.prediction.buttons.different_source', locale=locale),
            callback_data="pred_back_to_source"
        )],
        [InlineKeyboardButton(
            I18nManager.t('workflows.prediction.buttons.cancel_workflow', locale=locale),
            callback_data="pred_cancel"
        )]
    ]


# NEW: Button helpers for local file save workflow
def create_output_option_buttons(locale: Optional[str] = None) -> List[List[InlineKeyboardButton]]:
    """Create output method selection buttons.

    Note: 'Save to Local Path' button removed since predictions are auto-saved locally.
    """
    return [
        [InlineKeyboardButton(
            I18nManager.t('workflows.prediction.buttons.download_telegram', locale=locale),
            callback_data="pred_output_telegram"
        )],
        [InlineKeyboardButton(
            I18nManager.t('workflows.prediction.buttons.done', locale=locale),
            callback_data="pred_output_done"
        )]
    ]


def create_filename_confirmation_buttons(locale: Optional[str] = None) -> List[List[InlineKeyboardButton]]:
    """Create filename confirmation buttons."""
    return [
        [InlineKeyboardButton(
            I18nManager.t('workflows.prediction.buttons.accept_default', locale=locale),
            callback_data="pred_filename_default"
        )],
        [InlineKeyboardButton(
            I18nManager.t('workflows.prediction.buttons.custom_name', locale=locale),
            callback_data="pred_filename_custom"
        )]
    ]
