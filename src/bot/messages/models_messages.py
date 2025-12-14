"""Message templates for /models command - interactive model browser."""

from typing import List, Optional
from src.engines.model_catalog import ModelInfo
from src.utils.i18n_manager import I18nManager


class ModelsMessages:
    """Message formatting for models browser workflow."""

    @staticmethod
    def category_selection_message(locale: Optional[str] = None) -> str:
        """
        Format category selection message with i18n support.

        Args:
            locale: Language code (e.g., 'en', 'pt')

        Returns:
            Formatted message
        """
        header = I18nManager.t("workflows.models.category_header", locale=locale)
        description = I18nManager.t("workflows.models.category_description", locale=locale)
        regression_desc = I18nManager.t("workflow_state.model_selection.regression_description", locale=locale)
        classification_desc = I18nManager.t("workflow_state.model_selection.classification_description", locale=locale)
        neural_desc = I18nManager.t("workflow_state.model_selection.neural_description", locale=locale)
        prompt = I18nManager.t("workflows.models.category_prompt", locale=locale)

        return (
            f"{header}\n\n"
            f"{description}\n\n"
            f"{regression_desc}\n"
            f"{classification_desc}\n"
            f"{neural_desc}\n\n"
            f"{prompt}"
        )

    @staticmethod
    def models_list_message(
        page: int,
        total_pages: int,
        total_models: int,
        category: Optional[str] = None,
        locale: Optional[str] = None
    ) -> str:
        """
        Format models list message with i18n support.

        Args:
            page: Current page number (1-indexed)
            total_pages: Total number of pages
            total_models: Total number of models
            category: Category filter ("regression", "classification", "neural")
            locale: Language code (e.g., 'en', 'pt')

        Returns:
            Formatted message
        """
        # Get category-specific header if category is specified
        if category:
            category_name_key = f"workflows.models.category_{category}"
            category_name = I18nManager.t(category_name_key, locale=locale)
            header = I18nManager.t(
                "workflows.models.list_header_category",
                locale=locale,
                category=category_name
            )
        else:
            header = I18nManager.t("workflows.models.list_header", locale=locale)

        browse_msg = I18nManager.t(
            "workflows.models.browse_message",
            locale=locale,
            total_models=total_models
        )
        details_msg = I18nManager.t("workflows.models.list_details", locale=locale)
        page_info = I18nManager.t(
            "workflows.models.page_info",
            locale=locale,
            current=page,
            total=total_pages
        )
        tip = I18nManager.t("workflows.models.list_tip", locale=locale)

        return (
            f"{header}\n\n"
            f"{browse_msg}\n"
            f"{details_msg}\n\n"
            f"{page_info}\n\n"
            f"{tip}"
        )

    @staticmethod
    def model_details_message(
        model: ModelInfo,
        locale: Optional[str] = None
    ) -> str:
        """
        Format model details message with i18n support.

        Args:
            model: Model information
            locale: Language code (e.g., 'en', 'pt')

        Returns:
            Formatted message with full model details
        """
        # Header
        msg = f"{model.icon} **{model.display_name}**\n\n"

        # Short description
        msg += f"_{model.short_description}_\n\n"

        # Category and task type
        category_label = I18nManager.t("workflows.models.category", locale=locale)
        task_type_label = I18nManager.t("workflows.models.task_type", locale=locale)
        msg += f"**{category_label}:** {model.category.value.replace('_', ' ').title()}\n"
        msg += f"**{task_type_label}:** {model.task_type.value.replace('_', ' ').title()}\n\n"

        # Variants (if applicable)
        if model.variants:
            variants_label = I18nManager.t("workflows.models.variants", locale=locale)
            msg += f"**{variants_label}:**\n"
            for variant in model.variants:
                variant_display = variant.replace('_', ' ').replace(model.id, '').strip()
                msg += f"  • {variant_display.title()}\n"
            msg += "\n"

        # Long description
        description_label = I18nManager.t("workflows.models.description", locale=locale)
        msg += f"**{description_label}:**\n{model.long_description}\n\n"

        # Performance characteristics
        performance_label = I18nManager.t("workflows.models.performance", locale=locale)
        training_speed_label = I18nManager.t("workflows.models.training_speed", locale=locale)
        prediction_speed_label = I18nManager.t("workflows.models.prediction_speed", locale=locale)
        interpretability_label = I18nManager.t("workflows.models.interpretability", locale=locale)
        requires_tuning_label = I18nManager.t("workflows.models.requires_tuning", locale=locale)

        msg += f"**{performance_label}:**\n"
        msg += f"  • {training_speed_label}: {model.training_speed}\n"
        msg += f"  • {prediction_speed_label}: {model.prediction_speed}\n"
        msg += f"  • {interpretability_label}: {model.interpretability}\n"
        if model.requires_tuning:
            msg += f"  • {requires_tuning_label}: ✅ {I18nManager.t('workflows.models.yes', locale=locale)}\n"
        else:
            msg += f"  • {requires_tuning_label}: ❌ {I18nManager.t('workflows.models.no', locale=locale)}\n"
        msg += "\n"

        # Key parameters
        if model.parameters:
            key_params_label = I18nManager.t("workflows.models.key_parameters", locale=locale)
            more_params_label = I18nManager.t(
                "workflows.models.more_parameters",
                locale=locale,
                count=len(model.parameters) - 3
            )
            msg += f"**{key_params_label}:**\n"
            for param in model.parameters[:3]:  # Show top 3
                msg += f"  • **{param.display_name}** (`{param.name}`)\n"
                msg += f"    {param.description}\n"
                msg += f"    {I18nManager.t('workflows.models.default', locale=locale)}: `{param.default}`, {I18nManager.t('workflows.models.range', locale=locale)}: `{param.range}`\n"
            if len(model.parameters) > 3:
                msg += f"  {more_params_label}\n"
            msg += "\n"

        # Use cases
        if model.use_cases:
            best_for_label = I18nManager.t("workflows.models.best_for", locale=locale)
            msg += f"**{best_for_label}:**\n"
            for use_case in model.use_cases[:4]:  # Show top 4
                msg += f"  • {use_case}\n"
            msg += "\n"

        # Strengths
        if model.strengths:
            strengths_label = I18nManager.t("workflows.models.strengths", locale=locale)
            msg += f"**{strengths_label}:**\n"
            for strength in model.strengths[:3]:  # Show top 3
                msg += f"  ✅ {strength}\n"
            msg += "\n"

        # Limitations
        if model.limitations:
            limitations_label = I18nManager.t("workflows.models.limitations", locale=locale)
            msg += f"**{limitations_label}:**\n"
            for limitation in model.limitations[:3]:  # Show top 3
                msg += f"  ⚠️ {limitation}\n"
            msg += "\n"

        # Footer
        footer_tip = I18nManager.t("workflows.models.details_tip", locale=locale)
        msg += footer_tip

        return msg
