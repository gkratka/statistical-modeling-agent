"""Message templates for /models command - interactive model browser."""

from typing import List
from src.engines.model_catalog import ModelInfo


class ModelsMessages:
    """Message formatting for models browser workflow."""

    @staticmethod
    def models_list_message(
        page: int,
        total_pages: int,
        total_models: int
    ) -> str:
        """
        Format models list message.

        Args:
            page: Current page number (1-indexed)
            total_pages: Total number of pages
            total_models: Total number of models

        Returns:
            Formatted message
        """
        return (
            f"📚 **ML Model Catalog**\n\n"
            f"Browse {total_models} available models for training.\n"
            f"Click a model to see details, parameters, and use cases.\n\n"
            f"**Page {page}/{total_pages}**\n\n"
            f"💡 **Tip:** Models are organized by type and task."
        )

    @staticmethod
    def model_details_message(model: ModelInfo) -> str:
        """
        Format model details message.

        Args:
            model: Model information

        Returns:
            Formatted message with full model details
        """
        # Header
        msg = f"{model.icon} **{model.display_name}**\n\n"

        # Short description
        msg += f"_{model.short_description}_\n\n"

        # Category and task type
        msg += f"**Category:** {model.category.value.replace('_', ' ').title()}\n"
        msg += f"**Task Type:** {model.task_type.value.replace('_', ' ').title()}\n\n"

        # Variants (if applicable)
        if model.variants:
            msg += "**Variants:**\n"
            for variant in model.variants:
                variant_display = variant.replace('_', ' ').replace(model.id, '').strip()
                msg += f"  • {variant_display.title()}\n"
            msg += "\n"

        # Long description
        msg += f"**Description:**\n{model.long_description}\n\n"

        # Performance characteristics
        msg += "**Performance:**\n"
        msg += f"  • Training Speed: {model.training_speed}\n"
        msg += f"  • Prediction Speed: {model.prediction_speed}\n"
        msg += f"  • Interpretability: {model.interpretability}\n"
        if model.requires_tuning:
            msg += f"  • Requires Tuning: ✅ Yes\n"
        else:
            msg += f"  • Requires Tuning: ❌ No\n"
        msg += "\n"

        # Key parameters
        if model.parameters:
            msg += "**Key Parameters:**\n"
            for param in model.parameters[:3]:  # Show top 3
                msg += f"  • **{param.display_name}** (`{param.name}`)\n"
                msg += f"    {param.description}\n"
                msg += f"    Default: `{param.default}`, Range: `{param.range}`\n"
            if len(model.parameters) > 3:
                msg += f"  ... and {len(model.parameters) - 3} more parameters\n"
            msg += "\n"

        # Use cases
        if model.use_cases:
            msg += "**Best For:**\n"
            for use_case in model.use_cases[:4]:  # Show top 4
                msg += f"  • {use_case}\n"
            msg += "\n"

        # Strengths
        if model.strengths:
            msg += "**Strengths:**\n"
            for strength in model.strengths[:3]:  # Show top 3
                msg += f"  ✅ {strength}\n"
            msg += "\n"

        # Limitations
        if model.limitations:
            msg += "**Limitations:**\n"
            for limitation in model.limitations[:3]:  # Show top 3
                msg += f"  ⚠️ {limitation}\n"
            msg += "\n"

        # Footer
        msg += "💡 **Tip:** Use `/train` to start training with this model."

        return msg
