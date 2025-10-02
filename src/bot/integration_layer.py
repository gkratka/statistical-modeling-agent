"""Integration layer connecting State Manager to Telegram bot handlers."""

import logging
from typing import Any, Dict, Literal, Optional
import pandas as pd

from src.core.state_manager import (
    StateManager,
    UserSession,
    WorkflowType,
    MLTrainingState,
    MLPredictionState
)
from src.utils.exceptions import DataSizeLimitError, InvalidStateTransitionError

logger = logging.getLogger(__name__)

HandlerType = Literal["general", "workflow", "command"]


class IntegrationLayer:
    """Integration layer for Telegram bot and state management."""

    def __init__(self, state_manager: Optional[StateManager] = None):
        """Initialize integration layer with optional StateManager."""
        self.state_manager = state_manager or StateManager()
        logger.info("IntegrationLayer initialized")

    async def route_message(
        self,
        session: UserSession,
        message_text: str
    ) -> HandlerType:
        """Determine which handler should process message."""
        if session.workflow_type is not None:
            logger.debug(f"Routing to workflow handler for {session.workflow_type.value}")
            return "workflow"

        if message_text.startswith("/"):
            logger.debug("Routing to command handler")
            return "command"

        logger.debug("Routing to general handler")
        return "general"

    async def store_uploaded_data(
        self,
        session: UserSession,
        dataframe: pd.DataFrame,
        metadata: Dict[str, Any]
    ) -> None:
        """Store uploaded data in session with metadata."""
        logger.info(
            f"Storing data for session {session.session_key}: "
            f"shape={dataframe.shape}, file={metadata.get('filename', 'unknown')}"
        )
        await self.state_manager.store_data(session, dataframe)
        session.selections["data_metadata"] = metadata
        logger.debug(f"Data stored successfully: {dataframe.shape}")

    async def handle_workflow_message(
        self,
        session: UserSession,
        message_text: str
    ) -> str:
        """Handle message within active workflow context."""
        if session.workflow_type is None:
            return "No active workflow. Use /train, /predict, or /stats to start."

        logger.info(
            f"Handling workflow message: workflow={session.workflow_type.value}, "
            f"state={session.current_state}"
        )

        await self.state_manager.add_to_history(session, "user", message_text)

        if session.workflow_type == WorkflowType.ML_TRAINING:
            return await self._handle_ml_training_message(session, message_text)
        elif session.workflow_type in (WorkflowType.ML_PREDICTION, WorkflowType.STATS_ANALYSIS):
            return f"{session.workflow_type.value} workflow not yet implemented."

        return "Workflow type not yet implemented."

    async def _handle_ml_training_message(
        self,
        session: UserSession,
        message_text: str
    ) -> str:
        """Handle message in ML training workflow."""
        current_state = session.current_state

        if current_state == MLTrainingState.SELECTING_TARGET.value:
            # Parse target column selection
            target_col = message_text.strip()

            # Validate column exists
            if session.uploaded_data is None:
                return "Error: No data uploaded. Please upload data first."

            if target_col not in session.uploaded_data.columns:
                available = ", ".join(session.uploaded_data.columns)
                return f"Column '{target_col}' not found. Available: {available}"

            # Store selection
            session.selections["target"] = target_col

            # Transition to next state
            success, error_msg, missing = await self.state_manager.transition_state(
                session,
                MLTrainingState.SELECTING_FEATURES.value
            )

            if success:
                # Build feature selection prompt
                features = [c for c in session.uploaded_data.columns if c != target_col]
                return (
                    f"Target variable '{target_col}' selected.\n\n"
                    f"Select features (comma-separated):\n{', '.join(features)}"
                )
            else:
                return f"Error: {error_msg}"

        elif current_state == MLTrainingState.SELECTING_FEATURES.value:
            # Parse feature selection
            feature_names = [f.strip() for f in message_text.split(",")]

            # Validate features
            if session.uploaded_data is None:
                return "Error: No data uploaded."

            invalid = [f for f in feature_names if f not in session.uploaded_data.columns]
            if invalid:
                return f"Invalid columns: {', '.join(invalid)}"

            # Store selection
            session.selections["features"] = feature_names

            # Transition to model confirmation
            success, error_msg, missing = await self.state_manager.transition_state(
                session,
                MLTrainingState.CONFIRMING_MODEL.value
            )

            if success:
                return (
                    f"Features selected: {', '.join(feature_names)}\n\n"
                    "Choose model type:\n"
                    "1. Neural Network\n"
                    "2. Random Forest\n"
                    "3. Gradient Boosting"
                )
            else:
                return f"Error: {error_msg}"

        elif current_state == MLTrainingState.CONFIRMING_MODEL.value:
            # Parse model type selection
            model_map = {
                "1": "neural_network",
                "2": "random_forest",
                "3": "gradient_boosting",
                "neural network": "neural_network",
                "random forest": "random_forest",
                "gradient boosting": "gradient_boosting"
            }

            model_type = model_map.get(message_text.lower().strip())
            if not model_type:
                return "Invalid selection. Choose 1, 2, or 3."

            # Store selection
            session.selections["model_type"] = model_type

            # Transition to training
            success, error_msg, missing = await self.state_manager.transition_state(
                session,
                MLTrainingState.TRAINING.value
            )

            if success:
                return f"Model type '{model_type}' selected. Training will begin..."
            else:
                return f"Error: {error_msg}"

        return f"State '{current_state}' handler not implemented."

    async def get_session_info(self, session: UserSession) -> Dict[str, Any]:
        """Get session information for diagnostics."""
        return {
            "user_id": session.user_id,
            "conversation_id": session.conversation_id,
            "workflow_type": session.workflow_type.value if session.workflow_type else None,
            "current_state": session.current_state,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "message_count": len(session.history),
            "has_data": session.uploaded_data is not None,
            "data_shape": session.uploaded_data.shape if session.uploaded_data is not None else None,
            "selections": session.selections,
            "model_ids": session.model_ids
        }

    async def cancel_workflow(self, session: UserSession) -> None:
        """Cancel active workflow and clear session data."""
        logger.info(f"Canceling workflow for session {session.session_key}")
        await self.state_manager.cancel_workflow(session)
        session.uploaded_data = None
        logger.debug("Workflow canceled successfully")
