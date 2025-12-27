"""User-facing messages for join workflow (/join command)."""

from typing import List, Optional
from telegram import InlineKeyboardButton, InlineKeyboardMarkup


class JoinMessages:
    """Consolidated message templates for join workflow."""

    # Operation display names and descriptions
    OPERATIONS = {
        "left_join": ("Left Join", "Keep all rows from first dataframe, match from others"),
        "right_join": ("Right Join", "Keep all rows from last dataframe, match from first"),
        "inner_join": ("Inner Join", "Keep only matching rows"),
        "outer_join": ("Outer Join", "Keep all rows from all dataframes"),
        "cross_join": ("Cross Join", "Cartesian product (all combinations)"),
        "union": ("Union", "Stack vertically (same columns required)"),
        "concat": ("Concat", "Stack vertically (handles different columns)"),
        "merge": ("Merge", "Pandas merge with configurable options"),
    }

    # Operations that require key columns
    JOIN_OPERATIONS = {"left_join", "right_join", "inner_join", "outer_join", "merge"}
    # Operations that don't require key columns
    STACK_OPERATIONS = {"union", "concat", "cross_join"}

    @staticmethod
    def join_intro_message() -> str:
        """Welcome message explaining the /join command."""
        return (
            "*Dataframe Join/Union Operations*\n\n"
            "This command allows you to combine multiple dataframes (2-4) using "
            "various operations:\n\n"
            "*Join Operations* (require key columns):\n"
            "- Left Join - Keep all rows from first dataframe\n"
            "- Right Join - Keep all rows from last dataframe\n"
            "- Inner Join - Only matching rows\n"
            "- Outer Join - All rows from all dataframes\n"
            "- Merge - Advanced pandas merge\n\n"
            "*Stack Operations* (no key required):\n"
            "- Union - Vertical stack (same columns)\n"
            "- Concat - Vertical stack (different columns OK)\n"
            "- Cross Join - Cartesian product\n\n"
            "Select an operation to continue:"
        )

    @staticmethod
    def operation_selection_prompt() -> str:
        """Prompt for selecting operation type."""
        return (
            "*Select Join Operation*\n\n"
            "Choose the type of operation to perform on your dataframes:"
        )

    @staticmethod
    def dataframe_count_prompt(operation: str) -> str:
        """Prompt for selecting number of dataframes."""
        op_name = JoinMessages.OPERATIONS.get(operation, (operation, ""))[0]
        return (
            f"*{op_name}*\n\n"
            "How many dataframes do you want to combine?\n\n"
            "_Note: 2 is most common for simple operations_"
        )

    @staticmethod
    def dataframe_source_prompt(df_num: int, total: int) -> str:
        """Prompt for upload vs local path selection for a dataframe."""
        return (
            f"*Dataframe {df_num} of {total}*\n\n"
            f"How would you like to provide dataframe {df_num}?"
        )

    @staticmethod
    def file_path_input_prompt(
        df_num: int,
        allowed_dirs: List[str],
        allowed_extensions: List[str] = None
    ) -> str:
        """Prompt for local file path input."""
        if allowed_extensions is None:
            allowed_extensions = [".csv", ".xlsx", ".xls", ".parquet"]

        dirs = "\n".join(f"  `{d}`" for d in allowed_dirs[:5])
        if len(allowed_dirs) > 5:
            dirs += f"\n  ... and {len(allowed_dirs) - 5} more"

        exts = ", ".join(allowed_extensions)

        return (
            f"*Local File Path - Dataframe {df_num}*\n\n"
            f"*Allowed directories:*\n{dirs}\n\n"
            f"*Supported formats:* {exts}\n\n"
            f"*Example:*\n"
            f"  `/path/to/your/data.csv`\n\n"
            f"Type the full path to your file:"
        )

    @staticmethod
    def upload_file_prompt(df_num: int) -> str:
        """Prompt for file upload via Telegram."""
        return (
            f"*Upload File - Dataframe {df_num}*\n\n"
            f"Please upload dataframe {df_num} as a file.\n\n"
            f"Supported formats: CSV, Excel (.xlsx, .xls), Parquet"
        )

    @staticmethod
    def file_loaded_message(
        df_num: int,
        file_path: str,
        row_count: int,
        columns: List[str]
    ) -> str:
        """Confirmation message after loading a dataframe."""
        cols_display = ", ".join(f"`{c}`" for c in columns[:8])
        if len(columns) > 8:
            cols_display += f" ... (+{len(columns) - 8} more)"

        return (
            f"*Dataframe {df_num} Loaded*\n\n"
            f"File: `{file_path}`\n"
            f"Rows: {row_count:,}\n"
            f"Columns: {cols_display}"
        )

    @staticmethod
    def key_column_selection_prompt(common_columns: List[str]) -> str:
        """Prompt for selecting join key column(s)."""
        cols_display = "\n".join(f"  `{c}`" for c in common_columns[:15])
        if len(common_columns) > 15:
            cols_display += f"\n  ... and {len(common_columns) - 15} more"

        return (
            "*Select Key Column(s)*\n\n"
            "Choose the column(s) to join on. These columns must exist in all dataframes.\n\n"
            f"*Common columns:*\n{cols_display}\n\n"
            "Select one or more key columns, then click 'Confirm Keys':"
        )

    @staticmethod
    def output_path_prompt(default_path: str) -> str:
        """Prompt for output path selection."""
        return (
            "*Output Location*\n\n"
            f"*Default path:*\n  `{default_path}`\n\n"
            "Choose where to save the result:"
        )

    @staticmethod
    def custom_output_path_prompt(allowed_dirs: List[str]) -> str:
        """Prompt for custom output path input."""
        dirs = "\n".join(f"  `{d}`" for d in allowed_dirs[:5])
        if len(allowed_dirs) > 5:
            dirs += f"\n  ... and {len(allowed_dirs) - 5} more"

        return (
            "*Custom Output Path*\n\n"
            f"*Allowed directories:*\n{dirs}\n\n"
            "*Example:*\n"
            "  `/path/to/output/joined_data.csv`\n\n"
            "Type the full path for the output file:"
        )

    @staticmethod
    def executing_message(operation: str, df_count: int) -> str:
        """Progress message during execution."""
        op_name = JoinMessages.OPERATIONS.get(operation, (operation, ""))[0]
        return (
            f"*Executing {op_name}*\n\n"
            f"Combining {df_count} dataframes...\n\n"
            "This may take a moment for large files."
        )

    @staticmethod
    def complete_message(
        output_path: str,
        row_count: int,
        column_count: int,
        columns: List[str],
        operation: str
    ) -> str:
        """Success message after join completion."""
        op_name = JoinMessages.OPERATIONS.get(operation, (operation, ""))[0]
        cols_display = ", ".join(f"`{c}`" for c in columns[:10])
        if len(columns) > 10:
            cols_display += f" ... (+{len(columns) - 10} more)"

        return (
            f"*{op_name} Complete!*\n\n"
            f"*Output file:*\n  `{output_path}`\n\n"
            f"*Result:*\n"
            f"  Rows: {row_count:,}\n"
            f"  Columns: {column_count}\n\n"
            f"*Columns:*\n  {cols_display}"
        )

    @staticmethod
    def error_message(error: str) -> str:
        """Error message for join failures."""
        return (
            "*Join Operation Failed*\n\n"
            f"Error: {error}\n\n"
            "Please check your files and try again."
        )

    @staticmethod
    def no_common_columns_error(df1_cols: List[str], df2_cols: List[str]) -> str:
        """Error when dataframes have no common columns for join."""
        return (
            "*No Common Columns*\n\n"
            "The dataframes have no columns in common for joining.\n\n"
            f"*Dataframe 1 columns:* {', '.join(df1_cols[:5])}\n"
            f"*Dataframe 2 columns:* {', '.join(df2_cols[:5])}\n\n"
            "Consider using 'Concat' instead, or ensure dataframes have matching column names."
        )

    @staticmethod
    def path_validation_error(error_type: str, path: str, details: str = None) -> str:
        """Error message for path validation failures."""
        base_msg = f"*Path Validation Error*\n\nPath: `{path}`\n\n"

        if error_type == "not_found":
            return base_msg + "File not found. Please check the path and try again."
        elif error_type == "not_in_whitelist":
            return base_msg + "This directory is not in the allowed list."
        elif error_type == "invalid_extension":
            return base_msg + "Unsupported file format. Use CSV, Excel, or Parquet."
        elif error_type == "file_too_large":
            return base_msg + f"File is too large. {details or ''}"
        else:
            return base_msg + (details or "Invalid path.")


# Button creation helpers

def create_operation_buttons() -> List[List[InlineKeyboardButton]]:
    """Create buttons for operation selection."""
    return [
        [InlineKeyboardButton("Left Join", callback_data="join_op_left_join")],
        [InlineKeyboardButton("Right Join", callback_data="join_op_right_join")],
        [InlineKeyboardButton("Inner Join", callback_data="join_op_inner_join")],
        [InlineKeyboardButton("Outer Join", callback_data="join_op_outer_join")],
        [InlineKeyboardButton("Cross Join", callback_data="join_op_cross_join")],
        [InlineKeyboardButton("Union", callback_data="join_op_union")],
        [InlineKeyboardButton("Concat", callback_data="join_op_concat")],
        [InlineKeyboardButton("Merge", callback_data="join_op_merge")],
    ]


def create_dataframe_count_buttons() -> List[List[InlineKeyboardButton]]:
    """Create buttons for dataframe count selection."""
    return [
        [
            InlineKeyboardButton("2 (most common)", callback_data="join_count_2"),
            InlineKeyboardButton("3", callback_data="join_count_3"),
            InlineKeyboardButton("4", callback_data="join_count_4"),
        ]
    ]


def create_dataframe_source_buttons(df_num: int) -> List[List[InlineKeyboardButton]]:
    """Create buttons for upload vs local path selection."""
    return [
        [InlineKeyboardButton("Upload File", callback_data=f"join_df_source_upload_{df_num}")],
        [InlineKeyboardButton("Use Local Path", callback_data=f"join_df_source_local_{df_num}")],
    ]


def create_key_column_buttons(
    columns: List[str],
    selected: List[str] = None
) -> List[List[InlineKeyboardButton]]:
    """Create toggle buttons for key column selection."""
    if selected is None:
        selected = []

    buttons = []
    for col in columns[:15]:  # Limit to 15 columns for UI
        is_selected = col in selected
        label = f"{'[x]' if is_selected else '[ ]'} {col}"
        buttons.append([InlineKeyboardButton(label, callback_data=f"join_key_{col}")])

    # Add confirm button
    buttons.append([InlineKeyboardButton("Confirm Keys", callback_data="join_key_confirm")])

    return buttons


def create_output_path_buttons() -> List[List[InlineKeyboardButton]]:
    """Create buttons for output path selection."""
    return [
        [InlineKeyboardButton("Use Default Path", callback_data="join_output_default")],
        [InlineKeyboardButton("Custom Path...", callback_data="join_output_custom")],
    ]
