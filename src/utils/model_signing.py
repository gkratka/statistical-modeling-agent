"""
Model file signing utilities using HMAC-SHA256.

This module provides secure signing and verification for model files
to prevent arbitrary code execution from tampered pickle files.

Security Features:
    - HMAC-SHA256 signing for model integrity verification
    - Signing key loaded from MODEL_SIGNING_KEY environment variable (REQUIRED)
    - Signature stored in separate .sig file alongside model
    - Verification required before loading any model file

Usage:
    # Set MODEL_SIGNING_KEY environment variable first
    from src.utils.model_signing import sign_file, verify_file

    # Sign a model file after saving
    sign_file(Path("model.pkl"))

    # Verify before loading
    verify_file(Path("model.pkl"))  # Raises SecurityViolationError if invalid
"""

import hashlib
import hmac
import os
from pathlib import Path
from typing import Optional

from src.utils.exceptions import SecurityViolationError


def get_signing_key() -> bytes:
    """Get the model signing key from environment.

    Returns:
        bytes: The signing key encoded as UTF-8 bytes

    Raises:
        ValueError: If MODEL_SIGNING_KEY environment variable is not set
    """
    key = os.getenv('MODEL_SIGNING_KEY')
    if key is None:
        raise ValueError(
            "MODEL_SIGNING_KEY environment variable is required. "
            "Set it before using model signing functions."
        )
    return key.encode('utf-8')


def compute_file_signature(file_path: Path, key: Optional[bytes] = None) -> str:
    """Compute HMAC-SHA256 signature for a file.

    Args:
        file_path: Path to the file to sign
        key: Optional signing key (uses env var if not provided)

    Returns:
        str: Hexadecimal signature string
    """
    if key is None:
        key = get_signing_key()

    # Read file in chunks for memory efficiency
    h = hmac.new(key, digestmod=hashlib.sha256)

    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)

    return h.hexdigest()


def get_signature_path(file_path: Path) -> Path:
    """Get the path for the signature file.

    Args:
        file_path: Path to the model file

    Returns:
        Path: Path to the corresponding signature file
    """
    return file_path.with_suffix(file_path.suffix + '.sig')


def sign_file(file_path: Path, key: Optional[bytes] = None) -> Path:
    """Sign a file by creating an HMAC-SHA256 signature file.

    Args:
        file_path: Path to the file to sign
        key: Optional signing key (uses env var if not provided)

    Returns:
        Path: Path to the created signature file

    Raises:
        FileNotFoundError: If the file to sign doesn't exist
        ValueError: If MODEL_SIGNING_KEY is not set
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    signature = compute_file_signature(file_path, key)
    sig_path = get_signature_path(file_path)

    with open(sig_path, 'w') as f:
        f.write(signature)

    return sig_path


def verify_file(file_path: Path, key: Optional[bytes] = None) -> bool:
    """Verify a file's HMAC-SHA256 signature.

    Args:
        file_path: Path to the file to verify
        key: Optional signing key (uses env var if not provided)

    Returns:
        bool: True if signature is valid

    Raises:
        SecurityViolationError: If signature is missing, invalid, or file is tampered
        FileNotFoundError: If the file to verify doesn't exist
        ValueError: If MODEL_SIGNING_KEY is not set
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sig_path = get_signature_path(file_path)

    # Check if signature file exists
    if not sig_path.exists():
        raise SecurityViolationError(
            f"Model file is unsigned: {file_path}. "
            "Signature file not found. Cannot load unsigned models.",
            violations=["unsigned_model"]
        )

    # Read stored signature
    with open(sig_path, 'r') as f:
        stored_signature = f.read().strip()

    # Compute current signature
    computed_signature = compute_file_signature(file_path, key)

    # Use constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(stored_signature, computed_signature):
        raise SecurityViolationError(
            f"Model file has invalid signature: {file_path}. "
            "The file may have been tampered with. Cannot load.",
            violations=["tampered_model"]
        )

    return True


def sign_model_directory(model_dir: Path, key: Optional[bytes] = None) -> list:
    """Sign all pickle files in a model directory.

    Args:
        model_dir: Path to the model directory
        key: Optional signing key (uses env var if not provided)

    Returns:
        list: List of created signature file paths
    """
    model_dir = Path(model_dir)
    sig_files = []

    # Sign all .pkl files
    for pkl_file in model_dir.glob("*.pkl"):
        sig_path = sign_file(pkl_file, key)
        sig_files.append(sig_path)

    return sig_files


def verify_model_directory(model_dir: Path, key: Optional[bytes] = None) -> bool:
    """Verify all pickle files in a model directory.

    Args:
        model_dir: Path to the model directory
        key: Optional signing key (uses env var if not provided)

    Returns:
        bool: True if all signatures are valid

    Raises:
        SecurityViolationError: If any signature is missing or invalid
    """
    model_dir = Path(model_dir)

    # Verify all .pkl files
    for pkl_file in model_dir.glob("*.pkl"):
        verify_file(pkl_file, key)

    return True
