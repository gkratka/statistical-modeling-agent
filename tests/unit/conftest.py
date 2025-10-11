"""Shared fixtures for local path tests."""

import pytest
import pandas as pd
from pathlib import Path

from src.core.state_manager import StateManager
from src.processors.data_loader import DataLoader


@pytest.fixture
def config_enabled(tmp_path):
    return {
        'local_data': {
            'enabled': True,
            'allowed_directories': [str(tmp_path)],
            'max_file_size_mb': 10,
            'allowed_extensions': ['.csv', '.xlsx', '.xls']
        }
    }


@pytest.fixture
def config_disabled():
    return {
        'local_data': {
            'enabled': False,
            'allowed_directories': [],
            'max_file_size_mb': 10,
            'allowed_extensions': ['.csv']
        }
    }


@pytest.fixture
def sample_csv(tmp_path):
    csv_file = tmp_path / "sample.csv"
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [100, 200, 150, 300, 250]
    })
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def state_manager():
    return StateManager()


@pytest.fixture
def data_loader_enabled(config_enabled):
    return DataLoader(config=config_enabled)
