"""
test_splits.py

Unit tests for make_splits and load_splits.
"""
import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from pzest.dataset.splits import make_splits, load_splits

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.data.random_seed = 42
    config.data.val_fraction = 0.15
    config.data.test_fraction = 0.15
    with tempfile.TemporaryDirectory() as tmpdir:
        config.paths.splits_file = Path(tmpdir) / "splits.csv"
        yield config

def test_split_sizes(mock_config):
    n = 1000
    train, val, test = make_splits(mock_config, n)

    assert len(test) == int(np.floor(0.15 * n))
    assert len(val) == int(np.floor(0.15 * n))
    assert len(train) == n - len(test) - len(val)

def test_no_overlap(mock_config):
    n = 1000
    train, val, test = make_splits(mock_config, n)

    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0

def test_covers_all_indices(mock_config):
    n = 1000
    train, val, test = make_splits(mock_config, n)

    all_indices = np.concatenate([train, val, test])
    assert set(all_indices) == set(range(n))

def test_reproducibility(mock_config):
    n = 1000
    train1, val1, test1 = make_splits(mock_config, n)
    train2, val2, test2 = make_splits(mock_config, n)

    np.testing.assert_array_equal(np.sort(train1), np.sort(train2))
    np.testing.assert_array_equal(np.sort(val1), np.sort(val2))
    np.testing.assert_array_equal(np.sort(test1), np.sort(test2))

def test_csv_saved(mock_config):
    make_splits(mock_config, 100)
    assert mock_config.paths.splits_file.exists()

def test_load_splits_roundtrip(mock_config):
    n = 1000
    train, val, test = make_splits(mock_config, n)
    train_l, val_l, test_l = load_splits(mock_config)

    np.testing.assert_array_equal(np.sort(train), np.sort(train_l))
    np.testing.assert_array_equal(np.sort(val), np.sort(val_l))
    np.testing.assert_array_equal(np.sort(test), np.sort(test_l))

def test_load_splits_missing_file(mock_config):
    with pytest.raises(FileNotFoundError):
        load_splits(mock_config)
