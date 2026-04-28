"""
test_dataset.py

Unit tests for GalaxyDataset.
"""
import numpy as np
import pytest
import h5py
import tempfile
from pathlib import Path
from pzest.dataset.dataset import GalaxyDataset

@pytest.fixture
def temp_hdf5():
    """Create a minimal temporary HDF5 file for testing."""
    n = 20
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
        path = Path(f.name)

    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=np.random.randn(n, 5, 64, 64).astype(np.float32))
        f.create_dataset("redshift", data=np.random.uniform(0.1, 3.9, n))
        f.create_dataset("magnitudes", data=np.random.uniform(18, 25, (n, 5)))
        f.create_dataset("morphtype", data=np.array([b"SER"] * n))

    yield path

    for fid in list(h5py.h5f.get_obj_ids()):
        try:
            h5py.h5f.FileID(fid).close()
        except Exception:
            pass

@pytest.fixture
def bin_edges():
    return np.linspace(0.0, 4.0, 201)

@pytest.fixture
def dataset(temp_hdf5, bin_edges):
    indices = np.arange(20)
    return GalaxyDataset(
        hdf5_path=temp_hdf5,
        indices=indices,
        bin_edges=bin_edges,
        use_magnitudes=True,
    )

def test_len(dataset):
    assert len(dataset) == 20

def test_getitem_keys(dataset):
    sample = dataset[0]
    assert set(sample.keys()) == {"image", "label", "magnitudes"}

def test_image_shape(dataset):
    sample = dataset[0]
    assert sample["image"].shape == (5, 64, 64)

def test_label_valid(dataset, bin_edges):
    for i in range(len(dataset)):
        label = dataset[i]["label"].item()
        assert 0 <= label < len(bin_edges) - 1

def test_magnitudes_shape(dataset):
    sample = dataset[0]
    assert sample["magnitudes"].shape == (5,)

def test_subset_indices(temp_hdf5, bin_edges):
    """Dataset with a subset of indices should have correct length."""
    indices = np.array([0, 5, 10, 15])
    ds = GalaxyDataset(
        hdf5_path=temp_hdf5,
        indices=indices,
        bin_edges=bin_edges,
        use_magnitudes=True,
    )
    assert len(ds) == 4
