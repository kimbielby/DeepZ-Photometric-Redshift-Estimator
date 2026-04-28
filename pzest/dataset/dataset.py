"""
dataset.py

PyTorch Dataset for the preprocessed GalaxiesML-Spectra data.
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from pzest.config import Config

class GalaxyDataset(Dataset):
    """
    Dataset for preprocessed HSC galaxy images and DESI redshifts.

    Loads images and magnitudes lazily from the processed HDF5 file.
    Converts spectroscopic redshifts to bin indices for classification.
    """
    def __init__(
             self,
            hdf5_path: Path,
            indices: np.ndarray,
            bin_edges: np.ndarray,
            use_magnitudes: bool,
    ) -> None:
        """
        Args:
            hdf5_path: Path to the processed HDF5 file
            indices: Array of integer indices into the HDF5 dataset
            bin_edges: Array of redshift bin edges, shape (num_bins +1,)
            use_magnitudes: Whether to include magnitudes in returned samples
        """
        self.hdf5_path = Path(hdf5_path)
        self.indices = np.sort(indices)     # h5py requires sorted indices
        self.bin_edges = bin_edges
        self.use_magnitudes = use_magnitudes
        self._file = None

        # Load redshifts and magnitudes fully into memory
        with h5py.File(self.hdf5_path, "r") as f:
            self.redshifts = f["redshift"][self.indices]
            self.magnitudes = f["magnitudes"][self.indices].astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def _get_file(self) -> h5py.File:
        """ Return an open HDF5 file handle, opening it if necessary. """
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __getitem__(self, idx: int) -> dict:
        """
        Return a single sample.

        Args:
            idx: Position within this dataset (not the raw HDF5 index)

        Returns:
            Dict with keys:
                image: Float32 tensor of shape (5, 64, 64)
                label: Int64 scalar - redshift bin index
                magnitudes: Float32 tensor of shape (5,)
        """
        hdf5_idx = self.indices[idx]
        image = self._get_file()["images"][hdf5_idx]
        image = torch.tensor(image, dtype=torch.float32)

        redshift = self.redshifts[idx]
        label = np.searchsorted(self.bin_edges[1:-1], redshift)
        label = torch.tensor(label, dtype=torch.int64)

        magnitudes = torch.tensor(self.magnitudes[idx], dtype=torch.float32)

        return {
            "image": image,
            "label": label,
            "magnitudes": magnitudes
        }

    def __del__(self) -> None:
        """Close the HDF5 file handle when the dataset is garbage collected."""
        if self._file is not None:
            self._file.close()

def get_dataloader(
        hdf5_path: Path,
        indices: np.ndarray,
        bin_edges: np.ndarray,
        config: Config,
        shuffle: bool = False,
        sample_weights: np.ndarray | None = None,
) -> DataLoader:
    """
    Construct a DataLoader for a GalaxyDataset.

    Creates the GalaxyDataset internally from the provided parameters
    and wraps it in a DataLoader configured from config.

    Args:
        hdf5_path: Path to the processed HDF5 file
        indices: Array of integer indices into the HDF5 dataset
        bin_edges: Array of redshift bin edges, shape (num_bins +1,)
        config: Fully populated Config object
        shuffle: Whether to shuffle each epoch. Should be True for training, False for val/test
        sample_weights: Optional per-sample weights for weighted sampling.
            If provided, WeightedRandomSampler is used instead of shuffle.
            Should only be used for training.

    Returns:
        Configured DataLoader instance.
    """
    dataset = GalaxyDataset(
        hdf5_path=hdf5_path,
        indices=indices,
        bin_edges=bin_edges,
        use_magnitudes=config.model.use_magnitudes,
    )

    # If sample_weights are to be used, shuffle param will not be
    if sample_weights is not None:
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(dataset),
            replacement=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            sampler=sampler,
            num_workers=config.train.num_workers,
            pin_memory=True,
            persistent_workers=config.train.num_workers > 0,
        )

        return dataloader

    # If no sample_weights then return dataloader with shuffle param
    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=config.train.num_workers,
        pin_memory=True,
        persistent_workers=config.train.num_workers > 0,
    )

    return dataloader










