"""
preprocessing.py

Preprocessing for the GalaxiesML-Spectra dataset.
Applies arcsinh stretch and per-channel standardisation to images, filters
invalid galaxies and writes a clean HDF5 to data/processed

Called from notebooks/02_preprocess.ipynb
"""
import h5py
import numpy as np
from pzest.config import (
    Config,
    IMAGE_ARCSINH_SCALE,
    IMAGE_CHANNEL_MEAN,
    IMAGE_CHANNEL_STD,
    MAGNITUDE_COLS
)

def _build_valid_mask(
        redshifts: np.ndarray,
        morphtypes: np.ndarray,
        include_psf: bool,
) -> np.ndarray:
    """
    Build a boolean mask of galaxies to keep.

    Excludes galaxies with redshift <= 0 and, optionally, PSF objects.

    Args:
        redshifts: Array of DESI spectroscopic redshifts
        morphtypes: Array of DESI morphology type byte strings
        include_psf: Whether to keep PSF-classified objects

    Returns:
        Boolean mask of shape (N,). True for galaxies to keep.
    """
    mask = redshifts > 0

    if not include_psf:
        is_psf = np.array(
            [m.decode().strip() == "PSF" for m in morphtypes]
        )
        mask &= ~is_psf

    return mask

def _preprocess_images(
        batch: np.ndarray,
) -> np.ndarray:
    """
    Apply arcsinh stretch and per-channel standardisation to a batch of images.

    Pipeline per channel:
        1. arcsinh(x / scale) - compresses dynamic range
        2. (x - mean) / std - standardises to zero mean, unit variance

    Args:
        batch: Raw flux images of shape (N, 5, 64, 64), float32

    Returns:
        Preprocessed images of shape (N, 5, 64, 64), float32
    """
    arcsinh_scale = np.array(IMAGE_ARCSINH_SCALE, dtype=np.float32)
    channel_mean = np.array(IMAGE_CHANNEL_MEAN, dtype=np.float32)
    channel_std = np.array(IMAGE_CHANNEL_STD, dtype=np.float32)

    batch = batch.astype(np.float32)

    for c in range(5):
        batch[:, c] = np.arcsinh(batch[:, c] / arcsinh_scale[c])
        batch[:, c] = (batch[:, c] - channel_mean[c]) / channel_std[c]

    return batch

def preprocess(
        config: Config,
) -> None:
    """
    Run the full preprocessing pipeline.

    Reads the raw HDF5, filters invalid galaxies, preprocesses images and
    writes a clean HDF5 to config.paths.processed_hdf5_file.

    Args:
        config: Fully populated Config object.
    """
    raw_path = config.data.hdf5_path
    out_path = config.paths.processed_hdf5_file
    chunk_size = config.data.chunk_size

    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Raw data: {raw_path}")
    print(f"Output: {out_path}")

    # --- Load metadata needed for filtering ---
    print("\nLoading metadata...")
    with h5py.File(raw_path, "r") as f:
        redshifts = f["DESI_fibermap"]["DESI_redshift"][:]
        morphtypes = f["DESI_fibermap"]["MORPHTYPE"][:]
        object_ids = f["object_id"][:]
        magnitudes = np.stack(
            [f["HSC_metadata"][col][:] for col in MAGNITUDE_COLS],
            axis=1,
        )

    n_total = len(redshifts)
    print(f"Total galaxies in raw file: {n_total:,}")

    # --- Build valid mask and report filtering ---
    mask = _build_valid_mask(redshifts, morphtypes, include_psf=config.data.include_psf)
    valid_indices = np.where(mask)[0]
    n_valid = len(valid_indices)

    print("\nFiltering:")
    print(f"Removed (redshift <= 0): {(redshifts <= 0).sum():,}")
    if not config.data.include_psf:
        is_psf = np.array(
            [m.decode().strip() == "PSF" for m in morphtypes]
        )
        print(f"Removed (PSF): {is_psf.sum():,}")
    print(f"Remaining: {n_valid:,}")

    # --- Write processed HDF5 ---
    print("\nProcessing and writing...")

    with (h5py.File(raw_path, "r") as f_in,
          h5py.File(out_path, "w") as f_out):

        f_out.create_dataset(
            "images",
            shape=(n_valid, 5, 64, 64),
            dtype=np.float32,
            chunks=(min(chunk_size, n_valid), 5, 64, 64),
        )
        f_out.create_dataset(
            "object_id",
            data=object_ids[mask],
        )
        f_out.create_dataset(
            "redshift",
            data=redshifts[mask],
            dtype=np.float64,
        )
        f_out.create_dataset(
            "magnitudes",
            data=magnitudes[mask],
            dtype=np.float64,
        )
        f_out.create_dataset(
            "morphtype",
            data=morphtypes[mask],
        )

        n_chunks = int(np.ceil(n_valid / chunk_size))
        out_idx = 0

        for chunk_num, start in enumerate(
            range(0, n_valid, chunk_size),
            start=1,
        ):
            idx = valid_indices[start:start + chunk_size]
            batch = f_in["image"][np.sort(idx)]
            batch = _preprocess_images(batch)

            end_idx = out_idx + len(batch)
            f_out["images"][out_idx:end_idx] = batch
            out_idx = end_idx

            if chunk_num % 10 == 0 or chunk_num == n_chunks:
                print(f"Chunk {chunk_num}/{n_chunks} ({100*chunk_num/n_chunks:.0f}%)")

    print(f"\nDone. {n_valid:,} galaxies written to {out_path}")
