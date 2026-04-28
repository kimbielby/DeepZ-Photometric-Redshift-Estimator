"""
splits.py

Train/Validation/Test split generation for the galaxy dataset.
Splits are saved to a CSV file for reproducibility and reuse.
"""
import numpy as np
import pandas as pd
from pzest.config import Config

def make_splits(
        config: Config,
        num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random train, validation and test index splits and save to a
    CSV file at config.paths.splits_file

    The CSV is always overwritten to guarantee consistency with the current
    config seed.

    Args:
        config: Fully populated configuration object
        num_samples: Total number of samples in the processed dataset

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    rng = np.random.default_rng(config.data.random_seed)
    indices = rng.permutation(num_samples)

    num_test = int(np.floor(config.data.test_fraction * num_samples))
    num_val = int(np.floor(config.data.val_fraction * num_samples))

    test_indices = indices[:num_test]
    val_indices = indices[num_test:num_test + num_val]
    train_indices = indices[num_test + num_val:]

    # Build and save CSV
    labels = np.empty(num_samples, dtype=object)
    labels[train_indices] = "train"
    labels[val_indices] = "val"
    labels[test_indices] = "test"

    df = pd.DataFrame({
        "index": np.arange(num_samples),
        "split": labels,
    })

    config.paths.splits_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.paths.splits_file, index=False)

    print(f"Splits saved to {config.paths.splits_file}")
    print(f""
          f"train: {len(train_indices):,}, "
          f"val: {len(val_indices):,}, "
          f"test: {len(test_indices):,}")

    return train_indices, val_indices, test_indices

def load_splits(
        config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load split indices from the splits CSV file

    Args:
        config: Fully populated configuration object

    Returns:
        Tuple of (train_indices, val_indices, test_indices)

    Raises:
        FileNotFoundError: If splits file does not exist
    """
    if not config.paths.splits_file.exists():
        raise FileNotFoundError(
            f"Splits file not found at {config.paths.splits_file}. "
            f"Run pipelines/trainer.py to generate splits."
        )

    df = pd.read_csv(config.paths.splits_file)

    train_indices = df[df["split"] == "train"]["index"].to_numpy()
    val_indices = df[df["split"] == "val"]["index"].to_numpy()
    test_indices = df[df["split"] == "test"]["index"].to_numpy()

    return train_indices, val_indices, test_indices











