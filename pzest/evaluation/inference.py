"""
inference.py

Running DeepZ on galaxy images to produce redshift PDFs and point estimates.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def predict(
        model: nn.Module,
        loader: DataLoader,
        bin_edges: np.ndarray,
        device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model inference over a DataLoader and return PDFs and point estimates.

    Point estimates are computed as the weighted mean of bin centres,
    where weights are the predicted softmax probabilities.

    Args:
        model: Trained DeepZ model in eval mode
        loader: DataLoader to run inference on
        bin_edges: Array of redshift bin edges, shape (num_bins + 1)
        device: Device to run inference on

    Returns:
        Tuple of:
            pdfs: Predicted PDFs, shape (N, num_bins), float32
            point_estimates: Weighted mean redshift per galaxy, shape (N,)
    """
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    all_pdfs = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            magnitudes = batch["magnitudes"].to(device)

            pdf = model(image, magnitudes)
            all_pdfs.append(pdf.cpu().numpy())

    pdfs = np.concatenate(all_pdfs, axis=0)
    point_estimates = (pdfs * bin_centres).sum(axis=1)

    return pdfs, point_estimates

def predict_from_arrays(
        model: nn.Module,
        images: np.ndarray,
        bin_edges: np.ndarray,
        device: torch.device,
        magnitudes: np.ndarray | None = None,
        batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on raw numpy arrays without requiring a DataLoader.

    Args:
        model: Trained DeepZ model
        images: Preprocessed images, shape (N, 5, 64, 64), float32
        bin_edges: Redshift bin edges, shape (num_bins + 1,)
        device: Device to run on
        magnitudes: Optional magnitudes, shape (N, 5), float32
        batch_size: Number of galaxies per batch

    Returns:
        Tuple of (pdfs, point_estimates), shapes (N, num_bins) and (N,)
    """
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n = len(images)
    all_pdfs = []

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            img_batch = torch.tensor(
                images[start:end],
                dtype=torch.float32,
            ).to(device)

            mag_batch = None
            if magnitudes is not None:
                mag_batch = torch.tensor(
                    magnitudes[start:end].astype(np.float32),
                    dtype=torch.float32,
                ).to(device)

            pdf = model(img_batch, mag_batch)
            all_pdfs.append(pdf.cpu().numpy())

    pdfs = np.concatenate(all_pdfs, axis=0)
    point_estimates = (pdfs * bin_centres).sum(axis=1)

    return pdfs, point_estimates
