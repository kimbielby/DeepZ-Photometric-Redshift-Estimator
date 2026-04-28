"""
test_metrics.py

Unit tests for evaluation metrics.
"""
import numpy as np
import pytest
from pzest.evaluation.metrics import sigma_nmad, bias, outlier_rate, crps, pit

@pytest.fixture
def perfect_predictions():
    z = np.linspace(0.1, 2.0, 100)
    return z, z

@pytest.fixture
def bin_edges():
    return np.linspace(0.0, 4.0, 201)

def test_sigma_nmad_perfect(perfect_predictions):
    z_pred, z_true = perfect_predictions
    assert sigma_nmad(z_pred, z_true) == pytest.approx(0.0, abs=1e-10)

def test_sigma_nmad_positive():
    z_true = np.linspace(0.1, 2.0, 100)
    z_pred = z_true + 0.05
    assert sigma_nmad(z_pred, z_true) > 0.0

def test_bias_perfect(perfect_predictions):
    z_pred, z_true = perfect_predictions
    assert bias(z_pred, z_true) == pytest.approx(0.0, abs=1e-10)

def test_bias_sign():
    z_true = np.linspace(0.1, 2.0, 100)
    z_pred = z_true + 0.1
    assert bias(z_pred, z_true) > 0.0

    z_pred = z_true - 0.1
    assert bias(z_pred, z_true) < 0.0

def test_outlier_rate_perfect(perfect_predictions):
    z_pred, z_true = perfect_predictions
    assert outlier_rate(z_pred, z_true) == pytest.approx(0.0)

def test_outlier_rate_all_outliers():
    z_true = np.ones(100) * 0.5
    z_pred = np.ones(100) * 3.0
    assert outlier_rate(z_pred, z_true) == pytest.approx(1.0)

def test_outlier_rate_custom_threshold():
    z_true = np.ones(100) * 0.5
    z_pred = z_true + 0.1 * (1 + z_true)
    assert outlier_rate(z_pred, z_true, threshold=0.05) > 0.0
    assert outlier_rate(z_pred, z_true, threshold=0.15) == pytest.approx(0.0)

def test_crps_perfect(bin_edges):
    n = 50
    z_true = np.linspace(0.1, 3.9, n)
    true_bins = np.searchsorted(bin_edges[1:-1], z_true)
    num_bins = len(bin_edges) - 1
    pdfs = np.zeros((n, num_bins))
    for i, b in enumerate(true_bins):
        pdfs[i, b] = 1.0

    assert crps(pdfs, z_true, bin_edges) == pytest.approx(0.0, abs=1e-6)

def test_crps_positive(bin_edges):
    n = 50
    z_true = np.linspace(0.1, 3.9, n)
    num_bins = len(bin_edges) - 1
    pdfs = np.ones((n, num_bins)) / num_bins
    assert crps(pdfs, z_true, bin_edges) > 0.0

def test_pit_perfect_calibration(bin_edges):
    """
    Perfect predictions should give PIT values uniformly in [0, 1].
    """
    n = 200
    z_true = np.linspace(0.1, 3.9, n)
    true_bins = np.searchsorted(bin_edges[1:-1], z_true)
    num_bins = len(bin_edges) - 1
    pdfs = np.zeros((n, num_bins))
    for i, b in enumerate(true_bins):
        pdfs[i, b] = 1.0

    pit_values = pit(pdfs, z_true, bin_edges)
    assert pit_values.shape == (n,)
    assert np.all(pit_values >= 0.0)
    assert np.all(pit_values <= 1.0)
