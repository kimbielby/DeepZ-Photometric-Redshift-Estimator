"""
test_loss.py

Unit tests for CRPSLoss.
"""
import torch
import pytest
from pzest.training.loss import CRPSLoss

NUM_BINS = 10

@pytest.fixture
def loss_fn():
    return CRPSLoss(num_bins=NUM_BINS)

def test_crps_perfect_prediction(loss_fn):
    """
    A PDF with all mass on the true bin should give CRPS of 0.
    """
    true_bin = 3
    pdf = torch.zeros(1, NUM_BINS)
    pdf[0, true_bin] = 1.0
    labels = torch.tensor([true_bin])

    loss = loss_fn(pdf, labels)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

def test_crps_worst_prediction(loss_fn):
    """
    A PDF with all mass at the opposite end from the true bin should give a high CRPS.
    """
    pdf = torch.zeros(1, NUM_BINS)
    pdf[0, 0] = 1.0
    labels = torch.tensor([NUM_BINS - 1])

    loss_far = loss_fn(pdf, labels)

    pdf_close = torch.zeros(1, NUM_BINS)
    pdf_close[0, NUM_BINS - 2] = 1.0
    loss_close = loss_fn(pdf_close, labels)

    assert loss_far.item() > loss_close.item()

def test_crps_batch(loss_fn):
    """
    Loss should be a scalar mean over the batch.
    """
    batch_size = 8
    pdf = torch.softmax(torch.randn(batch_size, NUM_BINS), dim=1)
    labels = torch.randint(0, NUM_BINS, (batch_size, ))

    loss = loss_fn(pdf, labels)
    assert loss.shape == torch.Size([])

def test_crps_differentiable(loss_fn):
    """
    Loss should be differentiable - backwards should not raise.
    """
    pdf = torch.softmax(
        torch.randn(4, NUM_BINS, requires_grad=False), dim=1
    ).requires_grad_(True)
    labels = torch.randint(0, NUM_BINS, (4, ))

    loss = loss_fn(pdf, labels)
    loss.backward()
    assert pdf.grad is not None

def test_crps_ordinal_penalty(loss_fn):
    """
    Being wrong by more bins should incur a higher CRPS than being wrong by fewer bins.
    """
    true_bin = 5

    pdf_close = torch.zeros(1, NUM_BINS)
    pdf_close[0, true_bin + 1] = 1.0

    pdf_far = torch.zeros(1, NUM_BINS)
    pdf_far[0, true_bin + 3] = 1.0

    labels = torch.tensor([true_bin])

    loss_close = loss_fn(pdf_close, labels)
    loss_far = loss_fn(pdf_far, labels)

    assert loss_far.item() > loss_close.item()
