#!/usr/bin/python3

"""
Unit tests for all the functions defined in the module `near_score`.
"""

import warnings
import pytest
import torch
import numpy as np
import matplotlib
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from hypothesis import given, settings, strategies as st
from src import near_score


@pytest.fixture(autouse=True)
def random():
    """
    Fixture that will set the numpy and torch random seed. Will be called before the tests run.
    """
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture(autouse=True)
def disable_plots():
    """
    Fixture that disables interactive plotting during tests.
    """
    warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive, and thus cannot be shown")
    matplotlib.use("agg")


@pytest.fixture
def random_dataloader():
    """
    A fixture that creates a random dataset and provides a DataLoader for it.
    """
    x = torch.rand(200, 3, 10, 10)
    y = torch.rand(200, 5)
    return DataLoader(TensorDataset(x, y), batch_size=1, shuffle=True)


@settings(max_examples=30, deadline=None)
@given(st.floats(min_value=-0.99, max_value=0.99))
def test_get_effective_rank_circulant(rho):
    """
    Test the `get_effective_rank` function from the `near_score` module by comparing the calculated rank of the matrix
    `a` with the rank calculated by the closed-form solution presented in [1].

    [1] O. Roy and M. Vetterli,
        The effective rank: A measure of effective dimensionality,
        in 15th European signal processing conference (2007) pp. 606-610.
    """
    a = torch.tensor([
        [1, rho, rho**2, rho],
        [rho, 1, rho, rho**2],
        [rho**2, rho, 1, rho],
        [rho, rho**2, rho, 1]
    ])
    h = (1 + abs(rho))/2 * np.log((1 + abs(rho))/2) + (1 - abs(rho))/2 * np.log((1 - abs(rho))/2)
    h *= -1
    effective_rank = np.exp(2 * h)
    singular_values = torch.tensor([(1 + abs(rho))**2, 1 - abs(rho)**2, 1 - abs(rho)**2, (1 - abs(rho))**2])
    calculated_rank, calculated_singular_values = near_score.get_effective_rank(a, return_singular_values=True)
    assert pytest.approx(calculated_rank) == effective_rank
    assert torch.allclose(calculated_singular_values, singular_values, rtol=1e-3)


@given(st.integers(min_value=1, max_value=10))
def test_get_effective_rank_identity(size):
    """
    Test the `get_effective_rank` function from the `near_score` module by comparing the calculated rank of identity
    matrices of different sizes to their expected value.
    """
    a = torch.eye(size)
    assert pytest.approx(near_score.get_effective_rank(a)) == size


def test_get_near_score(random_dataloader):
    # pylint: disable-msg=redefined-outer-name
    """
    Test the `get__near_score` function from the `near_score` module by comparing the computed NEAR score
    of the model against a predefined value.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 3, 5),
        nn.Flatten(),
        nn.SiLU(),
        nn.Linear(6*6*3, 50),
        nn.SiLU(),
        nn.Linear(50, 25),
        nn.SiLU(),
        nn.Linear(25, 5)
    )
    assert pytest.approx(near_score.get_near_score(model, random_dataloader, repetitions=5)) == 132.1655230906162


def test_estimate_layer_size(random_dataloader):
    # pylint: disable-msg=redefined-outer-name
    """
    Test the `estimate_layer_size` function from the `near_score` module by comparing the layer size
    of the model against a predefined value.
    """
    sizes = [10 * 2**i for i in range(9)]
    models = [
        nn.Sequential(
            nn.Conv2d(3, 3, 5),
            nn.Flatten(),
            nn.Tanh(),
            nn.Linear(6 * 6 * 3, size),
            nn.Tanh(),
            nn.Linear(size, 5),
            nn.Tanh(),
        )
        for size in sizes
    ]
    assert (
        pytest.approx(
            near_score.estimate_layer_size(
                models, sizes, random_dataloader, 4, show_fit=True
            ),
            1e-4,
        )
        == 24.280967850662
    )
