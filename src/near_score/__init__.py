#!/usr/bin/python3

"""
Network Expressivity by Activation Rank (NEAR) module

This module provides functions for analyzing neural networks, including calculating the effective rank of a matrix,
the NEAR score of a neural network, and estimating the optimal size of a specific layer.

Functions
---------
get_effective_rank(matrix, return_singular_values=False)
    Calculates the effective rank of a matrix.

get_near_score(model, dataloader, layer_index=None, repetitions=1)
    Calculates the NEAR score of a given neural network.

estimate_layer_size(models, sizes, dataloader, layer_index, slope_threshold=0.005, repetitions=1)
    Estimates the optimal size of a specific layer.

"""

__copyright__ = """This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details."""

from ._version_ import __version__
from .near_score import get_effective_rank, get_near_score, estimate_layer_size
