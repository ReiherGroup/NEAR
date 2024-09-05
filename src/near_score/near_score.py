#!/usr/bin/python3

"""
Implementation of Network Expressivity by Activation Rank (NEAR)
"""

import torch
import scipy
import numpy as np
import matplotlib.pyplot as plt


def __reset_weights(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()


def __power_function(x, a, b, c):
    return a + b * x**c


def get_effective_rank(matrix, return_singular_values=False):
    """
    Calculates the effective rank of a matrix.

    Args:
        matrix (torch.Tensor): Input matrix.
        return_singular_values (bool, optional): If True, also returns the singular values. Default is False.

    Returns:
        float or tuple: Effective rank of the matrix. If `return_singular_values` is True, returns a tuple
        containing the effective rank and the singular values.
    """
    s = torch.linalg.svdvals(matrix)  # pylint: disable-msg=not-callable
    if return_singular_values:
        singular_values = s.detach().clone()
    s /= torch.sum(s)
    erank = torch.e ** scipy.stats.entropy(s.detach())
    if return_singular_values:
        return np.nan_to_num(erank), singular_values
    return np.nan_to_num(erank)


def __get_near_score(model, dataloader, layer_index=None):
    # pylint: disable-msg=too-many-locals
    # pylint: disable-msg=unused-argument
    """
    Calculates the NEAR score of a given neural network.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for input data.
        layer_index (int, optional): The index of the layer for which to calculate the NEAR score.
                                     If None, calculate the NEAR score for the whole model.

    Returns:
        float: The NEAR score.
    """
    hooks = []
    activations = []
    finished = []
    called = set()

    def get_activation(layer_id):
        def hook(model, input, output):
            # pylint: disable-msg=redefined-builtin
            called.add(layer_id)
            if isinstance(output, tuple):
                output = output[0]
            size = output.shape[-1]
            if output.dim() > 2:
                output = torch.transpose(output, 1, 3).flatten(0, 2)
            activations[layer_id] = torch.cat((activations[layer_id], output), dim=0)
            if activations[layer_id].shape[0] >= activations[layer_id].shape[1]:
                start = (
                    np.random.randint(0, activations[layer_id].shape[0] // size - 1) * size
                    if activations[layer_id].shape[0] // size - 1 > 0
                    else 0
                )
                end = start + activations[layer_id].shape[1]
                activations[layer_id] = activations[layer_id][start:end]
                hooks[layer_id].remove()
                finished.append(layer_id)

        return hook

    activation_functions = tuple(
        getattr(torch.nn, fct) for fct in torch.nn.modules.activation.__all__
    )
    layer_stack = [
        module
        for name, module in model.named_modules()
        if hasattr(module, "weight") or isinstance(module, activation_functions)
    ]
    if layer_index is not None:
        layer_stack = [layer_stack[layer_index]]
    for layer_id, layer in enumerate(layer_stack):
        activations.append(torch.tensor([]))
        hook = layer.register_forward_hook(get_activation(layer_id))
        hooks.append(hook)

    for X, _ in dataloader:  # pylint: disable=invalid-name
        model(X)
        if len(finished) == len(called):
            break

    score = 0.0
    for activation in activations:
        if len(activation) == 0:
            continue
        score += get_effective_rank(activation)
    return score


def get_near_score(model, dataloader, layer_index=None, repetitions=1):
    """
    Calculates the average NEAR score.

    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for input data.
        layer_index (int, optional): The index of the layer for which to calculate the NEAR score.
                                     If None, calculate the NEAR score for the whole model.
        repetitions (int, optional): Number of repetitions to compute average score. Default is 1.

    Returns:
        float: The average NEAR score.
    """
    scores = []
    for _ in range(repetitions):
        scores.append(__get_near_score(model, dataloader, layer_index))
        model.apply(__reset_weights)
    return np.mean(scores)


def estimate_layer_size(
    models, sizes, dataloader, layer_index, slope_threshold=0.005, repetitions=1, show_fit=False
):
    # pylint: disable-msg=too-many-arguments
    """
    Estimates the optimal size of a specific layer.

    This function calculates the average NEAR score for the layers at layer_index,
    normalizes these scores by the corresponding sizes, and fits a curve to the normalized scores.
    The optimal layer size is then estimated based on the slope threshold.

    Args:
        models (list): A list of models to evaluate.
        sizes (list): A list of ints corresponding to the size of the layer at layer_index for each model.
        dataloader (DataLoader): A DataLoader object to provide the data for evaluation.
        layer_index (int): The index of the layer to evaluate.
        slope_threshold (float, optional): The threshold for the slope to determine the optimal size. Default is 0.005.
        repetitions (int, optional): The number of repetitions for averaging the NEAR score. Default is 1.
        show_fit (bool, optional): If True, displays a plot of the fitted power function. Default is False.

    Returns:
        float: The estimated optimal size for the specified layer.
    """
    scores = []
    for i, model in enumerate(models):
        scores.append(
            get_near_score(
                model, dataloader, repetitions=repetitions, layer_index=layer_index
            )
            / sizes[i]
        )
    popt, *_ = scipy.optimize.curve_fit(
        __power_function, [1, *sizes], [1, *scores], p0=(1, 2, -1), maxfev=10000
    )
    estimated_size = slope_threshold ** (1 / (popt[2] - 1))
    if show_fit:
        x = np.linspace(1, sizes[-1], 1000)
        _, ax = plt.subplots()
        ax.scatter([1, *sizes], [1, *scores])
        ax.set_xlabel("Number of Neurons")
        ax.set_ylabel("Relative NEAR Score")
        ax.plot(x, __power_function(x, *popt))
        ax.scatter(estimated_size, __power_function(estimated_size, *popt), label="Estimated size")
        ax.legend()
        plt.show()
    return estimated_size
