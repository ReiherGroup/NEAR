#!/usr/bin/python3

"""
An example demonstrating the usage of the `near_score` module.
"""

import functools
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from near_score import get_near_score, estimate_layer_size


class NeuralNetwork(nn.Module):
    """
    A neural network model class that constructs a feedforward neural network with specified number of layers.
    """
    def __init__(self, layers):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential()

        for i in range(len(layers) - 2):
            self.layer_stack.append(nn.Linear(layers[i], layers[i + 1], bias=True))
            self.layer_stack.append(nn.SiLU())
        self.layer_stack.append(nn.Linear(layers[-2], layers[-1], bias=True))

    def forward(self, x):
        """
        The forward pass of the neural network.
        """
        x = self.flatten(x)
        logits = self.layer_stack(x)
        return logits


def get_mnist_dataloader(max_class=10):
    # pylint: disable-msg=redefined-outer-name
    """
    Creates a DataLoader for the MNIST dataset, filtered to include only the specified number of classes.
    """
    training_data = datasets.MNIST(
        root="~/Documents/Torch_Dataset",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    idx = functools.reduce(
        torch.logical_or, [(training_data.targets == i) for i in range(max_class)]
    )
    training_data.targets = training_data.targets[idx]
    training_data.data = training_data.data[idx]
    return DataLoader(
        training_data,
        batch_size=1,
        shuffle=True,
    )


# Set the seed of the random number generator to make the example deterministic.
torch.manual_seed(42)
# Create a DataLoader for the MNIST dataset containing all 10 classes.
train_dataloader = get_mnist_dataloader(10)

# Create a neural network model with 784 input units, 100 hidden units, and 10 output units.
model = NeuralNetwork([784, 100, 10])

# Calculate the model's NEAR score as the average of 10 calculations with different initialized networks.
# With `layer_index=None` the NEAR score will be calculated for all layers
# and summed up to get the NEAR score of the model.
score = get_near_score(model, train_dataloader, repetitions=10, layer_index=None)
print(f"A model with the architecture 784-100-10 has a NEAR score of {round(score)} on MNIST with 10 classes.")

# The different layer sizes used to estimate the layer size.
sizes = [10 * 2**i for i in range(8)]
# Create a list of neural network models with different numbers of units in the hidden layer.
models = [NeuralNetwork([784, size, 10]) for size in sizes]

# Estimate the optimal layer size of the neural network for the MNIST dataset with 2, 5, and 10 different classes.
for max_class in [2, 5, 10]:
    train_dataloader = get_mnist_dataloader(max_class)
    estimated_size = estimate_layer_size(
        models,
        sizes,
        train_dataloader,
        layer_index=1,
        slope_threshold=0.01,
        repetitions=5,
        show_fit=True,
    )
    print(
        f"The optimal size for layer 2 when training {max_class} MNIST classes is estimated to be "
        f"{round(estimated_size)}."
    )
