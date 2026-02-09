"""Script for input optimization."""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.func import grad


class CNN(nn.Module):
    """A CNN model."""

    def __init__(self):
        """Create a convolutional neural network."""
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(1152, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        """Run the forward pass."""
        x = f.relu(self.conv1(x))
        x = f.avg_pool2d(x, kernel_size=2, stride=2)
        x = f.relu(self.conv2(x))
        x = f.avg_pool2d(x, kernel_size=2, stride=2)
        x = f.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        x = f.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


if __name__ == "__main__":
    with open("./data/weights.pth", "rb") as file:
        weights = torch.load(file)

    net = CNN()
    net.load_state_dict(weights)
    neuron = 3  # Target neuron index

    def forward_pass(x):
        """Make single forward pass."""
        # 1.1 TODO: Compute and return the activation value of a single neuron.
        return 0.0

    get_grads = grad(forward_pass)

    # 1.2 TODO: Write an optimization loop to optimize an input image to maximize the output of the target neuron.

    # 1.3 TODO: Plot the optimized input image.
    # Compare the results from a random initialization and an initialization with ones.
