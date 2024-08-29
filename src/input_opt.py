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
    with open("weights.pth", "rb") as file:
        weights = torch.load(file)

    net = CNN()
    net.load_state_dict(weights)
    neuron = 3

    def forward_pass(x):
        """Make single forward pass."""
        out = net(x)
        return out[0, neuron]

    get_grads = grad(forward_pass)

    x = torch.ones([1, 1, 28, 28])

    grads = []
    for _i in range(10):
        x = (x - torch.mean(x)) / torch.std(x + 1e-5)
        grad_vals = get_grads(x)
        x = x + 0.1 * grad_vals
        grads.append(grad_vals)
        print(forward_pass(x).detach().item())

    mean_grad = torch.mean(torch.stack(grads, 0), 0)

    plt.imshow(x[0, 0, :, :].detach().numpy())
    plt.title("Input maxizing the " + str(neuron) + "- neuron")
    plt.savefig("input_opt.jpg")
