import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """A simple Convolutional Neural Network (CNN) for image classification.
    This model consists of two convolutional layers followed by two fully connected layers.
    It is designed to work with grayscale images of size 28x28, such as those
    found in the MNIST dataset.
    """

    def __init__(self):
        """Initializes the SimpleCNN model."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Defines the forward pass of the model."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
