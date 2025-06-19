# model/train.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn import SimpleCNN
import torch.nn as nn
import torch.optim as optim


def main():
    """Main function to train a simple CNN on the MNIST dataset."""
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.MNIST(
        root="../data", train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Model saved as mnist_cnn.pth")


if __name__ == "__main__":
    main()
