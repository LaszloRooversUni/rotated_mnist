# model/evaluate.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from cnn import SimpleCNN
from PIL import Image
import os
from torchvision.datasets import ImageFolder


def get_loader(angle, batch_size=64):
    path = f"../data/rotated_{angle}"
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def evaluate(model, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    model.eval()
    model.to(device)

    for angle in [0, 30, 60, 90]:
        if angle == 0:
            # Evaluate on normal MNIST test set
            from torchvision import datasets

            transform = transforms.Compose([transforms.ToTensor()])
            testset = datasets.MNIST(
                root="../data", train=False, download=True, transform=transform
            )
            testloader = DataLoader(testset, batch_size=64, shuffle=False)
        else:
            testloader = get_loader(angle)

        acc = evaluate(model, testloader, device)
        print(f"Accuracy on {angle}° rotated test set: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
