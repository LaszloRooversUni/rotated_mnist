from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image
import os

rotation_angles = [30, 60, 90]

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(project_root, "data")


def rotate_and_save() -> None:
    """Rotates MNIST images by specified angles and saves them in a structured directory."""
    dataset = MNIST(root="./", train=False, download=True)

    for angle in rotation_angles:
        angle_dir = os.path.join(output_dir, f"rotated_{angle}")
        os.makedirs(angle_dir, exist_ok=True)

        for i in range(len(dataset)):
            img, label = dataset[i]
            rotated_img = img.rotate(angle)
            label_dir = os.path.join(angle_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            rotated_img.save(os.path.join(label_dir, f"{i}.png"))


if __name__ == "__main__":
    print("Starting rotation and saving of MNIST images...")
    rotate_and_save()
    print("Rotation and saving completed.")
    print(f"Images saved in {output_dir} with rotation angles: {rotation_angles}")
    print("You can now use these images for training or testing your models.")
