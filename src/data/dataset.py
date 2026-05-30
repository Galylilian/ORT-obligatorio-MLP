import os
from torchvision import datasets
from torch.utils.data import DataLoader

from src.core.preprocessing.transforms import (
    get_train_transforms,
    get_test_transforms
)


def is_valid_file(path):
    return path.lower().endswith((".jpg", ".jpeg", ".png"))


def get_dataloaders(batch_size=32):
    """
    Crea dataloaders para train y test
    """

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )

    train_dir = os.path.join(BASE_DIR, "data", "processed", "train")
    test_dir = os.path.join(BASE_DIR, "data", "processed", "test")

    print("DEBUG train_dir:", train_dir)
    print("Contenido:", os.listdir(train_dir))

    train_data = datasets.ImageFolder(
        train_dir,
        transform=get_train_transforms(),
        is_valid_file=is_valid_file
    )

    test_data = datasets.ImageFolder(
        test_dir,
        transform=get_test_transforms(),
        is_valid_file=is_valid_file
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader