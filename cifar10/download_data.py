"""Скачивает и распаковывает CIFAR-10 в папку data проекта."""

from pathlib import Path

from torchvision.datasets import CIFAR10


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    train = CIFAR10(root=data_dir, train=True, download=True)
    test = CIFAR10(root=data_dir, train=False)
    print(f"CIFAR-10: {len(train)} train, {len(test)} test, {len(train.classes)} classes")


if __name__ == "__main__":
    main()
