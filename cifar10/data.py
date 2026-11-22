"""Готовит батчи CIFAR-10 для обучения, валидации и итоговой оценки."""

from copy import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VALIDATION_SIZE = 5_000
NORMALIZE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        NORMALIZE,
    ]
)
# Случайная обрезка и отражение применяются заново при каждом чтении примера.
TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)


def _split_indices(size: int, seed: int) -> tuple[list[int], list[int]]:
    """Возвращает индексы обучения и валидации для заданного seed."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(size, generator=generator).tolist()
    return indices[VALIDATION_SIZE:], indices[:VALIDATION_SIZE]


def make_train_val_loaders(
    batch_size: int = 128, seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """Воспроизводимо делит выборку на обучающую и валидационную."""
    train_data = CIFAR10(root=DATA_DIR, train=True, transform=TRAIN_TRANSFORM)
    # Используем общие изображения и метки с отдельными преобразованиями для валидации
    val_data = copy(train_data)
    val_data.transform = EVAL_TRANSFORM

    train_indices, val_indices = _split_indices(len(train_data), seed)

    train_loader = DataLoader(
        Subset(train_data, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        Subset(val_data, val_indices),
        batch_size=batch_size,
    )
    return train_loader, val_loader


def make_calibration_loader(
    batch_size: int = 128,
    seed: int = 42,
    sample_size: int = 1_024,
) -> DataLoader:
    """Берёт подвыборку из обучающих индексов без случайных аугментаций."""
    dataset = CIFAR10(root=DATA_DIR, train=True, transform=EVAL_TRANSFORM)
    train_indices, _ = _split_indices(len(dataset), seed)
    if not 1 <= sample_size <= len(train_indices):
        raise ValueError("Размер калибровочной выборки должен быть от 1 до размера обучающей.")
    return DataLoader(
        Subset(dataset, train_indices[:sample_size]),
        batch_size=batch_size,
    )


def make_test_loader(batch_size: int = 128) -> DataLoader:
    """Загружает тестовую выборку для итоговой оценки."""
    test_data = CIFAR10(root=DATA_DIR, train=False, transform=EVAL_TRANSFORM)
    return DataLoader(
        test_data,
        batch_size=batch_size,
    )
