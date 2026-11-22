"""Обучает модель-студента на CIFAR-10 по обычным меткам классов."""

from pathlib import Path

import torch

from cifar10.data import make_train_val_loaders
from cifar10.models import StudentCNN
from cifar10.train_teacher import train_model

EPOCHS = 10
LEARNING_RATE = 0.003
SEED = 42
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "student_baseline.pt"


def main() -> None:
    """Обучает студента и сохраняет веса лучшей эпохи по валидации."""
    torch.manual_seed(SEED)
    train_loader, val_loader = make_train_val_loaders(seed=SEED)
    train_model(
        model=StudentCNN(),
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        checkpoint_path=CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()
