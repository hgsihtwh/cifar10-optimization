"""Оценивает студента после однократного прунинга и дообучает его."""

from pathlib import Path

import torch
from torch import nn

from cifar10.data import make_train_val_loaders
from cifar10.models import StudentCNN
from cifar10.pruning import prune_student
from cifar10.train_student import SEED
from cifar10.train_teacher import run_epoch, train_model

CHANNELS = (12, 24, 48)
EPOCHS = 3
LEARNING_RATE = 0.001
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def main() -> None:
    """Сравнивает точность до прунинга, сразу после и после дообучения."""
    student = StudentCNN()
    student.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "student_distilled.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    compact = prune_student(student, CHANNELS)
    torch.manual_seed(SEED)
    train_loader, val_loader = make_train_val_loaders(seed=SEED)
    criterion = nn.CrossEntropyLoss()
    _, original_accuracy = run_epoch(student, val_loader, criterion)
    _, pruned_accuracy = run_epoch(compact, val_loader, criterion)
    print(f"Точность до прунинга: {original_accuracy:.2%}", flush=True)
    print(f"Точность сразу после прунинга: {pruned_accuracy:.2%}", flush=True)

    train_model(
        model=compact,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        checkpoint_path=CHECKPOINT_DIR / "student_pruned.pt",
    )


if __name__ == "__main__":
    main()
