"""Обучает студента по меткам CIFAR-10 и предсказаниям учителя."""

from pathlib import Path

import torch

from cifar10.data import make_train_val_loaders
from cifar10.models import StudentCNN, TeacherCNN
from cifar10.train_student import EPOCHS, LEARNING_RATE, SEED
from cifar10.train_teacher import train_model

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def main() -> None:
    """Загружает учителя и сохраняет лучшие веса студента по валидации."""
    teacher = TeacherCNN()
    teacher.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "teacher_tuned.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    teacher.eval()
    teacher.requires_grad_(False)

    # Сбрасываем генератор после создания учителя, чтобы повторить старт студента.
    torch.manual_seed(SEED)
    train_loader, val_loader = make_train_val_loaders(seed=SEED)
    train_model(
        model=StudentCNN(),
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        checkpoint_path=CHECKPOINT_DIR / "student_distilled.pt",
        teacher=teacher,
    )


if __name__ == "__main__":
    main()
