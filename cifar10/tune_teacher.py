"""Подбирает скорость обучения учителя по точности на валидации."""

from pathlib import Path

import torch
from torch import nn

from cifar10.data import make_train_val_loaders
from cifar10.models import TeacherCNN
from cifar10.train_teacher import run_epoch

LEARNING_RATES = (0.0003, 0.001, 0.003)
EPOCHS = 10
SEED = 42
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "teacher_tuned.pt"


def main() -> None:
    """Сравнивает настройки с одинаковым бюджетом и сохраняет лучшие веса."""
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_epoch = 0
    best_lr = LEARNING_RATES[0]
    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)

    for lr in LEARNING_RATES:
        # Повторяем инициализацию весов, порядок батчей и случайные аугментации.
        torch.manual_seed(SEED)
        train_loader, val_loader = make_train_val_loaders(seed=SEED)
        model = TeacherCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trial_best_accuracy = -1.0
        trial_best_epoch = 0
        print(f"Скорость обучения: {lr:g}", flush=True)

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_accuracy = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
            val_loss, val_accuracy = run_epoch(
                model,
                val_loader,
                criterion,
            )
            print(
                f"  Эпоха {epoch}/{EPOCHS} | "
                f"обучение: loss={train_loss:.4f}, accuracy={train_accuracy:.2%} | "
                f"валидация: loss={val_loss:.4f}, accuracy={val_accuracy:.2%}",
                flush=True,
            )

            if val_accuracy > trial_best_accuracy:
                trial_best_accuracy = val_accuracy
                trial_best_epoch = epoch

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch
                best_lr = lr
                torch.save(model.state_dict(), CHECKPOINT_PATH)

        print(
            f"  Лучший результат для lr={lr:g}: "
            f"{trial_best_accuracy:.2%}, эпоха {trial_best_epoch}",
            flush=True,
        )

    print(f"Итог: lr={best_lr:g}, эпоха {best_epoch}, accuracy={best_accuracy:.2%}")
    print(f"Лучшие веса: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
