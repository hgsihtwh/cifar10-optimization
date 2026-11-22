"""Обучает модель-учителя на CIFAR-10 и сохраняет лучшие веса по валидации."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from cifar10.data import make_train_val_loaders
from cifar10.distillation import distillation_loss
from cifar10.models import TeacherCNN


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: Optimizer | None = None,
    *,
    teacher: nn.Module | None = None,
) -> tuple[float, float]:
    """Вычисляет средние потери и точность, при обучении обновляет веса."""
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros(())
    correct = torch.zeros((), dtype=torch.long)
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            if training:
                optimizer.zero_grad()

            logits = model(images)
            if teacher is None:
                loss = criterion(logits, labels)
            else:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                loss = distillation_loss(logits, teacher_logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            loss_sum += loss.detach() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum()
            total += batch_size

    return loss_sum.item() / total, correct.item() / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    checkpoint_path: Path,
    *,
    teacher: nn.Module | None = None,
) -> None:
    """Обучает модель и сохраняет лучшие веса по точности на валидации."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    checkpoint_path.parent.mkdir(exist_ok=True)
    best_val_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            teacher=teacher,
        )
        # На валидации считаем ошибку по меткам для сравнения с базовой моделью.
        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion,
        )
        print(
            f"Эпоха {epoch}/{epochs} | "
            f"обучение: loss={train_loss:.4f}, accuracy={train_accuracy:.2%} | "
            f"валидация: loss={val_loss:.4f}, accuracy={val_accuracy:.2%}",
            flush=True,
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Сохранены лучшие веса: {checkpoint_path}", flush=True)


def main() -> None:
    """Запускает обучение учителя с заданным числом эпох."""
    parser = argparse.ArgumentParser(description="Обучение модели-учителя на CIFAR-10")
    parser.add_argument("--epochs", type=int, default=1, help="Число эпох обучения")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Число эпох должно быть не меньше 1")

    seed = 42
    torch.manual_seed(seed)
    train_loader, val_loader = make_train_val_loaders(seed=seed)
    train_model(
        model=TeacherCNN(),
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=0.001,
        checkpoint_path=Path(__file__).resolve().parent.parent / "checkpoints" / "teacher.pt",
    )


if __name__ == "__main__":
    main()
