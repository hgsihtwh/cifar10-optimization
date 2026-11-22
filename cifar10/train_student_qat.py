"""Дообучает компактного студента с имитацией 8-битного квантования."""

import torch
from torch import nn

from cifar10.data import make_train_val_loaders
from cifar10.models import StudentCNN
from cifar10.quantization import convert_qat, hide_known_torch_warnings, prepare_qat
from cifar10.train_student import SEED
from cifar10.train_student_pruned import CHANNELS, CHECKPOINT_DIR
from cifar10.train_teacher import run_epoch

EPOCHS = 3
LEARNING_RATE = 0.0003


def main() -> None:
    """Сравнивает конвертированные модели и сохраняет лучшую по валидации."""
    hide_known_torch_warnings()
    student = StudentCNN(CHANNELS)
    student.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "student_pruned.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    model = prepare_qat(student)
    torch.manual_seed(SEED)
    train_loader, val_loader = make_train_val_loaders(seed=SEED)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_accuracy = -1.0
    checkpoint_path = CHECKPOINT_DIR / "student_qat.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )
        # Оцениваем реальный INT8, а имитацию квантования оставляем для обучения.
        quantized = convert_qat(model)
        val_loss, val_accuracy = run_epoch(
            quantized,
            val_loader,
            criterion,
        )
        print(
            f"Эпоха {epoch}/{EPOCHS} | "
            f"обучение: loss={train_loss:.4f}, accuracy={train_accuracy:.2%} | "
            f"валидация INT8: loss={val_loss:.4f}, accuracy={val_accuracy:.2%}",
            flush=True,
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(quantized.state_dict(), checkpoint_path)
            print(f"Сохранён лучший чекпоинт INT8: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
