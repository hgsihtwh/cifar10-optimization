"""Калибрует PTQ-модель, сравнивает точность с FP32 и сохраняет тензоры."""

import torch
from torch import nn

from cifar10.data import make_calibration_loader, make_train_val_loaders
from cifar10.models import StudentCNN
from cifar10.quantization import hide_known_torch_warnings, quantize_ptq
from cifar10.train_student import SEED
from cifar10.train_student_pruned import CHANNELS, CHECKPOINT_DIR
from cifar10.train_teacher import run_epoch


def main() -> None:
    """Сравнивает обе модели на одной и той же валидационной выборке."""
    hide_known_torch_warnings()
    student = StudentCNN(CHANNELS)
    student.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "student_pruned.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    calibration_loader = make_calibration_loader(seed=SEED)
    _, val_loader = make_train_val_loaders(seed=SEED)
    quantized = quantize_ptq(student, calibration_loader)
    criterion = nn.CrossEntropyLoss()
    fp32_loss, fp32_accuracy = run_epoch(student, val_loader, criterion)
    int8_loss, int8_accuracy = run_epoch(quantized, val_loader, criterion)
    print(f"FP32: loss={fp32_loss:.4f}, accuracy={fp32_accuracy:.2%}")
    print(f"PTQ:  loss={int8_loss:.4f}, accuracy={int8_accuracy:.2%}")
    print(f"Изменение точности: {(int8_accuracy - fp32_accuracy) * 100:+.2f} п. п.")

    checkpoint_path = CHECKPOINT_DIR / "student_ptq.pt"
    torch.save(quantized.state_dict(), checkpoint_path)
    print(f"Сохранён чекпоинт PTQ: {checkpoint_path}")


if __name__ == "__main__":
    main()
