"""Прореживает студента по L1-норме за несколько раундов с дообучением."""

import torch
from torch import nn

from cifar10.data import make_train_val_loaders
from cifar10.models import StudentCNN
from cifar10.pruning import prune_student
from cifar10.train_student_pruned import CHANNELS, CHECKPOINT_DIR, LEARNING_RATE, SEED
from cifar10.train_teacher import run_epoch

PRUNING_STAGES = ((15, 29, 58), (14, 27, 53), CHANNELS)


def main() -> None:
    """Измеряет точность до прунинга и после каждого этапа с дообучением."""
    student = StudentCNN()
    student.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "student_distilled.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    torch.manual_seed(SEED)
    train_loader, val_loader = make_train_val_loaders(seed=SEED)
    criterion = nn.CrossEntropyLoss()
    _, original_accuracy = run_epoch(student, val_loader, criterion)
    print(f"Точность до прунинга: {original_accuracy:.2%}", flush=True)

    for stage, channels in enumerate(PRUNING_STAGES, start=1):
        # Считаем нормы заново по весам предыдущего раунда.
        student = prune_student(student, channels)
        _, pruned_accuracy = run_epoch(student, val_loader, criterion)
        # После изменения размеров тензоров создаём Adam для новой модели.
        optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
        train_loss, train_accuracy = run_epoch(
            student,
            train_loader,
            criterion,
            optimizer,
        )
        val_loss, val_accuracy = run_epoch(
            student,
            val_loader,
            criterion,
        )
        print(
            f"Раунд {stage}/{len(PRUNING_STAGES)} | каналы: {channels}\n"
            f"  Точность сразу после прунинга: {pruned_accuracy:.2%}\n"
            f"  Дообучение: loss={train_loss:.4f}, accuracy={train_accuracy:.2%}\n"
            f"  Валидация после дообучения: loss={val_loss:.4f}, accuracy={val_accuracy:.2%}",
            flush=True,
        )

    # Для сравнения сохраняем модель с итоговой шириной каналов.
    checkpoint_path = CHECKPOINT_DIR / "student_pruned_iterative.pt"
    torch.save(student.state_dict(), checkpoint_path)
    print(f"Сохранены итоговые веса: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
