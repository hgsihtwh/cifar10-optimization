"""Функция потерь для обучения студента по меткам и предсказаниям учителя."""

from torch import Tensor
from torch.nn import functional as F


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    temperature: float = 3.0,
    teacher_weight: float = 0.5,
) -> Tensor:
    """Смешивает ошибку по меткам и расхождение с распределением учителя."""
    label_loss = F.cross_entropy(student_logits, labels)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
    teacher_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    )
    return (1 - teacher_weight) * label_loss + teacher_weight * temperature**2 * teacher_loss
