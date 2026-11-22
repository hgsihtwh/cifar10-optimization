"""Структурированный прунинг свёрточных фильтров студента по L1-норме."""

import torch
from torch import nn

from cifar10.models import StudentCNN

@torch.no_grad()
def prune_student(
    model: StudentCNN,
    channels: tuple[int, int, int],
) -> StudentCNN:
    """Создаёт компактного студента и переносит в него сохранённые каналы."""
    convolutions = [layer for layer in model.features if isinstance(layer, nn.Conv2d)]
    for conv, keep_count in zip(convolutions, channels, strict=True):
        if not 1 <= keep_count <= conv.out_channels:
            raise ValueError("Число сохраняемых каналов должно быть от 1 до исходной ширины слоя.")

    batch_norms = [layer for layer in model.features if isinstance(layer, nn.BatchNorm2d)]
    compact = StudentCNN(channels).to(model.classifier.weight)
    compact_convs = [layer for layer in compact.features if isinstance(layer, nn.Conv2d)]
    compact_norms = [layer for layer in compact.features if isinstance(layer, nn.BatchNorm2d)]
    kept_inputs = torch.arange(convolutions[0].in_channels, device=model.classifier.weight.device)

    for conv, norm, new_conv, new_norm in zip(
        convolutions,
        batch_norms,
        compact_convs,
        compact_norms,
        strict=True,
    ):
        keep_count = new_conv.out_channels
        # Оцениваем каждый фильтр исходного слоя целиком и сохраняем порядок каналов.
        scores = conv.weight.abs().sum(dim=(1, 2, 3))
        kept_outputs = scores.argsort(descending=True, stable=True)[:keep_count].sort().values
        new_conv.weight.copy_(conv.weight[kept_outputs][:, kept_inputs])
        new_norm.weight.copy_(norm.weight[kept_outputs])
        new_norm.bias.copy_(norm.bias[kept_outputs])
        new_norm.running_mean.copy_(norm.running_mean[kept_outputs])
        new_norm.running_var.copy_(norm.running_var[kept_outputs])
        new_norm.num_batches_tracked.copy_(norm.num_batches_tracked)
        kept_inputs = kept_outputs

    # После глобального усреднения каждому каналу соответствует один вход классификатора.
    compact.classifier.weight.copy_(model.classifier.weight[:, kept_inputs])
    compact.classifier.bias.copy_(model.classifier.bias)
    return compact.train(model.training)
