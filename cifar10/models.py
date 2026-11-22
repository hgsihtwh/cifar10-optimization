"""Сверточные модели для классификации изображений CIFAR-10."""

from torch import Tensor, nn


class _CifarCNN(nn.Module):
    """Общая сверточная архитектура учителя и студента."""

    def __init__(self, channels: tuple[int, int, int]) -> None:
        super().__init__()
        # В каждом блоке «свертка -> BatchNorm -> ReLU -> MaxPool» число
        # каналов задает свертка. MaxPool уменьшает карты признаков с 32×32
        # до 16×16, 8×8 и 4×4. Затем усредняем каждую карту до одного числа
        # и по этим числам оцениваем принадлежность изображения к 10 классам.
        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, 10)

    def forward(self, images: Tensor) -> Tensor:
        """Возвращает для каждого изображения 10 оценок классов до softmax."""
        features = self.features(images)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)


class TeacherCNN(_CifarCNN):
    """Сверточная модель-учитель с 32, 64 и 128 каналами."""

    def __init__(self) -> None:
        super().__init__((32, 64, 128))


class StudentCNN(_CifarCNN):
    """Компактная сверточная модель-студент с настраиваемой шириной блоков."""

    def __init__(self, channels: tuple[int, int, int] = (16, 32, 64)) -> None:
        super().__init__(channels)
