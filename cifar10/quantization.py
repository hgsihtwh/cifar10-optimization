"""Статическое 8-битное квантование компактного студента на CPU."""

import warnings
from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.ao import quantization as aq
from torch.utils.data import DataLoader

from cifar10.models import StudentCNN


def hide_known_torch_warnings() -> None:
    """Скрывает в терминале два предупреждения об устаревших частях PyTorch."""
    warnings.filterwarnings(
        "ignore",
        message=r"torch\.quantize_per_tensor, torch\.quantize_per_channel.*are deprecated",
        category=UserWarning,
        module=r"torch\.ao\.nn\.quantized\.modules\.utils",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"TypedStorage is deprecated\.",
        category=UserWarning,
        module=r"torch\._utils",
    )


class QuantizableStudent(nn.Module):
    """Добавляет переходы между вещественными и квантованными тензорами."""

    def __init__(self, student: StudentCNN) -> None:
        super().__init__()
        self.quant = aq.QuantStub()
        self.student = student
        self.dequant = aq.DeQuantStub()

    def forward(self, images: Tensor) -> Tensor:
        """Принимает вещественные изображения и возвращает вещественные оценки."""
        return self.dequant(self.student(self.quant(images)))


def _prepare_student(student: StudentCNN, *, qat: bool) -> QuantizableStudent:
    """Объединяет слои студента и настраивает выбранный режим квантования."""
    model = QuantizableStudent(deepcopy(student).cpu()).train(qat)
    # В каждом блоке объединяем свёртку, BatchNorm и ReLU. MaxPool остаётся отдельным.
    features = model.student.features
    groups = [[str(start + offset) for offset in range(3)] for start in range(0, len(features), 4)]
    engine = torch.backends.quantized.engine
    if qat:
        aq.fuse_modules_qat(features, groups, inplace=True)
        model.qconfig = aq.get_default_qat_qconfig(engine)
        aq.prepare_qat(model, inplace=True)
    else:
        aq.fuse_modules(features, groups, inplace=True)
        model.qconfig = aq.get_default_qconfig(engine)
        aq.prepare(model, inplace=True)
    return model


@torch.no_grad()
def quantize_ptq(
    student: StudentCNN,
    calibration_loader: DataLoader,
) -> QuantizableStudent:
    """Калибрует диапазоны активаций и конвертирует модель без обновления весов."""
    model = _prepare_student(student, qat=False)

    for images, _ in calibration_loader:
        model(images.cpu())

    aq.convert(model, inplace=True)
    return model


def prepare_qat(student: StudentCNN) -> QuantizableStudent:
    """Готовит копию студента к дообучению с имитацией квантования."""
    return _prepare_student(student, qat=True)


def convert_qat(model: QuantizableStudent) -> QuantizableStudent:
    """Конвертирует копию дообученной модели для 8-битного инференса."""
    return aq.convert(deepcopy(model).eval(), inplace=True)
