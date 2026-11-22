"""Сравнивает точность, размер чекпоинтов и задержку моделей на CPU."""

from statistics import median
from time import perf_counter

import torch
from torch import Tensor, nn

from cifar10.data import make_calibration_loader, make_test_loader
from cifar10.models import StudentCNN, TeacherCNN
from cifar10.quantization import hide_known_torch_warnings, quantize_ptq
from cifar10.train_student_pruned import CHANNELS, CHECKPOINT_DIR, SEED
from cifar10.train_teacher import run_epoch

CPU_THREADS = 1
WARMUP_RUNS = 30
TIMED_RUNS = 200


@torch.inference_mode()
def median_latency_ms(model: nn.Module, image: Tensor) -> float:
    """Измеряет медианное время прямого прохода после прогрева."""
    for _ in range(WARMUP_RUNS):
        model(image)

    durations = []
    for _ in range(TIMED_RUNS):
        started = perf_counter()
        model(image)
        durations.append(perf_counter() - started)
    return median(durations) * 1_000


def main() -> None:
    """Выводит метрики всех сохранённых вариантов на одной тестовой выборке."""
    hide_known_torch_warnings()
    torch.set_num_threads(CPU_THREADS)
    test_loader = make_test_loader()
    image = test_loader.dataset[0][0].unsqueeze(0)
    criterion = nn.CrossEntropyLoss()

    # Создаём структуру INT8. Параметры калибровки и веса заменяются чекпоинтом.
    int8_model = quantize_ptq(
        StudentCNN(CHANNELS),
        make_calibration_loader(seed=SEED, sample_size=128),
    )
    models = (
        ("Учитель FP32", "teacher_tuned.pt", TeacherCNN()),
        ("Базовый студент FP32", "student_baseline.pt", StudentCNN()),
        ("Студент с дистилляцией FP32", "student_distilled.pt", StudentCNN()),
        ("Однократный прунинг FP32", "student_pruned.pt", StudentCNN(CHANNELS)),
        ("Итеративный прунинг FP32", "student_pruned_iterative.pt", StudentCNN(CHANNELS)),
        ("PTQ INT8", "student_ptq.pt", int8_model),
        ("QAT INT8", "student_qat.pt", int8_model),
    )

    print(
        f"CPU: {CPU_THREADS} поток | INT8: {torch.backends.quantized.engine} | "
        f"Прогрев: {WARMUP_RUNS} | Замеры: {TIMED_RUNS}",
        flush=True,
    )
    print(f"{'Модель':<30} {'Accuracy':>9} {'КиБ':>9} {'мс':>9}", flush=True)
    for name, filename, model in models:
        checkpoint = CHECKPOINT_DIR / filename
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        _, accuracy = run_epoch(model, test_loader, criterion)
        latency = median_latency_ms(model, image)
        size = checkpoint.stat().st_size / 1_024
        print(f"{name:<30} {accuracy:>9.2%} {size:>9.2f} {latency:>9.3f}", flush=True)


if __name__ == "__main__":
    main()
