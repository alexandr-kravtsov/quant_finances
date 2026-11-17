"""
JERICHON Purged K-Fold Cross-Validator
======================================

Реализует кросс-валидацию с очисткой (purging) и эмбарго
из книги "Advances in Financial Machine Learning".

Обычный TimeSeriesSplit не учитывает перекрытие событий
между тренировочной и тестовой выборками.
PurgedKFold удаляет все тренировочные события, которые
пересекаются с тестовыми, а embargo добавляет отступ,
гарантируя отсутствие утечки информации из будущего.
"""

from __future__ import annotations

import logging
from typing import Generator, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("jerichon.purged_kfold")
logger.addHandler(logging.NullHandler())


def purged_kfold_split(
    events: pd.DataFrame,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
    """
    Генератор индексов событий для train/test разбиения.

    Параметры
    ---------
    events : pd.DataFrame
        События после Triple Barrier.
        Индекс — event_time, обязательные колонки:
            - t1 : время вертикального барьера
            - (можно добавить любые другие, они сохранятся)
    n_splits : int
        Количество фолдов (обычно 5).
    embargo_pct : float
        Доля от общего временного диапазона, добавляемая
        в качестве эмбарго между train и test (например 0.01).

    Возвращает
    -------
    Генератор кортежей (train_indices, test_indices)
    для каждого фолда.
    """
    # Сортируем события по времени
    events_sorted = events.sort_index()
    event_times = events_sorted.index
    t1_times = events_sorted["t1"].values

    n_events = len(events_sorted)
    if n_events < n_splits:
        raise ValueError(f"Событий ({n_events}) меньше, чем фолдов ({n_splits})")

    # Общий временной диапазон
    total_span = (event_times[-1] - event_times[0]).total_seconds()
    embargo_seconds = int(embargo_pct * total_span)

    # Разбиваем индексы на примерно равные части по времени
    # (каждый фолд охватывает непрерывный промежуток времени)
    split_edges = np.linspace(0, n_events - 1, n_splits + 1, dtype=int)

    for fold in range(n_splits):
        test_start_idx = split_edges[fold]
        test_end_idx = split_edges[fold + 1] - 1

        # Определяем временной диапазон тестового фолда
        test_start_time = event_times[test_start_idx]
        test_end_time = event_times[test_end_idx]

        # --- Purge: удаляем тренировочные события, которые пересекаются с тестовым периодом ---
        # Событие удаляется, если его t1 > test_start_time
        # (т.е. оно "заходит" на тестовую территорию)
        train_mask = t1_times < test_start_time

        # Применяем маску ко всем событиям, кроме тех, что попали в test
        train_indices = []
        test_indices = []
        for i, evt_time in enumerate(event_times):
            if test_start_idx <= i <= test_end_idx:
                test_indices.append(evt_time)
            elif train_mask[i]:
                # Дополнительно проверяем, что событие не пересекает границу
                # (t1 может быть точно равен test_start_time, тогда оставляем)
                if t1_times[i] <= test_start_time:
                    train_indices.append(evt_time)

        # --- Embargo: отодвигаем тестовый период на embargo_seconds вправо ---
        # Все события, начинающиеся в течение embargo после test_end_time,
        # не должны попасть в train этого фолда (они слишком близки к test)
        embargo_cutoff = test_end_time + pd.Timedelta(seconds=embargo_seconds)
        # Удаляем из train те события, которые начинаются до embargo_cutoff
        # (но не входят в test)
        if embargo_cutoff > test_end_time:
            # Пересобираем train_indices с учётом эмбарго
            train_indices = [
                t for t in train_indices
                if t > embargo_cutoff or t <= test_start_time
            ]

        yield pd.Index(train_indices), pd.Index(test_indices)