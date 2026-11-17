"""
JERICHON Sample Weights (Average Uniqueness)
=============================================

Реализация метода взвешивания выборки из книги
"Advances in Financial Machine Learning".

Для каждого события вычисляется средняя доля времени,
в течение которого оно не перекрывается с другими событиями.
Полученные веса передаются в модель через параметр sample_weight.

Используется ТОЛЬКО на тренировочной выборке,
чтобы компенсировать избыточность перекрывающихся наблюдений.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("jerichon.sample_weights")
logger.addHandler(logging.NullHandler())


def get_average_uniqueness(
    events: pd.DataFrame,
    bars: pd.DataFrame,
) -> pd.Series:
    """
    Рассчитывает вес каждого события как среднюю уникальность его баров.

    Параметры
    ---------
    events : pd.DataFrame
        Информация о событиях после Triple Barrier.
        Должен содержать колонки:
            - event_time (индекс) — время начала события
            - t1 — время вертикального барьера
    bars : pd.DataFrame
        Исходные бары, использованные для разметки (индекс — время).

    Возвращает
    -------
    pd.Series с весами, индекс совпадает с events.index.
    """
    if events.empty:
        return pd.Series(dtype=float)

    # 1. Строим матрицу принадлежности баров событиям
    #    (какие бары используются каждым событием)
    bar_index = bars.index
    n_bars = len(bar_index)

    # Для каждого события определяем номера баров от event_time до t1
    event_bar_ranges = {}
    for evt_idx, row in events.iterrows():
        start = bar_index.get_loc(row.name)  # event_time
        # t1 может быть не точно на баре, берём последний бар до t1
        end = bar_index.get_indexer([row["t1"]], method="ffill")[0]
        if end >= start:
            event_bar_ranges[evt_idx] = (start, end)

    if not event_bar_ranges:
        return pd.Series(0.0, index=events.index)

    # 2. Считаем, сколько событий покрывают каждый бар
    bar_concurrency = np.zeros(n_bars, dtype=int)
    for start, end in event_bar_ranges.values():
        bar_concurrency[start:end + 1] += 1

    # 3. Уникальность каждого бара = 1 / concurrency
    bar_uniqueness = np.divide(1.0, bar_concurrency, where=bar_concurrency > 0)
    bar_uniqueness[bar_concurrency == 0] = 0.0

    # 4. Вес события = средняя уникальность его баров
    weights = {}
    for evt_idx, (start, end) in event_bar_ranges.items():
        avg_u = bar_uniqueness[start:end + 1].mean()
        weights[evt_idx] = avg_u

    return pd.Series(weights, name="sample_weight")