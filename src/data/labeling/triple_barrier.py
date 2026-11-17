"""
JERICHON Triple Barrier Labeling
=================================

Реализация метода тройного барьера из "Advances in Financial Machine Learning".
Превращает события CUSUM в метки (-1, 0, +1) для обучения моделей.

Алгоритм:
Для каждого события (t0):
    1. Определяется цена входа (close[t0]).
    2. Рассчитываются верхний и нижний барьеры на основе волатильности.
    3. Устанавливается вертикальный барьер (t1 = t0 + horizon баров).
    4. Отслеживается, какой из трёх барьеров был достигнут первым:
       - верхний → метка +1 (длинная позиция сработала)
       - нижний → метка -1 (короткая позиция сработала)
       - вертикальный → метка 0 (время истекло)

Вход:
    bars : pd.DataFrame с колонками HIGH, LOW, CLOSE
    events : pd.DatetimeIndex — моменты событий (из CUSUM)
    pt_sl : tuple (profit_taking, stop_loss) — множители для барьеров
    vertical_barrier : int — максимальная длина события в барах
    target_type : str — 'ewm' или 'rolling' для расчёта волатильности

Выход:
    pd.DataFrame с колонками:
        - event_time : время события
        - t1 : время вертикального барьера
        - target : волатильность на момент события
        - label : -1, 0, +1
        - ret : доходность до выхода
        - exit_time : время выхода
        - exit_price : цена выхода
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("jerichon.triple_barrier")
logger.addHandler(logging.NullHandler())


def _compute_target(
    close: pd.Series,
    window: int = 50,
    method: str = "ewm",
) -> pd.Series:
    log_ret = np.log(close).diff()

    if method == "ewm":
        target = log_ret.ewm(span=window).std()
    elif method == "rolling":
        target = log_ret.rolling(window=window).std()
    else:
        raise ValueError(f"Неизвестный метод расчёта target: {method}")

    return target


def triple_barrier_labeling(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    pt_sl: Tuple[float, float] = (1.0, 1.0),
    vertical_barrier: int = 20,
    target_window: int = 50,
    target_method: str = "ewm",
) -> pd.DataFrame:
    """
    Triple Barrier labeling.

    pt_sl = (profit_taking, stop_loss)

    Для long-style price-path labeling:
        upper = entry * (1 + pt * target)
        lower = entry * (1 - sl * target)

    КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ относительно старой версии:
    мы ищем ПЕРВЫЙ достигнутый barrier по времени.

    Старая версия сначала делала:
        if upper_reached.any():
            label = +1
    и только если upper вообще НЕ был достигнут, проверяла lower.

    Это означало:
        lower hit first
        lower hit
        затем upper hit
        => всё равно label +1

    Здесь:
        upper first  -> +1
        lower first  -> -1
        neither      -> 0

    Если оба barrier достигнуты на ОДНОМ И ТОМ ЖЕ баре,
    intrabar порядок неизвестен из OHLC. Для воспроизводимости
    используется tie_break="lower" policy, то есть в таком случае
    выбирается нижний barrier. Это можно позже заменить на "upper"
    или "drop", но выбор должен быть явным.
    """

    required_cols = {"HIGH", "LOW", "CLOSE"}
    missing = required_cols - set(bars.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки в bars: {missing}")

    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index, errors="coerce")
    bars = bars.sort_index()

    target_series = _compute_target(
        bars["CLOSE"],
        window=target_window,
        method=target_method,
    )

    events = pd.DatetimeIndex(events)
    events = events[events.isin(bars.index)]
    events = events.drop_duplicates().sort_values()

    if len(events) == 0:
        logger.warning("Нет событий CUSUM в индексе баров")
        return pd.DataFrame()

    pt, sl = pt_sl
    if pt <= 0 or sl <= 0:
        raise ValueError(f"pt_sl должны быть > 0, получено {pt_sl}")

    bar_index = bars.index
    results = []
    n_bars = len(bars)

    tie_lower_count = 0

    for pos, event_time in zip(
        bar_index.get_indexer(events),
        events
    ):
        if pos == -1:
            continue

        entry_price = float(bars["CLOSE"].iloc[pos])
        target_val = target_series.loc[event_time]

        if pd.isna(target_val) or target_val <= 0:
            continue

        upper = entry_price * (1.0 + pt * target_val)
        lower = entry_price * (1.0 - sl * target_val)

        t1_pos = min(pos + vertical_barrier, n_bars - 1)
        t1_time = bar_index[t1_pos]

        # Как и в старой реализации, event bar сам не используется
        # для hit-check. Проверяем следующие бары до t1 включительно.
        slice_bars = bars.iloc[pos + 1:t1_pos + 1]

        if len(slice_bars) == 0:
            continue

        upper_mask = slice_bars["HIGH"] >= upper
        lower_mask = slice_bars["LOW"] <= lower

        upper_hits = np.flatnonzero(upper_mask.to_numpy())
        lower_hits = np.flatnonzero(lower_mask.to_numpy())

        first_upper = int(upper_hits[0]) if len(upper_hits) else None
        first_lower = int(lower_hits[0]) if len(lower_hits) else None

        exit_type = "vertical"
        exit_time = t1_time
        exit_price = float(bars["CLOSE"].iloc[t1_pos])

        if first_upper is None and first_lower is None:
            pass
        elif first_upper is None:
            exit_type = "lower"
            exit_time = slice_bars.index[first_lower]
            exit_price = float(lower)
        elif first_lower is None:
            exit_type = "upper"
            exit_time = slice_bars.index[first_upper]
            exit_price = float(upper)
        elif first_upper < first_lower:
            exit_type = "upper"
            exit_time = slice_bars.index[first_upper]
            exit_price = float(upper)
        elif first_lower < first_upper:
            exit_type = "lower"
            exit_time = slice_bars.index[first_lower]
            exit_price = float(lower)
        else:
            # Both barriers touched on the same OHLC bar.
            # Intrabar order is unknowable from OHLC.
            tie_lower_count += 1
            exit_type = "lower"
            exit_time = slice_bars.index[first_lower]
            exit_price = float(lower)

        ret = np.log(exit_price / entry_price)

        if exit_type == "upper":
            label = 1
        elif exit_type == "lower":
            label = -1
        else:
            label = 0

        results.append({
            "event_time": event_time,
            "t1": t1_time,
            "target": float(target_val),
            "upper": float(upper),
            "lower": float(lower),
            "label": int(label),
            "ret": float(ret),
            "exit_time": exit_time,
            "exit_price": float(exit_price),
            "exit_type": exit_type,
        })

    if not results:
        logger.warning("Не удалось обработать ни одного события")
        return pd.DataFrame()

    events_df = pd.DataFrame(results)
    events_df.set_index("event_time", inplace=True)
    events_df.sort_index(inplace=True)

    counts = events_df["label"].value_counts()
    total = len(events_df)

    logger.info(
        f"Triple Barrier завершён: всего событий {total}. "
        f"+1: {counts.get(1, 0)} ({100*counts.get(1,0)/total:.1f}%), "
        f"-1: {counts.get(-1, 0)} ({100*counts.get(-1,0)/total:.1f}%), "
        f"0: {counts.get(0, 0)} ({100*counts.get(0,0)/total:.1f}%). "
        f"same-bar ties(lower policy): {tie_lower_count}"
    )

    return events_df
