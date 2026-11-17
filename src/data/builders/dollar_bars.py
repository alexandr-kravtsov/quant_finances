"""
JERICHON Information-Driven Bars
=================================
Реализация баров:
- Dollar bars (порог по накопленной денежной стоимости)
- Volume bars (порог по объёму)
- Tick bars (порог по числу сделок)
- Time bars (классические минутные/часовые свечи – для сравнения)

Принимает DataFrame сделок с колонками:
    ts (datetime64[ns, Europe/Moscow])
    price (float)
    quantity (int/float)

Возвращает DataFrame баров:
    open, high, low, close, volume, value (dollar volume), count (число сделок)

Пороги передаются непосредственно в методы, что позволяет легко менять их
из интерфейса (Streamlit) и использовать в гиперпараметрическом поиске.
Предусмотрен режим адаптивного порога (задел на будущее).
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("jerichon.bars")
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def _validate_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет и подготавливает DataFrame сделок."""
    if df.empty:
        return df
    required = {"price", "quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    if df["price"].isna().any() or df["quantity"].isna().any():
        logger.warning("NaN в ценах или объёмах, строки будут удалены")
        df = df.dropna(subset=["price", "quantity"])
    return df

def _aggregate_bars(ticks: pd.DataFrame, threshold: float, mode: str) -> pd.DataFrame:
    """
    Универсальный агрегатор баров по порогу.
    mode: 'dollar' | 'volume' | 'tick'
    """
    if ticks.empty:
        return pd.DataFrame()

    bars = []
    current = {
        "open": None, "high": -np.inf, "low": np.inf,
        "close": None, "volume": 0.0, "value": 0.0, "count": 0,
        "start_ts": ticks["ts"].iloc[0],
    }
    accumulator = 0.0

    for _, row in ticks.iterrows():
        price = row["price"]
        qty = row["quantity"]
        if price <= 0 or qty <= 0:
            continue

        # Определяем шаг накопления
        if mode == "dollar":
            step = price * qty
        elif mode == "volume":
            step = qty
        elif mode == "tick":
            step = 1
        else:
            raise ValueError(f"Неизвестный режим: {mode}")

        # Если это первая сделка в баре
        if current["open"] is None:
            current["open"] = price
            current["high"] = price
            current["low"] = price

        current["high"] = max(current["high"], price)
        current["low"] = min(current["low"], price)
        current["close"] = price
        current["volume"] += qty
        current["value"] += price * qty
        current["count"] += 1
        accumulator += step

        if accumulator >= threshold:
            # Закрываем бар
            current["end_ts"] = row["ts"]
            bars.append(current)
            # Сброс
            current = {
                "open": None, "high": -np.inf, "low": np.inf,
                "close": None, "volume": 0.0, "value": 0.0, "count": 0,
                "start_ts": row["ts"],  # начало следующего бара – timestamp текущей сделки
            }
            accumulator = 0.0

    # Добавляем последний незакрытый бар (опционально)
    if current["open"] is not None and current["count"] > 0:
        current["end_ts"] = ticks["ts"].iloc[-1]
        bars.append(current)

    if not bars:
        return pd.DataFrame()

    result = pd.DataFrame(bars)
    result.rename(columns={"start_ts": "open_time", "end_ts": "close_time"}, inplace=True)
    return result

# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------
class InformationDrivenBars:
    """
    Построитель информационно-эффективных баров.
    Все пороги передаются в методы, а не в конструктор,
    что позволяет легко интегрировать класс с UI и конфигами.
    """

    def __init__(self):
        """Конструктор без параметров."""
        pass

    # ------------------------------------------------------------------
    # Бары с фиксированным порогом
    # ------------------------------------------------------------------
    def build_dollar(
        self,
        trades: pd.DataFrame,
        threshold: float = 10_000_000,   # 10 млн рублей
    ) -> pd.DataFrame:
        """Долларовые бары (порог – накопленная стоимость сделок)."""
        trades = _validate_trades(trades)
        return _aggregate_bars(trades, threshold, mode="dollar")

    def build_volume(
        self,
        trades: pd.DataFrame,
        threshold: float = 500_000,      # объём в лотах/контрактах
    ) -> pd.DataFrame:
        """Объёмные бары (порог – накопленный объём)."""
        trades = _validate_trades(trades)
        return _aggregate_bars(trades, threshold, mode="volume")

    def build_tick(
        self,
        trades: pd.DataFrame,
        threshold: int = 25_000,         # количество сделок
    ) -> pd.DataFrame:
        """Тиковые бары (порог – число сделок)."""
        trades = _validate_trades(trades)
        return _aggregate_bars(trades, int(threshold), mode="tick")

    # ------------------------------------------------------------------
    # Адаптивный порог (задел на будущее)
    # ------------------------------------------------------------------
    def build_dollar_adaptive(
        self,
        trades: pd.DataFrame,
        target_bars_per_day: int = 50,
        window_days: int = 20,
    ) -> pd.DataFrame:
        """
        Долларовые бары с адаптивным порогом.
        threshold = средний дневной долларовый объём / target_bars_per_day.
        Пока усреднение идёт по всей выборке, позже можно добавить rolling window.
        """
        trades = _validate_trades(trades)
        total_dollar_value = (trades["price"] * trades["quantity"]).sum()
        n_days = trades["ts"].dt.date.nunique()
        if n_days == 0:
            return pd.DataFrame()
        avg_daily_dollar = total_dollar_value / n_days
        threshold = avg_daily_dollar / target_bars_per_day
        logger.info(
            f"Адаптивный порог: {threshold:,.0f} руб "
            f"(ср. дневной оборот {avg_daily_dollar:,.0f}, цель {target_bars_per_day} баров/день)"
        )
        return _aggregate_bars(trades, threshold, mode="dollar")

    # ------------------------------------------------------------------
    # Time bars (классические свечи)
    # ------------------------------------------------------------------
    def build_time(self, df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
        """
        Обычные временны́е бары (свечи).
        df может быть минутным снапшотом или сделками с ценой.
        Если это сделки – сначала агрегирует в минуты, потом ресамплит.
        """
        if "OPEN" in df.columns:
            # уже готовые свечи, просто ресампл
            return df.resample(freq).agg({
                "OPEN": "first",
                "HIGH": "max",
                "LOW": "min",
                "CLOSE": "last",
                "VOLUME": "sum"
            })
        else:
            # из сделок
            trades = _validate_trades(df)
            # Делаем минутные бары, затем ресампл
            minute = trades.set_index("ts").resample("1min").agg({
                "price": ["first", "max", "min", "last"],
                "quantity": "sum",
                "value": lambda x: (x * trades.loc[x.index, "price"]).sum()
            })
            minute.columns = ["open", "high", "low", "close", "volume", "value"]
            return minute.resample(freq).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "value": "sum"
            })

