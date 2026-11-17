"""
JERICHON Feature Builder
=========================

Объединяет все этапы конструирования признаков для ML:
- Технические индикаторы (RSI, MACD, Bollinger, ATR)
- Временны́е признаки (час, день недели, месяц)
- Лаговые признаки (сдвиги цены и объёма)
- Скользящие статистики (средние, стандартные отклонения)
- Признаки взаимодействия (произведения, отношения)
- Микроструктурные признаки (обёртка над MicroFeatures)

Работает с долларовыми/информационными барами или минутными снапшотами.
Никакого масштабирования – только генерация признаков.

Пример использования:
    fb = FeatureBuilder()
    X = fb.build_all(dollar_bars_df)
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Union

import numpy as np
import pandas as pd

# Импортируем наш микроструктурный модуль
from src.data.builders.micro_features import MicroFeatures

logger = logging.getLogger("jerichon.build_features")
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def _safe_divide(a, b, fill=0.0):
    """Безопасное деление с заменой NaN/Inf на fill."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(a, b)
        result = np.where(np.isfinite(result), result, fill)
    return result

# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------
class FeatureBuilder:
    """
    Построитель признаков для ML.
    Параметры:
        - windows: список окон для скользящих статистик (по умолчанию [5, 10, 20])
        - lags: список лагов для сдвигов (по умолчанию [1, 2, 3, 5])
        - include_micro: включать ли микроструктурные признаки
        - micro_params: параметры для MicroFeatures
    """

    def __init__(
        self,
        windows: List[int] = [5, 10, 20],
        lags: List[int] = [1, 2, 3, 5],
        include_micro: bool = True,
        micro_params: Optional[Dict] = None,
    ):
        self.windows = windows
        self.lags = lags
        self.include_micro = include_micro
        self.micro_params = micro_params or {}
        self._micro_builder = MicroFeatures(**self.micro_params) if include_micro else None

    # ------------------------------------------------------------------
    # 1. Технические индикаторы
    # ------------------------------------------------------------------
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет: RSI, MACD, Bollinger Bands, ATR, OBV.
        Ожидает колонки: close, high, low, volume.
        """
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean()
        rs = _safe_divide(avg_gain, avg_loss, fill=1.0)  # при avg_loss=0 даём большое число
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands (окно 20, 2 стандартных отклонения)
        ma20 = close.rolling(20, min_periods=1).mean()
        std20 = close.rolling(20, min_periods=1).std()
        df["bb_upper"] = ma20 + 2 * std20
        df["bb_lower"] = ma20 - 2 * std20
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = (close - ma20) / _safe_divide(std20, 1.0, fill=1e-10)

        # ATR (Average True Range, окно 14)
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14, min_periods=1).mean()

        # OBV (On-Balance Volume)
        df["obv"] = (np.sign(close.diff()) * volume).fillna(0).cumsum()

        return df

    # ------------------------------------------------------------------
    # 2. Временные признаки
    # ------------------------------------------------------------------
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет циклические и категориальные признаки времени.
        Ожидает DateTimeIndex.
        """
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Индекс не DatetimeIndex, временные признаки не добавлены")
            return df

        ts = df.index
        # Циклические признаки (sin/cos для сохранения непрерывности)
        df["minute_sin"] = np.sin(2 * np.pi * ts.minute / 60)
        df["minute_cos"] = np.cos(2 * np.pi * ts.minute / 60)
        df["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
        df["dayofweek_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7)
        df["month_sin"] = np.sin(2 * np.pi * ts.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * ts.month / 12)

        # Дополнительно: бинарные признаки начала/конца сессии (если нужно)
        df["is_session_start"] = ((ts.hour == 10) & (ts.minute < 10)).astype(int)
        df["is_session_end"] = ((ts.hour == 18) & (ts.minute >= 40)).astype(int)

        return df

    # ------------------------------------------------------------------
    # 3. Лаговые признаки
    # ------------------------------------------------------------------
    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет сдвиги (лаги) для ключевых колонок.
        По умолчанию: close, volume, trades_volume, trades_vwap (если есть).
        """
        df = df.copy()
        target_cols = ["close", "volume"]
        # Добавим дополнительные колонки, если присутствуют
        extra = ["trades_volume", "trades_vwap", "bid", "ask", "spread", "midprice"]
        for c in extra:
            if c in df.columns:
                target_cols.append(c)

        for col in target_cols:
            if col not in df.columns:
                continue
            for lag in self.lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df

    # ------------------------------------------------------------------
    # 4. Скользящие статистики
    # ------------------------------------------------------------------
    def add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Для каждого окна добавляет: rolling mean, std, min, max, skew, kurt
        для колонок close, volume, trades_volume, trades_vwap.
        """
        df = df.copy()
        numeric_cols = ["close", "volume"]
        if "trades_volume" in df.columns:
            numeric_cols.append("trades_volume")
        if "trades_vwap" in df.columns:
            numeric_cols.append("trades_vwap")

        for col in numeric_cols:
            if col not in df.columns:
                continue
            for w in self.windows:
                roll = df[col].rolling(w, min_periods=1)
                df[f"{col}_roll_mean_{w}"] = roll.mean()
                df[f"{col}_roll_std_{w}"] = roll.std()
                df[f"{col}_roll_min_{w}"] = roll.min()
                df[f"{col}_roll_max_{w}"] = roll.max()
                df[f"{col}_roll_skew_{w}"] = roll.skew()
                df[f"{col}_roll_kurt_{w}"] = roll.kurt()
        return df

    # ------------------------------------------------------------------
    # 5. Признаки взаимодействия
    # ------------------------------------------------------------------
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Произведения и отношения между парами признаков.
        Например: close * volume, close / volume, vwap deviation * spread.
        """
        df = df.copy()
        # Пары колонок для взаимодействия
        pairs = []
        if "close" in df.columns and "volume" in df.columns:
            pairs.append(("close", "volume"))
        if "close" in df.columns and "spread" in df.columns:
            pairs.append(("close", "spread"))
        if "close" in df.columns and "atr" in df.columns:
            pairs.append(("close", "atr"))
        if "trades_vwap" in df.columns and "spread" in df.columns:
            pairs.append(("trades_vwap", "spread"))
        if "vwap_deviation" in df.columns and "spread" in df.columns:
            pairs.append(("vwap_deviation", "spread"))

        for col_a, col_b in pairs:
            if col_a in df.columns and col_b in df.columns:
                df[f"{col_a}_x_{col_b}"] = df[col_a] * df[col_b]
                df[f"{col_a}_div_{col_b}"] = _safe_divide(df[col_a], df[col_b])

        return df

    # ------------------------------------------------------------------
    # 6. Микроструктурные признаки (обёртка)
    # ------------------------------------------------------------------
    def add_micro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет MicroFeatures, если они ещё не добавлены."""
        if self._micro_builder is None:
            return df
        # Проверим, есть ли уже микро-признаки (например, quoted_spread)
        if "quoted_spread" in df.columns:
            logger.debug("Микроструктурные признаки уже присутствуют, пропускаем")
            return df
        logger.info("Добавление микроструктурных признаков...")
        return self._micro_builder.build_all_features(df)

    # ------------------------------------------------------------------
    # 7. Сборка всего набора признаков
    # ------------------------------------------------------------------
    def build_all(
        self,
        df: pd.DataFrame,
        include_time: bool = True,
        include_lags: bool = True,
        include_rolling: bool = True,
        include_interaction: bool = True,
        include_micro: Optional[bool] = None,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """
        Последовательное добавление всех признаков.
        Параметры позволяют гибко настраивать конвейер.
        """
        df = df.copy()

        # 1. Технические индикаторы
        df = self.add_technical_indicators(df)

        # 2. Временные признаки
        if include_time:
            df = self.add_time_features(df)

        # 3. Лаги
        if include_lags:
            df = self.add_lag_features(df)

        # 4. Скользящие статистики
        if include_rolling:
            df = self.add_rolling_stats(df)

        # 5. Взаимодействия
        if include_interaction:
            df = self.add_interaction_features(df)

        # 6. Микроструктурные признаки (если запрошены и не отключены)
        use_micro = include_micro if include_micro is not None else self.include_micro
        if use_micro:
            df = self.add_micro_features(df)

        # 7. Очистка пропусков (если нужно)
        if dropna:
            # Удаляем строки, где целевые колонки (close) NaN
            df = df.dropna(subset=["close"])  # хотя бы close должен быть

        # 8. Убираем бесконечные значения
        df = df.replace([np.inf, -np.inf], np.nan)
        if dropna:
            # Удалим строки, где любой признак NaN (осторожно, может сильно проредить)
            # Вместо этого заменим NaN медианой или нулём? Пока оставим пользователю решение.
            pass

        logger.info(f"Итоговый набор признаков: {df.shape[1]} колонок")
        return df