"""
JERICHON Microstructure Features
=================================

Микроструктурные признаки рынка на основе
минутных снимков (Market Snapshots) и ленты сделок.

Используются подходы из "Advances in Financial Machine Learning"
и классические метрики ликвидности.

Все методы принимают DataFrame с колонками:
- Свечи: OPEN, HIGH, LOW, CLOSE, VOLUME (или доступные)
- Стакан: bid, ask, spread, midprice (если есть)
- Агрегаты сделок: trades_volume, trades_vwap, trades_count (если есть)
- SuperCandles: любые колонки с суффиксами _tradestats, _obstats, _orderstats

Возвращают DataFrame с добавленными колонками признаков.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger("jerichon.micro_features")
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Вспомогательные утилиты для работы с колонками
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    """Находит первую колонку, содержащую любой из паттернов (регистронезависимо)."""
    for col in df.columns:
        col_lower = col.lower()
        for pat in patterns:
            if pat.lower() in col_lower:
                return col
    return None


def _has_any_column(df: pd.DataFrame, patterns: List[str]) -> bool:
    return _find_column(df, patterns) is not None


def _safe_col(df: pd.DataFrame, default_name: str) -> Optional[str]:
    """Возвращает колонку, если существует, иначе None."""
    return default_name if default_name in df.columns else None


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class MicroFeatures:
    """
    Вычисление микроструктурных признаков.
    Можно настроить окна для скользящих статистик.
    """

    def __init__(
        self,
        spread_window: int = 20,
        volatility_window: int = 20,
        intensity_window: int = 20,
        amihud_window: int = 20,
        roll_window: int = 20,
        kyle_window: int = 20,
        vpin_volume_window: int = 50,  # число баров для оценки VPIN
    ):
        self.spread_window = spread_window
        self.volatility_window = volatility_window
        self.intensity_window = intensity_window
        self.amihud_window = amihud_window
        self.roll_window = roll_window
        self.kyle_window = kyle_window
        self.vpin_volume_window = vpin_volume_window

    # ------------------------------------------------------------------
    # 1. Спреды
    # ------------------------------------------------------------------
    def add_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет:
        - quoted_spread: ask - bid
        - relative_spread: (ask - bid) / midprice
        - effective_spread: 2 * |price - midprice| (требуется price сделки)
        """
        df = df.copy()
        bid_col = _safe_col(df, "bid")
        ask_col = _safe_col(df, "ask")
        mid_col = _safe_col(df, "midprice")

        if bid_col and ask_col:
            df["quoted_spread"] = df[ask_col] - df[bid_col]
            if mid_col:
                df["relative_spread"] = df["quoted_spread"] / df[mid_col]

        # effective spread использует цену сделки (например, trades_vwap или close)
        trade_price_col = _safe_col(df, "trades_vwap") or _safe_col(df, "CLOSE")
        if trade_price_col and mid_col:
            df["effective_spread"] = 2 * np.abs(df[trade_price_col] - df[mid_col])

        return df

    # ------------------------------------------------------------------
    # 2. Дисбаланс потока ордеров
    # ------------------------------------------------------------------
    def add_trade_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Order flow imbalance на основе:
        - агрегатов buy_volume / sell_volume из SuperCandles tradestats
        - или BUYSELL меток в сделках (если есть)
        """
        df = df.copy()
        # Ищем колонки с buy/sell
        buy_col = _find_column(df, ["buy_volume", "aggressor_buy_volume", "buy_vol"])
        sell_col = _find_column(df, ["sell_volume", "aggressor_sell_volume", "sell_vol"])

        if buy_col and sell_col:
            df["trade_imbalance"] = (df[buy_col] - df[sell_col]) / (df[buy_col] + df[sell_col] + 1e-10)
        else:
            # Пытаемся использовать метки BUYSELL в сделках (если были)
            logger.debug("Нет buy/sell volume, trade_imbalance не добавлен")
        return df

    # ------------------------------------------------------------------
    # 3. Волатильность
    # ------------------------------------------------------------------
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет историческую волатильность:
        - parkinson_vol (high-low)
        - garman_klass_vol (open-high-low-close)
        """
        df = df.copy()
        if all(c in df.columns for c in ["HIGH", "LOW"]):
            # Parkinson volatility estimator
            hl_ratio = np.log(df["HIGH"] / df["LOW"])
            parkinson_var = (1 / (4 * np.log(2))) * hl_ratio ** 2
            df["parkinson_vol"] = np.sqrt(
                parkinson_var.rolling(self.volatility_window).mean()
            )

        if all(c in df.columns for c in ["OPEN", "HIGH", "LOW", "CLOSE"]):
            # Garman-Klass
            log_ho = np.log(df["HIGH"] / df["OPEN"])
            log_lo = np.log(df["LOW"] / df["OPEN"])
            log_co = np.log(df["CLOSE"] / df["OPEN"])
            gk_var = 0.5 * (log_ho - log_lo)**2 - (2*np.log(2) - 1) * log_co**2
            df["garman_klass_vol"] = np.sqrt(
                gk_var.rolling(self.volatility_window).mean()
            )
        return df

    # ------------------------------------------------------------------
    # 4. Интенсивность сделок
    # ------------------------------------------------------------------
    def add_trade_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет:
        - trade_intensity: количество сделок в окне (trades_count)
        - avg_trade_size: средний размер сделки
        """
        df = df.copy()
        count_col = _safe_col(df, "trades_count")
        vol_col = _safe_col(df, "trades_volume")

        if count_col and vol_col:
            df["avg_trade_size"] = df[vol_col] / (df[count_col] + 1e-10)
            df["trade_intensity"] = df[count_col].rolling(self.intensity_window).sum()
        return df

    # ------------------------------------------------------------------
    # 5. Отклонение от VWAP
    # ------------------------------------------------------------------
    def add_vwap_deviation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет vwap_deviation = CLOSE - trades_vwap (или OPEN).
        """
        df = df.copy()
        vwap_col = _safe_col(df, "trades_vwap")
        price_col = _safe_col(df, "CLOSE") or _safe_col(df, "OPEN")
        if vwap_col and price_col:
            df["vwap_deviation"] = df[price_col] - df[vwap_col]
        return df

    # ------------------------------------------------------------------
    # 6. Kyle's Lambda (упрощённая оценка)
    # ------------------------------------------------------------------
    def add_kyle_lambda(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Оценка Kyle's lambda как |return| / volume (чем выше, тем ниже ликвидность).
        """
        df = df.copy()
        price_col = _safe_col(df, "CLOSE") or _safe_col(df, "trades_vwap")
        vol_col = _safe_col(df, "trades_volume") or _safe_col(df, "VOLUME")
        if price_col and vol_col:
            ret = df[price_col].pct_change().abs()
            df["kyle_lambda"] = ret / (df[vol_col] + 1e-10)
            df["kyle_lambda"] = df["kyle_lambda"].rolling(self.kyle_window).mean()
        return df

    # ------------------------------------------------------------------
    # 7. Amihud Illiquidity
    # ------------------------------------------------------------------
    def add_amihud_illiquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Amihud (2002) illiquidity = |return| / dollar_volume.
        """
        df = df.copy()
        price_col = _safe_col(df, "CLOSE") or _safe_col(df, "trades_vwap")
        vol_col = _safe_col(df, "trades_volume") or _safe_col(df, "VOLUME")
        if price_col and vol_col:
            ret = df[price_col].pct_change().abs()
            dollar_vol = df[price_col] * df[vol_col]
            df["amihud_illiquidity"] = ret / (dollar_vol + 1e-10)
            df["amihud_illiquidity"] = df["amihud_illiquidity"].rolling(self.amihud_window).mean()
        return df

    # ------------------------------------------------------------------
    # 8. Roll Spread (оценка спреда без стакана)
    # ------------------------------------------------------------------
    def add_roll_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Roll (1984) spread = 2 * sqrt(-cov(Δp_t, Δp_{t-1})), если cov < 0.
        """
        df = df.copy()
        price_col = _safe_col(df, "CLOSE") or _safe_col(df, "trades_vwap")
        if price_col:
            dp = df[price_col].diff()
            roll_cov = dp.rolling(self.roll_window).cov(dp.shift(1))
            # spread = 2 * sqrt(|min(cov, 0)|)
            df["roll_spread"] = 2 * np.sqrt(np.abs(roll_cov.clip(upper=0)))
        return df

    # ------------------------------------------------------------------
    # 9. Order Book Pressure (давление в стакане)
    # ------------------------------------------------------------------
    def add_orderbook_pressure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Давление на основе объемов bid/ask (если есть).
        Иначе пытается использовать bid_volume/ask_volume из SuperCandles obstats.
        """
        df = df.copy()
        bid_vol_col = _find_column(df, ["bid_volume", "bid_vol"])
        ask_vol_col = _find_column(df, ["ask_volume", "ask_vol"])
        if bid_vol_col and ask_vol_col:
            df["book_pressure"] = (df[bid_vol_col] - df[ask_vol_col]) / (df[bid_vol_col] + df[ask_vol_col] + 1e-10)
        return df

    # ------------------------------------------------------------------
    # 10. SuperCandles features (прямое переиспользование)
    # ------------------------------------------------------------------
    def add_supercandles_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Просто копирует колонки из SuperCandles с понятными именами,
        если они ещё не приведены.
        Пытается выделить ключевые: spread_bbo, cancel_vol, imbalance и т.д.
        """
        df = df.copy()
        # Пример: если есть колонка spread_bbo в tradestats, оставляем как есть
        # Можно добавить префиксы для единообразия, но оставим как есть.
        # Ничего не делаем, т.к. колонки уже присутствуют.
        return df

    # ------------------------------------------------------------------
    # Build all
    # ------------------------------------------------------------------
    def build_all_features(
        self,
        df: pd.DataFrame,
        include_supercandles: bool = True,
    ) -> pd.DataFrame:
        """
        Последовательно добавляет все доступные микроструктурные признаки.
        """
        df = df.copy()
        # Применяем только если колонки есть, иначе без ошибок.
        try:
            df = self.add_spread_features(df)
            df = self.add_trade_imbalance(df)
            df = self.add_volatility_features(df)
            df = self.add_trade_intensity(df)
            df = self.add_vwap_deviation(df)
            df = self.add_kyle_lambda(df)
            df = self.add_amihud_illiquidity(df)
            df = self.add_roll_spread(df)
            df = self.add_orderbook_pressure(df)
            if include_supercandles:
                df = self.add_supercandles_features(df)
        except Exception as e:
            logger.error(f"Ошибка при расчёте признаков: {e}", exc_info=True)
        return df