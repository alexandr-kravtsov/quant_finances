"""
JERICHON Dataset Builder
=========================

Собирает сырые данные от AlgoPackProvider в единый 
минутный снимок рынка (Market Snapshot) для ML.
Обязательно сохраняет все исходные данные в историю.

Порядок работы:
1. Запросить данные за конкретный день (download_day)
2. Сохранить сырые данные в history/realtime и history/supercandles
3. Создать полный минутный индекс торговой сессии
4. Объединить свечи, сделки, стакан и статистику через merge_asof
5. Проверить целостность (validate)
6. Сохранить снапшот в parquet
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

from src.data.providers.algopack_provider import AlgoPackProvider

logger = logging.getLogger("jerichon.dataset_builder")
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
SESSION_START = time(10, 0)
SESSION_END = time(18, 50)

DEFAULT_STORAGE = Path("./data/snapshots")
DEFAULT_HISTORY = Path("./data/for_ml/storage/history")  # корень для realtime/supercandles

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def _minute_index(date: str | datetime) -> pd.DatetimeIndex:
    """Полный минутный индекс торговой сессии (Europe/Moscow)."""
    if isinstance(date, str):
        date = pd.Timestamp(date)
    start = pd.Timestamp(
        year=date.year, month=date.month, day=date.day,
        hour=SESSION_START.hour, minute=SESSION_START.minute,
        tz="Europe/Moscow"
    )
    end = pd.Timestamp(
        year=date.year, month=date.month, day=date.day,
        hour=SESSION_END.hour, minute=SESSION_END.minute,
        tz="Europe/Moscow"
    )
    return pd.date_range(start, end, freq="1min")

def _aggregate_trades_to_minutes(df: pd.DataFrame, minute_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Агрегирует ленту сделок в минутные бары: объём, VWAP, количество сделок."""
    if df.empty:
        return pd.DataFrame(index=minute_index)

    # Приводим названия колонок к ожидаемым, если требуется
    col_map = {}
    for col in df.columns:
        if col.upper() == "PRICE" or col.upper() == "TRADEPRICE":
            col_map[col] = "price"
        elif col.upper() == "QUANTITY" or col.upper() == "VOLUME":
            col_map[col] = "quantity"
    if col_map:
        df = df.rename(columns=col_map)

    if "price" not in df.columns or "quantity" not in df.columns:
        logger.error("Сделки не содержат колонок price/quantity после маппинга")
        return pd.DataFrame(index=minute_index)

    df = df.copy()
    df["minute"] = df["ts"].dt.floor("1min")

    volume = df.groupby("minute")["quantity"].sum()
    df["turnover"] = df["price"] * df["quantity"]
    turnover = df.groupby("minute")["turnover"].sum()
    total_qty = df.groupby("minute")["quantity"].sum()
    vwap = turnover / total_qty
    trades_count = df.groupby("minute").size()

    result = pd.DataFrame({
        "trades_volume": volume,
        "trades_vwap": vwap,
        "trades_count": trades_count
    }, index=minute_index)
    result.fillna(0, inplace=True)
    return result

def _aggregate_orderbook_to_minutes(df: pd.DataFrame, minute_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Из стакана берёт лучшие bid/ask на конец минуты (заглушка)."""
    if df.empty:
        return pd.DataFrame(index=minute_index, columns=["bid", "ask", "spread", "midprice"])

    df = df.copy()
    bid_col = next((c for c in df.columns if "bid" in c.lower()), None)
    ask_col = next((c for c in df.columns if "ask" in c.lower()), None)
    if not bid_col or not ask_col:
        logger.warning("В стакане не найдены колонки bid/ask")
        return pd.DataFrame(index=minute_index)

    df["minute"] = df["ts"].dt.floor("1min")
    snapshot = df.groupby("minute").agg({bid_col: "last", ask_col: "last"})
    snapshot.columns = ["bid", "ask"]
    snapshot["spread"] = snapshot["ask"] - snapshot["bid"]
    snapshot["midprice"] = (snapshot["ask"] + snapshot["bid"]) / 2

    return snapshot.reindex(minute_index)

# ---------------------------------------------------------------------------
# DatasetBuilder
# ---------------------------------------------------------------------------
class DatasetBuilder:
    """Сборщик минутных снапшотов рынка с обязательным архивированием сырых данных."""

    def __init__(
        self,
        provider: Optional[AlgoPackProvider] = None,
        storage_path: Path | str = DEFAULT_STORAGE,
        history_path: Path | str = DEFAULT_HISTORY,
        session_start: time = SESSION_START,
        session_end: time = SESSION_END,
        default_ticker: str = "ROSN",
    ):
        self.provider = provider or AlgoPackProvider()
        self.storage_path = Path(storage_path)
        self.history_path = Path(history_path)
        self.session_start = session_start
        self.session_end = session_end
        self.default_ticker = default_ticker

    def download_day(self, ticker: str, date: str) -> Dict[str, pd.DataFrame]:
        """
        Загружает все доступные данные за день.
        Возвращает словарь с ключами:
        'candles', 'trades', 'orderbook', 'tradestats', 'obstats', 'orderstats'
        """
        logger.info(f"Загрузка данных {ticker} за {date}")
        try:
            bundle = self.provider.fetch_day_bundle(ticker, date)
        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}")
            raise
        return bundle

    # ------------------------------------------------------------------
    # Сохранение сырых данных
    # ------------------------------------------------------------------
    def _save_raw_realtime(self, ticker: str, date: str, data: Dict[str, pd.DataFrame]) -> None:
        """Сохраняет сырые свечи, сделки и стакан в history/realtime."""
        base = self.history_path / "realtime" / ticker / date
        base.mkdir(parents=True, exist_ok=True)

        for key in ["candles", "trades", "orderbook"]:
            df = data.get(key)
            if df is not None and not df.empty:
                path = base / f"{key}.parquet"
                df.to_parquet(path, index=False)
                logger.info(f"Сырые данные сохранены: {path}")
            else:
                logger.debug(f"Нет данных {key} для сохранения")

    def _save_raw_supercandles(self, ticker: str, date: str, data: Dict[str, pd.DataFrame]) -> None:
        """Сохраняет сырые SuperCandles в history/supercandles."""
        base = self.history_path / "supercandles" / ticker / date
        base.mkdir(parents=True, exist_ok=True)

        for key in ["tradestats", "obstats", "orderstats"]:
            df = data.get(key)
            if df is not None and not df.empty:
                path = base / f"{key}.parquet"
                df.to_parquet(path, index=False)
                logger.info(f"SuperCandles сохранены: {path}")
            else:
                logger.debug(f"Нет данных {key} для сохранения")

    # ------------------------------------------------------------------
    # Построение снапшота
    # ------------------------------------------------------------------
    def build_minute_snapshot(
        self,
        ticker: str,
        date: str,
        include_supercandles: bool = True,
    ) -> pd.DataFrame:
        """
        Основной метод: собирает единый минутный DataFrame.
        Перед сборкой обязательно сохраняет сырые данные.
        """
        # 1. Загрузка
        data = self.download_day(ticker, date)

        # 1.1 Обязательное сохранение сырых данных
        try:
            self._save_raw_realtime(ticker, date, data)
            self._save_raw_supercandles(ticker, date, data)
        except Exception as e:
            logger.error(f"Критическая ошибка сохранения сырых данных: {e}")
            raise  # прерываем выполнение, чтобы не потерять историю

        # 2. Минутный индекс
        minute_idx = _minute_index(date)
        logger.debug(f"Минутный индекс: {minute_idx[0]} – {minute_idx[-1]}")

        # 3. Обработка свечей (они уже минутные)
        candles = data.get("candles", pd.DataFrame())
        if candles.empty:
            logger.warning("Нет свечей – пропускаем")
            candles = pd.DataFrame(index=minute_idx)
        else:
            candles = candles.set_index("ts").reindex(minute_idx)

        # 4. Агрегация сделок
        trades = data.get("trades", pd.DataFrame())
        trades_agg = _aggregate_trades_to_minutes(trades, minute_idx)

        # 5. Агрегация стакана
        orderbook = data.get("orderbook", pd.DataFrame())
        ob_agg = _aggregate_orderbook_to_minutes(orderbook, minute_idx)

        # 6. Объединение свечей, сделок и стакана по индексу
        snapshot = pd.concat([candles, trades_agg, ob_agg], axis=1)

        # 7. Добавление SuperCandles (если нужно)
        if include_supercandles:
            for sc_type in ["tradestats", "obstats", "orderstats"]:
                sc_df = data.get(sc_type, pd.DataFrame())
                if sc_df.empty:
                    continue
                sc_df = sc_df.set_index("ts").sort_index()
                snapshot = pd.merge_asof(
                    snapshot.sort_index(),
                    sc_df,
                    left_index=True,
                    right_index=True,
                    direction="forward",
                    suffixes=("", f"_{sc_type}")
                )

        snapshot.index.name = "ts"
        logger.info(f"Снапшот собран: {snapshot.shape}")
        return snapshot

    # ------------------------------------------------------------------
    # Валидация
    # ------------------------------------------------------------------
    def validate_snapshot(self, df: pd.DataFrame, ticker: str, date: str) -> pd.DataFrame:
        if df.empty:
            logger.warning(f"Пустой снапшот для {ticker} {date}")
            return df

        expected = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            logger.warning(f"Отсутствуют колонки: {missing}")

        all_nan = df.isna().all(axis=1)
        if all_nan.any():
            empty_minutes = df.index[all_nan]
            logger.warning(f"Минут без данных: {len(empty_minutes)} (первая: {empty_minutes[0]})")

        diffs = df.index.to_series().diff()
        breaks = diffs[diffs > pd.Timedelta("2min")]
        if not breaks.empty:
            logger.warning(f"Разрывы во времени: {len(breaks)}")

        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        return df

    # ------------------------------------------------------------------
    # Сохранение / загрузка снапшота
    # ------------------------------------------------------------------
    def _snapshot_path(self, ticker: str, date: str) -> Path:
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        return self.storage_path / ticker / f"{date_str}.parquet"

    def save_snapshot(self, df: pd.DataFrame, ticker: str, date: str) -> None:
        if df.empty:
            logger.warning(f"Попытка сохранить пустой снапшот {ticker} {date}")
            return
        path = self._snapshot_path(ticker, date)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=True)
        logger.info(f"Снапшот сохранён: {path}")

    def load_snapshot(self, ticker: str, date: str) -> Optional[pd.DataFrame]:
        path = self._snapshot_path(ticker, date)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def build_and_save(
        self,
        ticker: str,
        date: str,
        include_supercandles: bool = True,
        validate: bool = True,
    ) -> pd.DataFrame:
        df = self.build_minute_snapshot(ticker, date, include_supercandles)
        if validate:
            df = self.validate_snapshot(df, ticker, date)
        self.save_snapshot(df, ticker, date)
        return df

    # ------------------------------------------------------------------
    # Инкрементальное обновление
    # ------------------------------------------------------------------
    def _missing_dates(self, ticker: str, start_date: str, end_date: str) -> list[str]:
        all_dates = pd.date_range(start_date, end_date, freq="B")
        missing = []
        for d in all_dates:
            date_str = d.strftime("%Y-%m-%d")
            if not self._snapshot_path(ticker, date_str).exists():
                missing.append(date_str)
        return missing

    def update_snapshots(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        include_supercandles: bool = True,
    ) -> int:
        missing = self._missing_dates(ticker, start_date, end_date)
        if not missing:
            logger.info(f"Все дни уже загружены для {ticker}")
            return 0
        logger.info(f"Дней для загрузки {ticker}: {len(missing)}")
        loaded = 0
        for i, date_str in enumerate(missing, 1):
            try:
                logger.info(f"[{i}/{len(missing)}] Загрузка {ticker} за {date_str}")
                self.build_and_save(ticker, date_str, include_supercandles)
                loaded += 1
            except Exception as e:
                logger.error(f"Не удалось обработать {ticker} {date_str}: {e}")
        return loaded

    # ------------------------------------------------------------------
    # Пакетная загрузка
    # ------------------------------------------------------------------
    def build_snapshot_batch(
        self,
        tickers: list[str],
        date: str,
        include_supercandles: bool = True,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        for t in tickers:
            try:
                df = self.build_minute_snapshot(t, date, include_supercandles, **kwargs)
                results[t] = df
            except Exception as e:
                logger.error(f"Ошибка для {t}: {e}")
                results[t] = pd.DataFrame()
        return results

    # ------------------------------------------------------------------
    # Остальные утилиты (fill_gaps, build_ml_dataset)
    # ------------------------------------------------------------------
    def fill_gaps(
        self,
        df: pd.DataFrame,
        method: str = "ffill",
        limit: int = 5,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df
        if columns is None:
            sc_cols = [c for c in df.columns if any(
                suffix in c for suffix in ("_tradestats", "_obstats", "_orderstats")
            )]
            if not sc_cols:
                return df
            columns = sc_cols

        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = df[col].fillna(method=method, limit=limit)
        return df

    def build_ml_dataset(
        self,
        tickers: list[str],
        dates: list[str],
        include_supercandles: bool = True,
        fill_supercandles: bool = True,
    ) -> pd.DataFrame:
        all_frames = []
        total = len(tickers) * len(dates)
        count = 0
        for ticker in tickers:
            for date in dates:
                count += 1
                try:
                    logger.info(f"[{count}/{total}] Сборка {ticker} {date}")
                    snap = self.build_minute_snapshot(ticker, date, include_supercandles)
                    if fill_supercandles:
                        snap = self.fill_gaps(snap)
                    snap["ticker"] = ticker
                    snap = snap.set_index("ticker", append=True).swaplevel()
                    all_frames.append(snap)
                except Exception as e:
                    logger.error(f"Пропущен {ticker} {date}: {e}")
        if not all_frames:
            return pd.DataFrame()
        dataset = pd.concat(all_frames)
        dataset.index.names = ["ticker", "ts"]
        return dataset.sort_index()

    # ------------------------------------------------------------------
    # Compatibility with the canonical feature pipeline
    # ------------------------------------------------------------------
    def archive_raw_day(self, ticker: str, date: str, data: Dict[str, pd.DataFrame]) -> None:
        """Archive raw AlgoPack frames using the established on-disk layout.

        This compatibility method preserves the historical storage contract:
        history/realtime/<TICKER>/<YYYY-MM-DD>/{candles,trades,orderbook}.parquet
        history/supercandles/<TICKER>/<YYYY-MM-DD>/{tradestats,obstats,orderstats}.parquet
        """
        self._save_raw_realtime(ticker, date, data)
        self._save_raw_supercandles(ticker, date, data)

    def build_feature_dataset(self, ticker: str, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """Delegate to the canonical causal feature pipeline without changing it."""
        from src.data.builders.jerichon_feature_pipeline import build_feature_dataset

        kwargs.setdefault("provider", self.provider)
        kwargs.setdefault("history_path", self.history_path)
        return build_feature_dataset(ticker, start_date, end_date, **kwargs)

