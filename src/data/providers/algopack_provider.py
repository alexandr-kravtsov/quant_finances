"""
JERICHON AlgoPack Provider (MOEX ISS)
=====================================

Провайдер рыночных данных MOEX AlgoPack через ISS REST API.

Источники:
    1. Real-Time Market Data:
        - candles
        - trades
        - orderbook
        - statistics (заглушка)

    2. SuperCandles:
        - tradestats
        - obstats
        - orderstats

Принципы:
    - provider не строит признаки;
    - provider не объединяет потоки между собой;
    - provider не хранит историю;
    - provider возвращает сырые данные с оригинальными названиями колонок MOEX;
    - каждый метод добавляет унифицированную колонку `ts` (Europe/Moscow, tz-aware);
    - каждый DataFrame имеет колонку `ticker`, если она может быть определена;
    - модуль можно запускать напрямую для самопроверки.

Переменные окружения:
    ALGOPACK_BASE_URL
    ALGOPACK_TOKEN
    ALGOPACK_API_KEY
    ALGOPACK_TIMEOUT
    ALGOPACK_MAX_RETRIES
    ALGOPACK_LIMIT
    ALGOPACK_DEFAULT_TICKER
    ALGOPACK_VERIFY_SSL
    ALGOPACK_MOCK
"""

from __future__ import annotations
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import InsecureRequestWarning

# SSL verification intentionally disabled for JERICHON AlgoPack.
import urllib3
urllib3.disable_warnings(InsecureRequestWarning)
import argparse
import datetime as dt
import json
import logging
import math
import pandas as pd
import os
import sys
import time
from enum import Enum
from pathlib import Path
try:
    import config as _project_config  # type: ignore
except Exception:
    _project_config = None
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Environment / config loading
# ---------------------------------------------------------------------------

def _load_env_file() -> None:
    """
    Мини-загрузчик .env без внешней зависимости python-dotenv.
    Читает .env в текущей директории и рядом с файлом модуля.
    Существующие переменные окружения не перезаписываются.
    """
    candidates = [
        Path.cwd() / "API.env",
        Path(__file__).resolve().parent / "API.env",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

_load_env_file()

def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Возвращает значение настройки из environment или project config.py.
    Приоритет:
        1. os.environ
        2. config.py, если он существует и содержит такой атрибут
        3. default
    """
    value = os.environ.get(key)
    if value is not None and value != "":
        return value
    if _project_config is not None:
        try:
            config_value = getattr(_project_config, key, None)
            if config_value is not None:
                return str(config_value)
        except Exception:
            pass
    return default

def _env_bool(key: str, default: bool = False) -> bool:
    value = _env(key)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_int(key: str, default: int) -> int:
    value = _env(key)
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except Exception:
        return default

def _env_float(key: str, default: float) -> float:
    value = _env(key)
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except Exception:
        return default

# ---------------------------------------------------------------------------
# Настройки по умолчанию
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = _env("ALGOPACK_BASE_URL", "https://apim.moex.com")
DEFAULT_TOKEN = _env("ALGOPACK_TOKEN") or _env("ALGOPACK_API_KEY")
DEFAULT_TIMEOUT = _env_float("ALGOPACK_TIMEOUT", 30.0)
DEFAULT_MAX_RETRIES = _env_int("ALGOPACK_MAX_RETRIES", 3)
DEFAULT_LIMIT = _env_int("ALGOPACK_LIMIT", 1000)
DEFAULT_TICKER = _env("ALGOPACK_DEFAULT_TICKER", "ROSN")
DEFAULT_VERIFY_SSL = False
DEFAULT_MOCK = _env_bool("ALGOPACK_MOCK", False)

# Режимы торгов MOEX ISS
ENGINE = "stock"
MARKET = "shares"
BOARD = "TQBR"

ENDPOINTS = {
    "candles": f"/iss/engines/{ENGINE}/markets/{MARKET}/boards/{BOARD}/securities/{{ticker}}/candles.json",
    "trades": f"/iss/engines/{ENGINE}/markets/{MARKET}/boards/{BOARD}/securities/{{ticker}}/trades.json",
    "orderbook": f"/iss/engines/{ENGINE}/markets/{MARKET}/boards/{BOARD}/securities/{{ticker}}/orderbook.json",
    "tradestats": f"/iss/datashop/algopack/eq/tradestats/{{ticker}}.json",
    "obstats": f"/iss/datashop/algopack/eq/obstats/{{ticker}}.json",
    "orderstats": f"/iss/datashop/algopack/eq/orderstats/{{ticker}}.json",
}

LOGGER = logging.getLogger("jerichon.algopack")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AlgoPackError(RuntimeError):
    """Базовая ошибка AlgoPack provider."""
    def __init__(self, message: str = "", retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable

class AlgoPackAuthError(AlgoPackError):
    """Ошибка авторизации AlgoPack."""
    def __init__(self, message: str = "") -> None:
        super().__init__(message, retryable=False)

class AlgoPackRequestError(AlgoPackError):
    """Ошибка сетевого запроса или ответа сервера."""
    def __init__(self, message: str = "", retryable: bool = True) -> None:
        super().__init__(message, retryable=retryable)

class AlgoPackDataError(AlgoPackError):
    """Некорректная структура данных в ответе API."""
    def __init__(self, message: str = "") -> None:
        super().__init__(message, retryable=False)

class SuperCandleKind(str, Enum):
    TRADESTATS = "tradestats"
    OBSTATS = "obstats"
    ORDERSTATS = "orderstats"

VALID_SUPER_KINDS = {item.value for item in SuperCandleKind}

# ---------------------------------------------------------------------------
# Date / time helpers
# ---------------------------------------------------------------------------

def _today_msk() -> dt.date:
    """Текущая дата в московском часовом поясе."""
    msk = dt.timezone(dt.timedelta(hours=3))
    return dt.datetime.now(msk).date()

def _parse_date(value: Any) -> Optional[dt.date]:
    """Приводит входное значение к dt.date."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, (int, float)):
        try:
            return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc).date()
        except Exception as exc:
            raise ValueError(f"Не удалось распознать timestamp: {value!r}") from exc
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lower = text.lower()
        if lower in {"today", "now"}:
            return _today_msk()
        text = text.replace("/", "-")
        if "T" in text:
            text = text.split("T", 1)[0]
        if len(text) > 10:
            text = text[:10]
        if len(text) == 8 and text.isdigit():
            text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
        try:
            return dt.date.fromisoformat(text)
        except Exception as exc:
            raise ValueError(f"Не удалось распознать дату: {value!r}") from exc
    raise ValueError(f"Не удалось распознать дату: {value!r}")

def _datetime_param(value: Any) -> str:
    """Приводит datetime/date/Timestamp к строке для query-параметров."""
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day).isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)

def _midnight_msk(value: Any) -> pd.Timestamp:
    """Начало торгового дня в Europe/Moscow (без конвертации в UTC)."""
    date_obj = _parse_date(value)
    ts = pd.Timestamp(date_obj)
    try:
        return ts.tz_localize("Europe/Moscow")
    except Exception:
        return ts

# ---------------------------------------------------------------------------
# MOEX ISS JSON парсинг и обработка времени
# ---------------------------------------------------------------------------

def _parse_moex_block(payload: Dict[str, Any], block_name: str) -> pd.DataFrame:
    """
    Парсит стандартный блок MOEX ISS.
    Формат ответа биржи:
    {
        "block_name": {
            "columns": ["col1", "col2", ...],
            "data": [
                [val1, val2, ...],
                ...
            ]
        }
    }
    Возвращает DataFrame с оригинальными названиями колонок MOEX.
    """
    if not payload or block_name not in payload:
        return pd.DataFrame()
    block = payload[block_name]
    columns = block.get("columns", [])
    data = block.get("data", [])
    if not columns or not data:
        return pd.DataFrame()
    return pd.DataFrame(data, columns=columns)


def _parse_moex_datetime(series: pd.Series) -> pd.Series:
    """Parse MOEX timestamps without inventing a UTC shift.

    MOEX ISS commonly returns naive exchange-local timestamps. Naive values
    are therefore interpreted as Europe/Moscow. Already timezone-aware values
    are converted to Europe/Moscow.
    """
    raw = pd.to_datetime(series, format="mixed", errors="coerce", utc=False)
    try:
        if isinstance(raw.dtype, pd.DatetimeTZDtype):
            return raw.dt.tz_convert("Europe/Moscow")
    except Exception:
        pass

    # Normal naive Datetime Series.
    if pd.api.types.is_datetime64_dtype(raw):
        return raw.dt.tz_localize("Europe/Moscow", ambiguous="NaT", nonexistent="shift_forward")

    # Mixed aware/naive fallback: parse each value explicitly. This is slower
    # but only used for malformed/mixed feeds and guarantees correct semantics.
    def one(value: Any) -> pd.Timestamp:
        if pd.isna(value):
            return pd.NaT
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("Europe/Moscow")
        return ts.tz_convert("Europe/Moscow")

    return pd.Series([one(v) for v in series], index=series.index)


def _parse_datetime_candles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Парсинг времени для свечей.
    Использует колонку 'begin', приводит к Europe/Moscow.
    Колонки 'begin' и 'end' удаляются, остальные остаются без изменений.
    """
    if df.empty or "begin" not in df.columns:
        return df
    try:
        df["ts"] = _parse_moex_datetime(df["begin"])
    except Exception as exc:
        LOGGER.warning(f"Failed to parse candle datetime: {exc}")
        df["ts"] = pd.NaT
    # Удаляем служебные колонки MOEX, которые больше не нужны
    df.drop(columns=["begin", "end"], errors="ignore", inplace=True)
    return df


def _parse_datetime_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Парсинг времени для сделок и стакана.
    Ищет колонку TRADETIME или SYSTIME, приводит к Europe/Moscow.
    """
    if df.empty:
        return df

    time_col = None
    for col in ["TRADETIME", "tradetime", "SYSTIME"]:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        # Если колонка времени не найдена — возвращаем как есть,
        # DatasetBuilder сам разберётся при merge_asof
        return df

    try:
        df["ts"] = _parse_moex_datetime(df[time_col])
    except Exception as exc:
        LOGGER.warning(f"Failed to parse trade datetime: {exc}")
        df["ts"] = pd.NaT

    return df


def _parse_datetime_supercandles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Парсинг времени для SuperCandles.
    Склеивает tradedate (YYYY-MM-DD) и tradetime (HH:MM:SS).
    """
    if df.empty or "tradedate" not in df.columns or "tradetime" not in df.columns:
        return df

    try:
        dt_strings = df["tradedate"].astype(str) + " " + df["tradetime"].astype(str)
        df["ts"] = _parse_moex_datetime(dt_strings)
    except Exception as exc:
        LOGGER.warning(f"Failed to parse supercandle datetime: {exc}")
        df["ts"] = pd.NaT

    # Удаляем служебные колонки, которые уже склеены в ts
    df.drop(columns=["tradedate", "tradetime"], errors="ignore", inplace=True)
    return df


def _cast_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит строковые представления чисел к float/int.
    ВАЖНО: Никакого масштабирования (MinMax, StandardScaler).
    Только приведение типов, чтобы pandas корректно работал с числами.
    """
    if df.empty:
        return df

    exclude = {
        "ts", "ticker", "begin", "end", "tradedate", "tradetime",
        "TRADETIME", "SYSTIME", "secid", "SECID", "boardid", "BOARDID"
    }

    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            if df[col].dtype == object:
                converted = pd.to_numeric(df[col], errors="coerce")
                # Конвертируем, только если уверены (>= 80% не-null)
                non_null = converted.notna().sum()
                total = len(converted)
                if total > 0 and non_null / total >= 0.8:
                    df[col] = converted
        except Exception:
            pass

    return df


def _add_ticker(df: pd.DataFrame, ticker: Optional[str]) -> pd.DataFrame:
    """
    Добавляет колонку ticker, если её ещё нет в данных.
    Если колонка SECID присутствует — оставляем и её (исходные данные MOEX).
    """
    if df is None or df.empty:
        return df
    if ticker and "ticker" not in df.columns:
        df["ticker"] = ticker
    return df


def _validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate without deleting observations.

    IMPORTANT: ``ts`` is NOT a unique key for trades. Multiple executions can
    legitimately have the same TRADETIME. TRADENO is the trade identity.
    Therefore this function only sorts/reset-indexes and NEVER calls
    drop_duplicates().
    """
    if df.empty:
        return df
    if "ts" in df.columns:
        if df["ts"].duplicated().any():
            n_dups = int(df["ts"].duplicated(keep=False).sum())
            LOGGER.info(
                "В trades обнаружены повторяющиеся timestamps: %d строк. "
                "НЕ удаляем: timestamp не является идентификатором сделки.",
                n_dups,
            )
        if not df["ts"].is_monotonic_increasing:
            df = df.sort_values("ts", kind="mergesort")
    return df.reset_index(drop=True)

def _drop_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет служебные колонки MOEX, которые не несут полезной информации.
    Сохраняет все данные, специфичные для каждого типа запроса.
    """
    if df.empty:
        return df

    service_cols = ["secid", "SECID", "boardid", "BOARDID", "SYSTIME"]
    df.drop(columns=[c for c in service_cols if c in df.columns], errors="ignore", inplace=True)
    return df


# ---------------------------------------------------------------------------
# AlgoPackProvider
# ---------------------------------------------------------------------------

class AlgoPackProvider:
    """
    Провайдер данных AlgoPack (MOEX ISS) для JERICHON Data Engine.

    Особенности:
    - Оригинальные названия колонок MOEX (OPEN, HIGH, LOW, CLOSE, VOLUME,
      TRADENO, TRADEPRICE, QUANTITY, VALUE, BUYSELL и т.д.)
    - Время: Europe/Moscow (tz-aware), колонка `ts`
    - Никакого масштабирования цен и объёмов
    - Автоматическая пагинация
    - Автоматические повторные запросы при сбоях (urllib3.Retry)
    - Rate limiting между запросами

    Методы:
        fetch_candles(...)      — минутные свечи OHLCV
        fetch_trades(...)       — обезличенные сделки
        fetch_order_book(...)   — история стакана
        fetch_tradestats(...)   — SuperCandles: статистика сделок
        fetch_obstats(...)      — SuperCandles: статистика стакана
        fetch_orderstats(...)   — SuperCandles: статистика заявок
        fetch_day_bundle(...)   — все данные за день
        healthcheck()           — проверка доступности API
        close()                 — закрыть HTTP-сессию
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        mock: Optional[bool] = None,
        default_ticker: Optional[str] = None,
        default_limit: Optional[int] = None,
        verify_ssl: Optional[bool] = None,
    ) -> None:
        """
        Инициализация провайдера.

        Параметры:
            token: API-токен AlgoPack (если None — берётся из ALGOPACK_TOKEN)
            base_url: базовый URL API
            timeout: таймаут HTTP-запросов в секундах
            max_retries: максимальное число повторных попыток
            mock: режим заглушек (без реальных запросов к API)
            default_ticker: тикер по умолчанию
            default_limit: размер страницы по умолчанию
            verify_ssl: проверять SSL-сертификаты
        """
        self.token = token or DEFAULT_TOKEN
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = DEFAULT_TIMEOUT if timeout is None else float(timeout)
        self.max_retries = DEFAULT_MAX_RETRIES if max_retries is None else int(max_retries)
        self.mock = DEFAULT_MOCK if mock is None else bool(mock)
        self.default_ticker = default_ticker or DEFAULT_TICKER
        self.default_limit = DEFAULT_LIMIT if default_limit is None else int(default_limit)
        self.verify_ssl = DEFAULT_VERIFY_SSL if verify_ssl is None else bool(verify_ssl)

        # Настройка HTTP-сессии с автоматическими повторными запросами
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "JERICHON/1.0 (Quant Research Platform)"
        })

        if self.token:
            self._session.headers["Authorization"] = f"Bearer {self.token}"

        # Стратегия повторных запросов:
        # - всего попыток: max_retries
        # - экспоненциальная задержка с коэффициентом 0.5
        # - повторять при 429 (Rate Limit), 5xx (ошибки сервера)
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        LOGGER.info(
            "AlgoPackProvider initialized: base_url=%s, mock=%s, ticker=%s",
            self.base_url, self.mock, self.default_ticker
        )

    # ------------------------------------------------------------------
    # HTTP-запросы и пагинация
    # ------------------------------------------------------------------

    def _request(self, url: str, params: Optional[Dict] = None) -> Any:
        """
        Выполняет HTTP GET-запрос с обработкой ошибок.
        Возвращает распарсенный JSON или пустой dict при 404.

        Raises:
            AlgoPackAuthError: ошибка авторизации (401, 403)
            AlgoPackRequestError: сетевая ошибка или ошибка сервера
        """
        start = time.perf_counter()
        try:
            resp = self._session.get(
                url,
                params=params,
                timeout=self.timeout,
                verify=False
            )
            elapsed = time.perf_counter() - start
            LOGGER.debug("HTTP %s %s (%.2fs)", resp.status_code, url, elapsed)
        except requests.RequestException as e:
            raise AlgoPackRequestError(f"Сетевая ошибка: {e}") from e

        # Обработка HTTP-статусов
        if resp.status_code in (401, 403):
            raise AlgoPackAuthError(
                f"Ошибка авторизации (HTTP {resp.status_code}). Проверьте токен."
            )
        if resp.status_code == 404:
            LOGGER.warning("MOEX endpoint вернул 404: %s", url)
            return {}
        if resp.status_code >= 500:
            raise AlgoPackRequestError(
                f"Ошибка сервера MOEX (HTTP {resp.status_code})"
            )
        resp.raise_for_status()

        if not resp.content:
            return {}
        return resp.json()

    def _fetch_paginated(
        self,
        url: str,
        params: Dict,
        block_name: str,
        page_size: int = 1000
    ) -> pd.DataFrame:
        """
        Загружает все страницы данных через MOEX ISS API.

        Использует параметр `start` для постраничной навигации.
        Для SuperCandles дополнительно проверяет курсор `{block_name}.cursor`.
        Между запросами выдерживает паузу (rate limiting).
        """
        frames: List[pd.DataFrame] = []
        start = 0

        while True:
            p = params.copy()
            p["start"] = start

            payload = self._request(url, p)
            df = _parse_moex_block(payload, block_name)

            if df.empty:
                break

            frames.append(df)

            # Проверка курсора (актуально для SuperCandles)
            # Формат: {"data.cursor": {"data": [[INDEX, TOTAL, PAGESIZE]]}}
            cursor_key = f"{block_name}.cursor"
            cursor_block = payload.get(cursor_key)
            if cursor_block and "data" in cursor_block and cursor_block["data"]:
                cursor_data = cursor_block["data"][0]
                if len(cursor_data) >= 3:
                    idx = int(cursor_data[0])
                    total = int(cursor_data[1])
                    psize = int(cursor_data[2])
                    if idx + psize >= total:
                        break
                    start += psize
                    time.sleep(0.15)  # rate limiting
                    continue

            # Стандартная проверка: если страница неполная — это конец
            if len(df) < page_size:
                break

            start += len(df)
            time.sleep(0.15)  # rate limiting

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Canonical trade representation used by Terminal / LiveStrategy
    # ------------------------------------------------------------------
    def _normalize_trades(
        self, df: pd.DataFrame, ticker: Optional[str] = None, date: Any = None
    ) -> pd.DataFrame:
        """Normalize trades without changing the number of observations."""
        if df is None or df.empty:
            return pd.DataFrame(columns=["ts", "trade_id", "price", "quantity", "value", "side", "ticker"])
        df = _parse_datetime_trades(df.copy())
        cmap = {str(c).lower(): c for c in df.columns}

        def copy_candidate(dst: str, candidates: list[str]) -> None:
            if dst in df.columns:
                return
            for candidate in candidates:
                src = cmap.get(candidate.lower())
                if src is not None:
                    df[dst] = df[src]
                    return

        copy_candidate("trade_id", ["TRADENO", "tradeno", "trade_id"])
        copy_candidate("price", ["TRADEPRICE", "PRICE", "price"])
        copy_candidate("quantity", ["QUANTITY", "quantity", "size", "volume"])
        copy_candidate("value", ["VALUE", "value", "amount"])
        copy_candidate("side", ["BUYSELL", "buysell", "side"])
        df = _cast_numeric_types(df)
        if "value" not in df.columns and {"price", "quantity"}.issubset(df.columns):
            df["value"] = df["price"] * df["quantity"]
        if "side" in df.columns:
            mapping = {"B": "buy", "S": "sell", "BUY": "buy", "SELL": "sell"}
            df["side"] = df["side"].astype(str).str.upper().map(mapping).fillna("unknown")
        df = _add_ticker(df, ticker)
        return _validate_frame(_drop_service_columns(df))

    # ------------------------------------------------------------------
    # Публичные методы загрузки данных
    # ------------------------------------------------------------------

    def fetch_candles(
        self,
        ticker: Optional[str] = None,
        date: Any = None,
        *,
        from_date: Optional[str] = None,
        till_date: Optional[str] = None,
        interval: int = 1,
        limit: Optional[int] = None,
        **extra_params: Any,
    ) -> pd.DataFrame:
        """
        Минутные свечи OHLCV.

        Возвращает DataFrame с оригинальными колонками MOEX:
            OPEN, HIGH, LOW, CLOSE, VOLUME, ...
            + ts (Europe/Moscow)
            + ticker

        Параметры:
            ticker: тикер инструмента
            date: дата (если указана — from_date и till_date игнорируются)
            from_date: начальная дата (YYYY-MM-DD)
            till_date: конечная дата (YYYY-MM-DD)
            interval: интервал свечей в минутах (по умолчанию 1)
            limit: максимальное количество записей
        """
        if self.mock:
            return self._mock_candles(ticker, date, limit)

        ticker = ticker or self.default_ticker
        from_ = from_date or str(_parse_date(date) or _today_msk())
        till = till_date or from_

        url = f"{self.base_url}{ENDPOINTS['candles'].format(ticker=ticker)}"
        params: Dict[str, Any] = {
            "from": from_,
            "till": till,
            "interval": interval,
        }
        params.update(extra_params)

        df = self._fetch_paginated(url, params, "candles", page_size=500)
        if df.empty:
            return df

        df = _parse_datetime_candles(df)
        df = _cast_numeric_types(df)
        df = _add_ticker(df, ticker)
        df = _drop_service_columns(df)
        return _validate_frame(df)

    def fetch_trades(
        self,
        ticker: Optional[str] = None,
        date: Any = None,
        *,
        from_date: Optional[str] = None,
        till_date: Optional[str] = None,
        limit: Optional[int] = None,
        **extra_params: Any,
    ) -> pd.DataFrame:
        """
        Обезличенные сделки.

        Возвращает DataFrame с оригинальными колонками MOEX:
            TRADENO, TRADEPRICE, QUANTITY, VALUE, BUYSELL, ...
            + ts (Europe/Moscow)
            + ticker
        """
        if self.mock:
            return self._mock_trades(ticker, date, limit)

        ticker = ticker or self.default_ticker
        from_ = from_date or str(_parse_date(date) or _today_msk())
        till = till_date or from_

        url = f"{self.base_url}{ENDPOINTS['trades'].format(ticker=ticker)}"
        params: Dict[str, Any] = {"from": from_, "till": till}
        params.update(extra_params)

        df = self._fetch_paginated(url, params, "trades", page_size=5000)
        if df.empty:
            return df

        df = _parse_datetime_trades(df)
        df = _cast_numeric_types(df)
        df = _add_ticker(df, ticker)
        df = _drop_service_columns(df)
        return _validate_frame(df)

    def fetch_recent_trades(
        self,
        ticker: Optional[str] = None,
        date: Any = None,
        *,
        tail_size: int = 50000,
        **extra_params: Any,
    ) -> pd.DataFrame:
        """Return the latest executions without timestamp de-duplication.

        TRADENO remains the trade identity. Equal timestamps are valid and are
        returned as separate rows. This method is intended for the live polling
        loop so it does not download the whole day's tape on every poll.
        """
        if self.mock:
            return self._mock_frame("trades", ticker, date, tail_size)

        ticker = ticker or self.default_ticker
        from_ = str(_parse_date(date) or _today_msk())
        till = from_
        url = f"{self.base_url}{ENDPOINTS['trades'].format(ticker=ticker)}"

        meta = {"from": from_, "till": till, "start": 0, "limit": 1}
        meta.update(extra_params)
        payload = self._request(url, meta)
        cursor = payload.get("trades.cursor", {}) if isinstance(payload, dict) else {}
        total = None
        if isinstance(cursor, dict) and cursor.get("data"):
            row = cursor["data"][0]
            if len(row) >= 2:
                try:
                    total = int(row[1])
                except (TypeError, ValueError):
                    total = None

        if total is None:
            df = self._fetch_paginated(url, {"from": from_, "till": till}, "trades", page_size=5000)
        else:
            start = max(0, total - int(tail_size))
            params = {"from": from_, "till": till, "start": start, "limit": int(tail_size)}
            params.update(extra_params)
            payload_tail = self._request(url, params)
            df = _parse_moex_block(payload_tail, "trades")

        return self._normalize_trades(df, ticker, date)

    def fetch_order_book(
        self,
        ticker: Optional[str] = None,
        date: Any = None,
        *,
        from_date: Optional[str] = None,
        till_date: Optional[str] = None,
        limit: Optional[int] = None,
        **extra_params: Any,
    ) -> pd.DataFrame:
        """
        История стакана заявок.

        Возвращает DataFrame с колонками MOEX (состав зависит от версии API):
            BUYSELL, PRICE, QUANTITY, ...
            + ts (Europe/Moscow)
            + ticker
        """
        if self.mock:
            return self._mock_orderbook(ticker, date, limit)

        ticker = ticker or self.default_ticker
        from_ = from_date or str(_parse_date(date) or _today_msk())
        till = till_date or from_

        url = f"{self.base_url}{ENDPOINTS['orderbook'].format(ticker=ticker)}"
        params: Dict[str, Any] = {"from": from_, "till": till}
        params.update(extra_params)

        df = self._fetch_paginated(url, params, "orderbook", page_size=1000)
        if df.empty:
            return df

        # Стакан использует тот же формат времени, что и сделки
        df = _parse_datetime_trades(df)
        df = _cast_numeric_types(df)
        df = _add_ticker(df, ticker)
        df = _drop_service_columns(df)
        return _validate_frame(df)

    # ------------------------------------------------------------------
    # SuperCandles
    # ------------------------------------------------------------------

    def _fetch_supercandles(
        self,
        kind: str,
        ticker: str,
        from_date: str,
        till_date: str
    ) -> pd.DataFrame:
        """
        Внутренний метод для загрузки SuperCandles любого типа.
        kind: одно из 'tradestats', 'obstats', 'orderstats'.
        """
        url = f"{self.base_url}{ENDPOINTS[kind].format(ticker=ticker)}"
        params = {"from": from_date, "till": till_date}

        df = self._fetch_paginated(url, params, "data", page_size=self.default_limit)
        if df.empty:
            return df

        df = _parse_datetime_supercandles(df)
        df = _cast_numeric_types(df)
        df = _add_ticker(df, ticker)
        df["super_kind"] = kind
        return _validate_frame(df)

    def fetch_tradestats(
        self,
        ticker: Optional[str] = None,
        date: Any = None,
        *,
        from_date: Optional[str] = None,
        till_date: Optional[str] = None,
        limit: Optional[int] = None,
        **extra_params: Any,
    ) -> pd.DataFrame:
        """
        SuperCandles: статистика сделок (tradestats).

        Содержит агрегированные метрики: количество сделок, объём,
        оборот, объём покупок/продаж, агрессивные покупки/продажи и др.
        """
        if self.mock:
            return self._mock_supercandles("tradestats", ticker, date, limit)

        ticker = ticker or self.default_ticker
        from_ = from_date or str(_parse_date(date) or _today_msk())
        till = till_date or from_

        return self._fetch_supercandles("tradestats", ticker, from_, till)

    def fetch_obstats(
        self,
        ticker: Optional[str] = None,
        date: Any = None,
        *,
        from_date: Optional[str] = None,
        till_date: Optional[str] = None,
        limit: Optional[int] = None,
        **extra_params: Any,
    ) -> pd.DataFrame:
        """
        SuperCandles: статистика стакана (obstats).

        Содержит метрики ликвидности: объёмы на покупку/продажу,
        спред, дисбаланс стакана, глубина и др.
        """
        if self.mock:
            return self._mock_supercandles("obstats", ticker, date, limit)

        ticker = ticker or self.default_ticker
        from_ = from_date or str(_parse_date(date) or _today_msk())
        till = till_date or from_

        return self._fetch_supercandles("obstats", ticker, from_, till)

    def fetch_orderstats(
        self,
        ticker: Optional[str] = None,
        date: Any = None,
        *,
        from_date: Optional[str] = None,
        till_date: Optional[str] = None,
        limit: Optional[int] = None,
        **extra_params: Any,
    ) -> pd.DataFrame:
        """
        SuperCandles: статистика заявок (orderstats).

        Содержит метрики потока заявок: количество рыночных/лимитных
        заявок, отмен, объём заявок, дисбаланс и др.
        """
        if self.mock:
            return self._mock_supercandles("orderstats", ticker, date, limit)

        ticker = ticker or self.default_ticker
        from_ = from_date or str(_parse_date(date) or _today_msk())
        till = till_date or from_

        return self._fetch_supercandles("orderstats", ticker, from_, till)

    # ------------------------------------------------------------------
    # Дневной пакет данных
    # ------------------------------------------------------------------

    def fetch_day_bundle(
        self,
        ticker: Optional[str] = None,
        date: Any = None,
        include_supercandles: bool = True,
        strict: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Возвращает все доступные данные за торговый день одним словарём.

        Ключи словаря:
            candles      — OHLCV свечи
            trades       — обезличенные сделки
            orderbook    — стакан заявок
            tradestats   — статистика сделок (SuperCandles)
            obstats      — статистика стакана (SuperCandles)
            orderstats   — статистика заявок (SuperCandles)

        Параметры:
            ticker: тикер инструмента
            date: дата (строка YYYY-MM-DD, date, datetime или Timestamp)
            include_supercandles: загружать ли SuperCandles
            strict: если True — пробрасывать исключения;
                    если False — возвращать пустой DataFrame при ошибке
        """
        ticker = ticker or self.default_ticker
        date_obj = _parse_date(date) or _today_msk()
        date_str = str(date_obj)

        fetchers: Dict[str, Callable[[], pd.DataFrame]] = {
            "candles": lambda: self.fetch_candles(ticker, date=date_str),
            "trades": lambda: self.fetch_trades(ticker, date=date_str),
            "orderbook": lambda: self.fetch_order_book(ticker, date=date_str),
        }

        if include_supercandles:
            fetchers.update({
                "tradestats": lambda: self.fetch_tradestats(ticker, date=date_str),
                "obstats": lambda: self.fetch_obstats(ticker, date=date_str),
                "orderstats": lambda: self.fetch_orderstats(ticker, date=date_str),
            })

        result: Dict[str, pd.DataFrame] = {}
        for name, fetcher in fetchers.items():
            try:
                result[name] = fetcher()
                LOGGER.info("✓ %s: %d записей", name, len(result[name]))
            except Exception as exc:
                LOGGER.error("✗ %s (%s): %s", name, ticker, exc)
                if strict:
                    raise
                # fallback — пустой DataFrame
                result[name] = pd.DataFrame()

        return result

    # ------------------------------------------------------------------
    # Health / служебные методы
    # ------------------------------------------------------------------

    def healthcheck(self) -> Dict[str, Any]:
        """
        Проверка доступности API.
        Пытается загрузить одну свечу, чтобы убедиться, что:
        - сеть доступна
        - токен валиден
        - API отвечает
        """
        if self.mock:
            return {"status": "ok", "mock": True}

        try:
            df = self.fetch_candles(self.default_ticker, limit=1)
            return {
                "ok": True,
                "rows": len(df),
                "ticker": self.default_ticker,
            }
        except AlgoPackAuthError:
            return {"ok": False, "error": "Ошибка авторизации. Проверьте токен."}
        except AlgoPackRequestError as e:
            return {"ok": False, "error": f"Сетевая ошибка: {e}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def test_connection(self) -> Dict[str, Any]:
        """Синоним для healthcheck."""
        return self.healthcheck()

    def close(self) -> None:
        """Закрывает HTTP-сессию."""
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def __enter__(self) -> "AlgoPackProvider":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"AlgoPackProvider(base_url={self.base_url!r}, "
            f"mock={self.mock}, ticker={self.default_ticker!r})"
        )

    # ------------------------------------------------------------------
    # Mock-заглушки для тестирования без API
    # ------------------------------------------------------------------

    def _mock_candles(
        self,
        ticker: Optional[str] = None,
        date: Any = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Генерирует правдоподобные свечи для отладки."""
        ticker = ticker or self.default_ticker
        date_obj = _parse_date(date) if date else _today_msk()
        n = max(1, int(limit or 60))
        n = min(n, 500)

        base = _midnight_msk(date_obj) + pd.Timedelta(hours=10)  # 10:00 MSK
        ts = [base + pd.Timedelta(minutes=i) for i in range(n)]

        base_price = 100.0
        open_prices = [base_price + 2.0 * math.sin(i / 14.0) + 0.01 * i for i in range(n)]
        close_prices = [o + 0.4 * math.sin(i / 6.0) for o in open_prices]
        high_prices = [max(o, c) + 0.3 * abs(math.sin(i / 10.0)) for o, c in zip(open_prices, close_prices)]
        low_prices = [min(o, c) - 0.3 * abs(math.cos(i / 10.0)) for o, c in zip(open_prices, close_prices)]
        volumes = [5000 + 200 * (i % 20) + 1000 * int(math.sin(i / 5.0)) for i in range(n)]

        df = pd.DataFrame({
            "ts": ts,
            "ticker": ticker,
            "OPEN": open_prices,
            "HIGH": high_prices,
            "LOW": low_prices,
            "CLOSE": close_prices,
            "VOLUME": volumes,
        })
        df["is_mock"] = True
        return df

    def _mock_trades(
        self,
        ticker: Optional[str] = None,
        date: Any = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Генерирует правдоподобные сделки для отладки."""
        ticker = ticker or self.default_ticker
        date_obj = _parse_date(date) if date else _today_msk()
        n = max(1, int(limit or 30))
        n = min(n, 500)

        base = _midnight_msk(date_obj) + pd.Timedelta(hours=10)
        ts = [base + pd.Timedelta(seconds=i * 30) for i in range(n)]

        prices = [100.0 + 0.5 * math.sin(i / 8.0) + 0.02 * i for i in range(n)]
        quantities = [1 + (i % 10) * 10 for i in range(n)]
        buysell = ["B" if i % 3 != 0 else "S" for i in range(n)]

        df = pd.DataFrame({
            "ts": ts,
            "ticker": ticker,
            "TRADENO": list(range(1, n + 1)),
            "TRADEPRICE": prices,
            "QUANTITY": quantities,
            "VALUE": [p * q for p, q in zip(prices, quantities)],
            "BUYSELL": buysell,
        })
        df["is_mock"] = True
        return df

    def _mock_orderbook(
        self,
        ticker: Optional[str] = None,
        date: Any = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Генерирует правдоподобный стакан для отладки."""
        ticker = ticker or self.default_ticker
        date_obj = _parse_date(date) if date else _today_msk()
        n = max(1, int(limit or 10))
        n = min(n, 100)

        base = _midnight_msk(date_obj) + pd.Timedelta(hours=10)
        ts = [base + pd.Timedelta(seconds=i * 10) for i in range(n)]

        bid_prices = [100.0 - 0.1 - 0.01 * (i % 5) for i in range(n)]
        ask_prices = [100.0 + 0.1 + 0.01 * (i % 5) for i in range(n)]

        df = pd.DataFrame({
            "ts": ts,
            "ticker": ticker,
            "BID": bid_prices,
            "ASK": ask_prices,
            "BIDDEPTH": [100 + i % 20 for i in range(n)],
            "ASKDEPTH": [100 + (i + 7) % 20 for i in range(n)],
        })
        df["is_mock"] = True
        return df

    def _mock_supercandles(
        self,
        kind: str,
        ticker: Optional[str] = None,
        date: Any = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Генерирует правдоподобные SuperCandles для отладки."""
        ticker = ticker or self.default_ticker
        date_obj = _parse_date(date) if date else _today_msk()
        n = max(1, int(limit or 20))
        n = min(n, 200)

        base = _midnight_msk(date_obj) + pd.Timedelta(hours=10)
        ts = [base + pd.Timedelta(minutes=i * 5) for i in range(n)]

        if kind == "tradestats":
            df = pd.DataFrame({
                "ts": ts,
                "ticker": ticker,
                "super_kind": kind,
                "trades_count": [50 + (i % 20) for i in range(n)],
                "volume": [10000 + 200 * (i % 15) for i in range(n)],
                "buy_volume": [5000 + 100 * (i % 12) for i in range(n)],
                "sell_volume": [4800 + 90 * ((i + 3) % 12) for i in range(n)],
            })
        elif kind == "obstats":
            df = pd.DataFrame({
                "ts": ts,
                "ticker": ticker,
                "super_kind": kind,
                "spread": [0.1 + 0.01 * (i % 5) for i in range(n)],
                "bid_depth": [500 + 20 * (i % 10) for i in range(n)],
                "ask_depth": [480 + 18 * ((i + 2) % 10) for i in range(n)],
            })
        elif kind == "orderstats":
            df = pd.DataFrame({
                "ts": ts,
                "ticker": ticker,
                "super_kind": kind,
                "market_orders": [10 + (i % 7) for i in range(n)],
                "limit_orders": [40 + (i % 15) for i in range(n)],
                "cancel_orders": [5 + (i % 4) for i in range(n)],
            })
        else:
            df = pd.DataFrame({
                "ts": ts,
                "ticker": ticker,
                "super_kind": kind,
            })

        df["is_mock"] = True
        return df


# ---------------------------------------------------------------------------
# CLI / Self-test
# ---------------------------------------------------------------------------

def _save_output(
    name: str,
    df: Optional[pd.DataFrame],
    outdir: str,
    ticker: str,
    date: dt.date,
) -> None:
    """Сохраняет DataFrame в parquet или csv."""
    if df is None or df.empty:
        print(f"  [{name}] пустой DataFrame — не сохраняем")
        return

    base = Path(outdir) / str(ticker) / str(date)
    base.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow  # noqa: F401
        path = base / f"{name}.parquet"
        df.to_parquet(path, index=False)
    except Exception:
        path = base / f"{name}.csv"
        df.to_csv(path, index=False)

    print(f"  [{name}] сохранён: {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI для самопроверки и загрузки данных.

    Примеры:
        python algopack_provider.py --mock
        python algopack_provider.py --ticker ROSN --date 2026-08-01 --method candles
        python algopack_provider.py --ticker SBER --date 2026-08-01 --method all --outdir ./data
    """
    parser = argparse.ArgumentParser(
        description="JERICHON AlgoPack Provider — загрузка рыночных данных MOEX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python algopack_provider.py --mock
  python algopack_provider.py --ticker ROSN --date 2026-08-01 --method candles
  python algopack_provider.py --ticker SBER --date 2026-08-01 --method all --outdir ./data
        """,
    )
    parser.add_argument("--ticker", default=None, help="Тикер (например, ROSN, SBER)")
    parser.add_argument("--date", default=None, help="Дата в формате YYYY-MM-DD")
    parser.add_argument(
        "--method",
        default="all",
        choices=[
            "all", "health",
            "candles", "trades", "orderbook",
            "tradestats", "obstats", "orderstats",
        ],
        help="Метод для тестирования",
    )
    parser.add_argument("--limit", type=int, default=5, help="Лимит строк для вывода")
    parser.add_argument("--mock", action="store_true", help="Принудительный mock-режим")
    parser.add_argument(
        "--force-real",
        action="store_true",
        help="Не переключаться в mock при отсутствии токена",
    )
    parser.add_argument("--token", default=None, help="API-токен AlgoPack")
    parser.add_argument("--base-url", default=None, help="Базовый URL API")
    parser.add_argument("--outdir", default=None, help="Директория для сохранения данных")
    parser.add_argument("--strict", action="store_true", help="Выходить с ошибкой при сбоях")
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробное логирование")

    args = parser.parse_args(argv)

    # Настройка логирования
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Определяем дату
    try:
        date_obj = _parse_date(args.date) if args.date else _today_msk()
    except Exception as exc:
        print(f"Ошибка в дате: {exc}", file=sys.stderr)
        return 2

    # Определяем режим (mock или реальный API)
    token_present = bool(args.token or DEFAULT_TOKEN)
    use_mock = args.mock or (not args.force_real and not token_present)

    if use_mock and not args.mock:
        LOGGER.warning(
            "⚠️  ALGOPACK_TOKEN не найден. Переключаюсь в mock-режим.\n"
            "   Используйте --token или установите переменную окружения ALGOPACK_TOKEN.\n"
            "   Для принудительной попытки без токена: --force-real"
        )

    # Создаём провайдер
    provider = AlgoPackProvider(
        token=args.token,
        base_url=args.base_url,
        mock=use_mock,
        default_ticker=args.ticker,
    )

    ticker = args.ticker or provider.default_ticker
    limit = args.limit if args.limit and args.limit > 0 else None
    date_str = str(date_obj)

    # Заголовок
    print("=" * 60)
    print("  JERICHON AlgoPack Provider — Self Test")
    print("=" * 60)
    print(f"  Режим:      {'MOCK' if provider.mock else 'РЕАЛЬНЫЙ API'}")
    print(f"  Базовый URL: {provider.base_url}")
    print(f"  Тикер:       {ticker}")
    print(f"  Дата:        {date_str}")
    print("=" * 60)
    print()

    failures: List[str] = []

    def run_health() -> None:
        """Проверка соединения."""
        started = time.perf_counter()
        try:
            result = provider.healthcheck()
            elapsed = time.perf_counter() - started
            status = "✓" if result.get("ok") else "✗"
            print(f"  [{status}] healthcheck ({elapsed:.2f}s)")
            print(f"         {json.dumps(result, ensure_ascii=False)}")
            if not result.get("ok"):
                failures.append("health")
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(f"  [✗] healthcheck ({elapsed:.2f}s): {exc}")
            failures.append("health")
        print()

    def run_method(name: str, fetcher: Callable[[], pd.DataFrame]) -> None:
        """Запуск одного метода и вывод результата."""
        started = time.perf_counter()
        try:
            df = fetcher()
            elapsed = time.perf_counter() - started
            print(f"  [✓] {name}: {len(df)} строк, {len(df.columns)} колонок ({elapsed:.2f}s)")
            print(f"      Колонки: {', '.join(df.columns[:10])}")

            if not df.empty:
                with pd.option_context(
                    "display.max_columns", 10,
                    "display.width", 140,
                    "display.max_colwidth", 20,
                ):
                    print(df.head(limit))
            else:
                print("      (пустой DataFrame)")

            if args.outdir:
                _save_output(name, df, args.outdir, ticker, date_obj)

        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(f"  [✗] {name} ({elapsed:.2f}s): {exc}")
            if args.verbose:
                LOGGER.exception("Детали ошибки для %s:", name)
            failures.append(name)
        print()

    # Словарь методов для тестирования
    fetchers: Dict[str, Callable[[], pd.DataFrame]] = {
        "candles": lambda: provider.fetch_candles(ticker, date=date_str, limit=limit),
        "trades": lambda: provider.fetch_trades(ticker, date=date_str, limit=limit),
        "orderbook": lambda: provider.fetch_order_book(ticker, date=date_str, limit=limit),
        "tradestats": lambda: provider.fetch_tradestats(ticker, date=date_str, limit=limit),
        "obstats": lambda: provider.fetch_obstats(ticker, date=date_str, limit=limit),
        "orderstats": lambda: provider.fetch_orderstats(ticker, date=date_str, limit=limit),
    }

    # Выполнение
    if args.method == "health":
        run_health()
    elif args.method == "all":
        run_health()
        for name, fetcher in fetchers.items():
            run_method(name, fetcher)
    else:
        if args.method not in fetchers:
            print(f"Неизвестный метод: {args.method}", file=sys.stderr)
            return 2
        run_method(args.method, fetchers[args.method])

    provider.close()

    # Итоги
    print("=" * 60)
    if failures:
        print(f"  Завершено с ошибками: {', '.join(failures)}")
        print("=" * 60)
        return 1 if args.strict else 0

    print("  Все проверки пройдены успешно.")
    print("=" * 60)
    return 0