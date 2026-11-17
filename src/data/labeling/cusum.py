"""JERICHON CUSUM filter.

Trade timestamps are allowed to repeat. This module NEVER de-duplicates
indices: if several dollar bars close at the same timestamp, each bar remains
an independent observation/event.
"""
from __future__ import annotations

import logging
from typing import Union
import numpy as np
import pandas as pd

logger = logging.getLogger("jerichon.cusum")
logger.addHandler(logging.NullHandler())


def cusum_filter(close: pd.Series, threshold: float, return_type: str = "index") -> Union[pd.DatetimeIndex, pd.Series]:
    if len(close) < 2:
        logger.warning("Недостаточно данных для CUSUM")
        return pd.DatetimeIndex([]) if return_type == "index" else pd.Series(dtype=float)

    log_ret = np.log(close.astype(float)).diff().dropna()
    if len(log_ret) == 0:
        return pd.DatetimeIndex([]) if return_type == "index" else pd.Series(dtype=float)

    s_pos = 0.0
    s_neg = 0.0
    events = []
    for t, r in log_ret.items():
        if not np.isfinite(r):
            continue
        s_pos = max(0.0, s_pos + float(r))
        s_neg = min(0.0, s_neg + float(r))
        if s_pos > threshold or s_neg < -threshold:
            s_pos = 0.0
            s_neg = 0.0
            events.append(t)  # intentionally preserve duplicate timestamps

    if return_type == "index":
        return pd.DatetimeIndex(events)
    if return_type != "series":
        raise ValueError("return_type должен быть 'index' или 'series'")
    event_series = pd.Series(0, index=close.index, dtype=int)
    for t in events:
        # With duplicate index labels, .loc marks all equal labels. The index
        # form above is the authoritative representation for event identity.
        event_series.loc[t] = 1
    return event_series


def adaptive_cusum_filter(
    close: pd.Series,
    multiplier: float = 3.0,
    volatility_window: int = 20,
    return_type: str = "index",
) -> Union[pd.DatetimeIndex, pd.Series]:
    if len(close) < volatility_window + 1:
        logger.warning("Недостаточно данных для адаптивного CUSUM")
        return pd.DatetimeIndex([]) if return_type == "index" else pd.Series(dtype=float)

    log_ret = np.log(close.astype(float)).diff().dropna()
    rolling_vol = log_ret.rolling(volatility_window).std()
    s_pos = 0.0
    s_neg = 0.0
    events = []
    for t, r in log_ret.items():
        vol = rolling_vol.loc[t]
        if pd.isna(vol) or vol <= 0 or not np.isfinite(r):
            continue
        threshold = multiplier * float(vol)
        s_pos = max(0.0, s_pos + float(r))
        s_neg = min(0.0, s_neg + float(r))
        if s_pos > threshold or s_neg < -threshold:
            s_pos = 0.0
            s_neg = 0.0
            events.append(t)

    if return_type == "index":
        return pd.DatetimeIndex(events)
    if return_type != "series":
        raise ValueError("return_type должен быть 'index' или 'series'")
    event_series = pd.Series(0, index=close.index, dtype=int)
    for t in events:
        event_series.loc[t] = 1
    return event_series
