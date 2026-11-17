"""Adapter for the Optuna notebook: delegates trade simulation to the canonical engine.

No confidence threshold is applied: every valid probability generates a direction
(p >= 0.5 => LONG, otherwise SHORT). Execution and sizing are owned by backtest_engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest_engine import backtest, BacktestConfig


def simulate_oos_fold(
    fold,
    probs,
    labels,
    bars,
    *,
    initial_capital=100_000.0,
    commission=0.0006,
    slippage=0.0001,
    vertical_horizon=50,
    risk_fraction=0.0025,
    max_gross_leverage=3.0,
    session_timezone="Europe/Moscow",
):
    probs = probs.dropna().sort_index()
    empty = {
        "total_return": 0.0, "trades": 0, "win_rate": np.nan,
        "profit_factor": 0.0, "max_drawdown": 0.0,
        "trade_rows": [], "equity": pd.Series(dtype=float),
    }
    if probs.empty:
        return empty

    # No confidence filter: all valid predictions become directional signals.
    signal_values = np.where(probs.to_numpy() >= 0.5, 1, -1)
    signals = pd.Series(signal_values, index=probs.index, name="signal")

    b = bars.copy()
    if "bar_ts" not in b:
        b["bar_ts"] = b["close_time"] if "close_time" in b else b.index
    b["bar_ts"] = pd.to_datetime(b["bar_ts"], errors="coerce", utc=True)
    b = b.dropna(subset=["bar_ts"]).sort_values("bar_ts").drop_duplicates("bar_ts", keep="last")
    start, end = pd.Timestamp(fold["val_start"]), pd.Timestamp(fold["val_end"])
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    b = b[(b["bar_ts"] >= start) & (b["bar_ts"] <= end)].copy()
    signals = signals.loc[signals.index.isin(b["bar_ts"])].copy()

    labels_fold = labels.copy()
    labels_fold.index = pd.to_datetime(labels_fold.index, errors="coerce", utc=True)
    labels_fold = labels_fold.loc[~labels_fold.index.isna()].copy()
    if "t1" not in labels_fold:
        raise ValueError("labels must contain t1")
    labels_fold["t1"] = pd.to_datetime(labels_fold["t1"], errors="coerce", utc=True)
    labels_fold = labels_fold.reindex(signals.index)
    valid = labels_fold["t1"].notna() & (labels_fold["t1"] <= end)
    signals = signals.loc[valid]
    labels_fold = labels_fold.loc[valid]
    if b.empty or signals.empty:
        return empty

    cfg = BacktestConfig(
        barrier_mode="event_labels",
        vertical_horizon=int(vertical_horizon),
        commission=float(commission),
        slippage=float(slippage),
        sizing_mode="risk_fraction",
        risk_fraction=float(risk_fraction),
        max_gross_leverage=float(max_gross_leverage),
        session_timezone=session_timezone,
        close_opposite_on_signal=True,
    )
    result = backtest(
        bars=b,
        signals=signals,
        config=cfg,
        event_labels=labels_fold,
        initial_capital=float(initial_capital),
    )
    trades = result.trades.to_dict("records") if not result.trades.empty else []
    return {
        "total_return": float(result.metrics.get("total_return", 0.0)),
        "trades": int(len(result.trades)),
        "win_rate": result.metrics.get("win_rate", np.nan),
        "profit_factor": result.metrics.get("profit_factor", 0.0),
        "max_drawdown": abs(float(result.metrics.get("max_drawdown", 0.0))),
        "trade_rows": trades,
        "equity": result.equity,
        "diagnostics": result.diagnostics,
    }
