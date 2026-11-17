"""Metadata-driven wrapper for saved CatBoost models."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .backtest_engine import backtest_saved_catboost, BacktestConfig, BacktestResult


def _resolve_metadata(model_path: Path, metadata_path: Path | None) -> Path:
    if metadata_path is not None and Path(metadata_path).is_file():
        return Path(metadata_path)
    candidate = Path(model_path).with_suffix(".json")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Metadata JSON not found beside model: {candidate}")


def _build_event_labels(dollar_bars, cusum_threshold, tp_mult, sl_mult, vertical_horizon, target_window):
    from src.data.labeling.cusum import cusum_filter
    from src.data.labeling.triple_barrier import triple_barrier_labeling
    bars = dollar_bars.copy()
    bars = bars.rename(columns={c:c.upper() for c in ("open","high","low","close") if c in bars.columns})
    for col in ("HIGH","LOW","CLOSE"):
        if col not in bars:
            raise RuntimeError(f"OHLC column {col} missing from dollar_bars")
    if "bar_ts" not in bars:
        bars["bar_ts"] = bars["close_time"] if "close_time" in bars else bars.index
    bars["bar_ts"] = pd.to_datetime(bars["bar_ts"], errors="coerce", utc=True)
    bars = bars.dropna(subset=["bar_ts","HIGH","LOW","CLOSE"]).sort_values("bar_ts").drop_duplicates("bar_ts",keep="last").set_index("bar_ts")
    bars.index.name = "bar_ts"
    events = pd.DatetimeIndex(cusum_filter(bars["CLOSE"].astype(float), threshold=float(cusum_threshold)))
    events = events[events.isin(bars.index)].drop_duplicates().sort_values()
    if not len(events):
        raise RuntimeError("CUSUM returned no events; check threshold and bars range")
    labels = triple_barrier_labeling(bars=bars, events=events, pt_sl=(float(tp_mult),float(sl_mult)),
                                     vertical_barrier=int(vertical_horizon), target_window=int(target_window),
                                     target_method="ewm")
    if labels is None or labels.empty:
        raise RuntimeError("triple_barrier_labeling returned empty frame")
    labels = labels[labels["label"].isin([-1,1])].copy()
    for col in ("upper","lower","t1"):
        if col not in labels:
            raise RuntimeError(f"Label frame missing {col}; columns={list(labels.columns)}")
    labels["t1"] = pd.to_datetime(labels["t1"], errors="coerce", utc=True)
    labels.index = pd.to_datetime(labels.index, errors="coerce", utc=True)
    return labels


def run_saved_model_backtest(
    model_path, X, dollar_bars, *, metadata_path=None, initial_capital=100_000.0,
    commission=None, slippage=None, entry_cutoff_hour=None, forced_close_hour=None,
    session_timezone="Europe/Moscow", position_fraction=0.10, sizing_mode="risk_fraction",
    risk_fraction=0.0025, max_gross_leverage=3.0, close_opposite_on_signal=True,
) -> BacktestResult:
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    meta_file = _resolve_metadata(model_file, Path(metadata_path) if metadata_path else None)
    with meta_file.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    label_cfg = metadata.get("label_config", {})
    required = ("CUSUM_THRESHOLD","TP_MULTIPLIER","SL_MULTIPLIER","VERTICAL_HORIZON","TARGET_WINDOW")
    for key in required:
        if key not in label_cfg:
            raise KeyError(f"Metadata JSON missing label_config.{key}")
    labels = _build_event_labels(dollar_bars, label_cfg["CUSUM_THRESHOLD"], label_cfg["TP_MULTIPLIER"],
                                 label_cfg["SL_MULTIPLIER"], label_cfg["VERTICAL_HORIZON"], label_cfg["TARGET_WINDOW"])
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X must have DatetimeIndex aligned with bar timestamps")
    common = X.index.intersection(labels.index).sort_values()
    if common.empty:
        raise ValueError("No common timestamps between X and event labels")
    Xf = X.loc[common].copy()
    labels = labels.loc[common].copy()
    bt = metadata.get("backtest", {})
    cfg = BacktestConfig(
        barrier_mode="event_labels",
        vertical_horizon=int(label_cfg["VERTICAL_HORIZON"]),
        commission=float(commission if commission is not None else bt.get("commission",0.0006)),
        slippage=float(slippage if slippage is not None else bt.get("slippage",0.0001)),
        entry_cutoff_hour=int(entry_cutoff_hour if entry_cutoff_hour is not None else bt.get("entry_cutoff_hour",21)),
        forced_close_hour=int(forced_close_hour if forced_close_hour is not None else bt.get("forced_close_hour",23)),
        session_timezone=session_timezone, position_fraction=float(position_fraction),
        sizing_mode=sizing_mode, risk_fraction=float(risk_fraction),
        max_gross_leverage=float(max_gross_leverage), close_opposite_on_signal=bool(close_opposite_on_signal),
    )
    return backtest_saved_catboost(bars=dollar_bars, X=Xf, model_path=model_file, metadata_path=meta_file,
                                   config=cfg, event_labels=labels, initial_capital=float(initial_capital))
