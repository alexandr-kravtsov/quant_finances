"""Canonical JERICHON causal backtest engine.

Contract: signal at bar T close -> order at bar T+1 open.
Supports overlapping same-direction positions, per-position barriers, and a
hard aggregate gross-exposure cap. Historical backtest only; not live execution.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd

from .execution import FixedFractionSizer, RiskFractionSizer, execution_price, gross_pnl, round_trip_fees
from .metrics import calculate_metrics


@dataclass(frozen=True)
class BacktestConfig:
    barrier_mode: str = "event_labels"   # target | fixed_pct | event_labels
    pt_mult: float = 2.4
    sl_mult: float = 1.5
    tp_pct: float = 0.01
    sl_pct: float = 0.01
    vertical_horizon: int = 60
    commission: float = 0.0006
    slippage: float = 0.0001
    position_fraction: float = 0.10      # fixed-fraction sizing: notional/equity per entry
    sizing_mode: str = "fixed_fraction"  # fixed_fraction | risk_fraction
    risk_fraction: float = 0.0025       # equity-at-stop / equity, before gaps/costs
    max_gross_leverage: float = 3.0     # aggregate abs notional / current equity
    periods_per_year: float = 252.0
    entry_cutoff_hour: int | None = None
    forced_close_hour: int | None = None
    session_timezone: str | None = None
    close_opposite_on_signal: bool = True


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict
    diagnostics: dict


@dataclass
class Position:
    side: int
    qty: float
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry: float
    target: float
    tp: float
    sl: float
    vertical_exit_time: pd.Timestamp | None = None
    bars_held: int = 0


def _bars(bars: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise ValueError("bars must be a non-empty pandas DataFrame")
    x = bars.copy()
    raw_ts = x["bar_ts"] if "bar_ts" in x.columns else x.index
    x["bar_ts"] = pd.to_datetime(raw_ts, errors="coerce", utc=True)
    x = x.dropna(subset=["bar_ts"]).sort_values("bar_ts").drop_duplicates("bar_ts", keep="last").reset_index(drop=True)
    for col in ("OPEN", "HIGH", "LOW", "CLOSE"):
        if col not in x and col.lower() in x:
            x[col] = x[col.lower()]
    missing = [c for c in ("OPEN", "HIGH", "LOW", "CLOSE") if c not in x]
    if missing:
        raise ValueError(f"bars missing OHLC columns: {missing}")
    for col in ("OPEN", "HIGH", "LOW", "CLOSE"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    if x[["OPEN", "HIGH", "LOW", "CLOSE"]].isna().any().any():
        raise ValueError("OHLC contains NaN/non-numeric values")
    if (x[["OPEN", "HIGH", "LOW", "CLOSE"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    return x


def _session_hour(ts: pd.Timestamp, cfg: BacktestConfig) -> int:
    return int(ts.tz_convert(cfg.session_timezone).hour) if cfg.session_timezone else int(ts.hour)


def _lookup(obj: Any, i: int, ts: pd.Timestamp) -> float:
    if obj is None:
        return np.nan
    if isinstance(obj, pd.Series):
        v = obj.get(ts, np.nan)
        if pd.isna(v):
            try:
                idx = pd.to_datetime(obj.index, errors="coerce", utc=True)
                loc = np.flatnonzero(idx == ts)
                if len(loc):
                    v = obj.iloc[int(loc[0])]
            except Exception:
                pass
        return float(v) if pd.notna(v) else np.nan
    arr = np.asarray(obj)
    return float(arr[i]) if i < len(arr) and pd.notna(arr[i]) else np.nan


def _label_at(labels: pd.DataFrame | None, ts: pd.Timestamp):
    if labels is None:
        return None
    if ts in labels.index:
        row = labels.loc[ts]
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row
    try:
        idx = pd.to_datetime(labels.index, errors="coerce", utc=True)
        loc = np.flatnonzero(idx == ts)
        if len(loc):
            return labels.iloc[int(loc[0])]
    except Exception:
        pass
    return None


def _close_record(pos: Position, raw_price: float, ts: pd.Timestamp, config: BacktestConfig, reason: str) -> dict:
    action = "sell" if pos.side == 1 else "buy"
    px = execution_price(raw_price, pos.side, action, config.slippage)
    gross = gross_pnl(pos.side, pos.qty, pos.entry, px)
    fees = round_trip_fees(pos.qty, pos.entry, px, config.commission)
    return {
        "signal_time": pos.signal_time, "entry_time": pos.entry_time, "exit_time": ts,
        "side": pos.side, "side_name": "LONG" if pos.side == 1 else "SHORT",
        "qty": pos.qty, "entry": pos.entry, "exit": px,
        "entry_notional": abs(pos.qty * pos.entry), "exit_notional": abs(pos.qty * px),
        "target": pos.target, "tp": pos.tp, "sl": pos.sl,
        "gross_pnl": gross, "fees": fees, "net_pnl": gross - fees,
        "reason": reason, "bars_held": pos.bars_held,
    }


def _unrealized(pos: Position, price: float, commission: float) -> float:
    return gross_pnl(pos.side, pos.qty, pos.entry, price) - abs(pos.qty * pos.entry) * commission


def backtest(
    bars: pd.DataFrame,
    signals: pd.Series | np.ndarray,
    target: pd.Series | np.ndarray | None = None,
    config: BacktestConfig | None = None,
    *,
    event_labels: pd.DataFrame | None = None,
    sizer: Any = None,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    if cfg.barrier_mode not in {"target", "fixed_pct", "event_labels"}:
        raise ValueError("barrier_mode must be target, fixed_pct, or event_labels")
    if cfg.vertical_horizon < 1 or cfg.commission < 0 or cfg.slippage < 0:
        raise ValueError("invalid horizon, commission or slippage")
    if cfg.max_gross_leverage <= 0:
        raise ValueError("max_gross_leverage must be positive")
    if cfg.sizing_mode not in {"fixed_fraction", "risk_fraction"}:
        raise ValueError("sizing_mode must be fixed_fraction or risk_fraction")
    if cfg.sizing_mode == "fixed_fraction" and not (0 < cfg.position_fraction <= 1):
        raise ValueError("position_fraction must be in (0, 1]")
    if cfg.sizing_mode == "risk_fraction" and not (0 < cfg.risk_fraction <= 0.10):
        raise ValueError("risk_fraction must be in (0, 0.10]")
    if cfg.barrier_mode == "target" and target is None:
        raise ValueError("target is required for barrier_mode='target'")
    if cfg.barrier_mode == "event_labels" and (event_labels is None or not {"upper", "lower", "t1"}.issubset(event_labels.columns)):
        raise ValueError("event_labels mode requires upper/lower/t1")

    b = _bars(bars)
    if sizer is None:
        sizer = RiskFractionSizer(cfg.risk_fraction) if cfg.sizing_mode == "risk_fraction" else FixedFractionSizer(cfg.position_fraction)
    capital = float(initial_capital)
    positions: list[Position] = []
    pending = None
    trades: list[dict] = []
    eq_vals, eq_times = [], []
    diag = {"long_signals":0, "short_signals":0, "entries":0, "rejected_exposure_cap":0,
            "tp_exits":0, "sl_exits":0, "vertical_exits":0, "forced_exits":0,
            "reversals":0, "same_direction_additions":0, "holds":0,
            "max_gross_exposure":0.0, "same_bar_execution_violations":0}

    def close_one(pos, raw, ts, reason):
        nonlocal capital, positions
        rec = _close_record(pos, raw, ts, cfg, reason)
        capital += rec["net_pnl"]
        trades.append(rec)
        key = {"take_profit":"tp_exits", "stop_loss":"sl_exits", "vertical_barrier":"vertical_exits",
               "forced_close":"forced_exits", "reverse":"reversals"}.get(reason)
        if key:
            diag[key] += 1
        positions.remove(pos)

    def mark_equity(price):
        return capital + sum(_unrealized(p, price, cfg.commission) for p in positions)

    def gross_exposure(price):
        return sum(abs(p.qty * price) for p in positions)

    for i, row in b.iterrows():
        ts = row.bar_ts
        hour = _session_hour(ts, cfg)

        # Signal from prior bar close executes at this bar's OPEN.
        if pending is not None:
            sig = int(pending["signal"])
            signal_time = pending["signal_time"]
            label = pending.get("label")
            tgt = pending.get("target", np.nan)
            signal_close = pending.get("signal_close", np.nan)

            if sig and positions and cfg.close_opposite_on_signal:
                for pos in list(positions):
                    if pos.side != sig:
                        close_one(pos, float(row.OPEN), ts, "reverse")

            can_enter = (
                sig != 0
                and (cfg.entry_cutoff_hour is None or hour < cfg.entry_cutoff_hour)
                and (cfg.forced_close_hour is None or hour < cfg.forced_close_hour)
            )
            if can_enter:
                entry = execution_price(row.OPEN, sig, "buy" if sig == 1 else "sell", cfg.slippage)
                vertical_exit_time = None
                if cfg.barrier_mode == "target":
                    if not np.isfinite(tgt) or tgt <= 0:
                        can_enter = False
                        tp = sl = np.nan
                    else:
                        tp = entry * (1 + cfg.pt_mult * tgt) if sig == 1 else entry * (1 - cfg.pt_mult * tgt)
                        sl = entry * (1 - cfg.sl_mult * tgt) if sig == 1 else entry * (1 + cfg.sl_mult * tgt)
                elif cfg.barrier_mode == "fixed_pct":
                    tp = entry * (1 + cfg.tp_pct) if sig == 1 else entry * (1 - cfg.tp_pct)
                    sl = entry * (1 - cfg.sl_pct) if sig == 1 else entry * (1 + cfg.sl_pct)
                else:
                    if label is None or not np.isfinite(signal_close):
                        can_enter = False
                        tp = sl = np.nan
                    else:
                        upper, lower = float(label["upper"]), float(label["lower"])
                        up_dist, down_dist = upper - float(signal_close), float(signal_close) - lower
                        if not (np.isfinite(up_dist) and np.isfinite(down_dist) and up_dist > 0 and down_dist > 0):
                            can_enter = False
                            tp = sl = np.nan
                        else:
                            tp = entry + up_dist if sig == 1 else entry - down_dist
                            sl = entry - down_dist if sig == 1 else entry + up_dist
                            vertical_exit_time = pd.to_datetime(label["t1"], errors="coerce", utc=True)
                            if pd.isna(vertical_exit_time):
                                vertical_exit_time = None

                if can_enter and np.isfinite(tp) and np.isfinite(sl) and tp > 0 and sl > 0 and tp != sl:
                    equity_at_open = mark_equity(float(row.OPEN))
                    try:
                        qty = float(sizer.size(equity_at_open, entry, sig, stop_price=sl))
                    except TypeError:
                        qty = float(sizer.size(equity_at_open, entry, sig))
                    if np.isfinite(qty) and qty > 0 and equity_at_open > 0:
                        current_exposure = gross_exposure(entry)
                        cap = equity_at_open * cfg.max_gross_leverage
                        available = max(0.0, cap - current_exposure)
                        wanted = qty * entry
                        if wanted > available and wanted > 0:
                            qty *= available / wanted
                        if qty > 0 and qty * entry > 1e-10:
                            positions.append(Position(sig, qty, signal_time, ts, entry, float(tgt) if np.isfinite(tgt) else np.nan,
                                                      float(tp), float(sl), vertical_exit_time))
                            diag["entries"] += 1
                            if any(p.side == sig for p in positions[:-1]):
                                diag["same_direction_additions"] += 1
                        else:
                            diag["rejected_exposure_cap"] += 1
                    else:
                        diag["rejected_exposure_cap"] += 1
            pending = None

        # Manage every open position independently through current bar OHLC.
        for pos in list(positions):
            pos.bars_held += 1
            if cfg.forced_close_hour is not None and hour >= cfg.forced_close_hour:
                close_one(pos, float(row.CLOSE), ts, "forced_close")
                continue
            hit_sl = float(row.LOW) <= pos.sl if pos.side == 1 else float(row.HIGH) >= pos.sl
            hit_tp = float(row.HIGH) >= pos.tp if pos.side == 1 else float(row.LOW) <= pos.tp
            if hit_sl:  # conservative: stop wins when both barriers are touched in one OHLC bar
                close_one(pos, pos.sl, ts, "stop_loss")
            elif hit_tp:
                close_one(pos, pos.tp, ts, "take_profit")
            elif pos.vertical_exit_time is not None and ts >= pos.vertical_exit_time:
                close_one(pos, float(row.CLOSE), ts, "vertical_barrier")
            elif pos.bars_held >= cfg.vertical_horizon:
                close_one(pos, float(row.CLOSE), ts, "vertical_barrier")

        eq = mark_equity(float(row.CLOSE))
        eq_times.append(ts)
        eq_vals.append(eq)
        diag["max_gross_exposure"] = max(diag["max_gross_exposure"], gross_exposure(float(row.CLOSE)))

        # Generate signal at this close for next bar open.
        if i < len(b) - 1:
            raw = _lookup(signals, i, ts)
            sig = int(np.sign(raw)) if np.isfinite(raw) else 0
            if sig not in (-1, 0, 1):
                raise ValueError(f"Invalid signal {sig} at {ts}")
            if sig == 1: diag["long_signals"] += 1
            elif sig == -1: diag["short_signals"] += 1
            if sig:
                label = _label_at(event_labels, ts) if cfg.barrier_mode == "event_labels" else None
                pending = {"signal":sig, "signal_time":ts, "target":_lookup(target, i, ts),
                           "label":label, "signal_close":float(row.CLOSE)}

    # Close any remaining positions at the last bar close.
    if positions:
        last = b.iloc[-1]
        for pos in list(positions):
            close_one(pos, float(last.CLOSE), last.bar_ts, "end_of_test")
        if eq_vals:
            eq_vals[-1] = capital

    equity = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_times), name="equity")
    trades_df = pd.DataFrame(trades)
    metrics = calculate_metrics(equity, trades_df, cfg.periods_per_year)
    metrics.update({
        "initial_capital": float(initial_capital),
        "final_equity": float(capital),
        "total_return": float(capital / initial_capital - 1.0) if initial_capital else np.nan,
        "max_gross_leverage_observed": float(diag["max_gross_exposure"] / initial_capital) if initial_capital else np.nan,
    })
    diag.update({"initial_capital":float(initial_capital), "final_capital":float(capital),
                 "bars":int(len(b)), "trades":int(len(trades_df)),
                 "execution_rule":"signal_at_T_close_execute_at_T_plus_1_open",
                 "both_tp_sl_rule":"stop_first", "barrier_mode":cfg.barrier_mode,
                 "sizing_mode":cfg.sizing_mode, "position_fraction":cfg.position_fraction,
                 "risk_fraction":cfg.risk_fraction, "max_gross_leverage":cfg.max_gross_leverage})
    return BacktestResult(trades_df, equity, metrics, diag)


def _metadata_feature_names(metadata: Any) -> list[str] | None:
    if not isinstance(metadata, dict):
        return None
    for key in ("feature_names", "features", "feature_order", "model_features", "feature_columns"):
        val = metadata.get(key)
        if isinstance(val, list) and all(isinstance(v, str) for v in val):
            return val
    return None


def backtest_saved_catboost(
    bars: pd.DataFrame, X: pd.DataFrame, model_path: str | Path,
    metadata_path: str | Path | None = None, target=None,
    config: BacktestConfig | None = None, *, event_labels: pd.DataFrame | None = None,
    initial_capital: float = 100_000.0, sizer: Any = None,
) -> BacktestResult:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("Install catboost in the active Jupyter Python environment") from exc
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"CatBoost model not found: {model_file}")
    model = CatBoostClassifier()
    model.load_model(str(model_file))
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a DataFrame with feature names")
    feature_names = list(model.feature_names_ or [])
    if not feature_names:
        raise ValueError("Model contains no feature names; cannot safely validate feature order")
    if metadata_path is not None:
        meta_file = Path(metadata_path)
        if not meta_file.is_file():
            raise FileNotFoundError(f"Metadata JSON not found: {meta_file}")
        with meta_file.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        metadata_features = _metadata_feature_names(metadata)
        if metadata_features is not None and metadata_features != feature_names:
            raise ValueError("Feature order in metadata JSON differs from the feature order embedded in the model")
    missing = [name for name in feature_names if name not in X.columns]
    if missing:
        raise ValueError(f"Missing model features ({len(missing)}): {missing[:20]}")
    X_ordered = X.loc[:, feature_names].copy().replace([np.inf, -np.inf], np.nan)
    if "bar_ts" in X_ordered.columns:
        raise ValueError("bar_ts must not be an ML feature")
    if isinstance(X_ordered.index, pd.DatetimeIndex):
        times = pd.to_datetime(X_ordered.index, errors="coerce", utc=True)
    elif "ts" in X.columns:
        times = pd.to_datetime(X["ts"], errors="coerce", utc=True)
    else:
        raise ValueError("X must have a DatetimeIndex aligned to bar timestamps")
    if pd.isna(times).any() or times.duplicated().any():
        raise ValueError("X timestamp index contains NaT or duplicates")
    pred = np.asarray(model.predict(X_ordered)).reshape(-1)
    pred_num = pd.to_numeric(pd.Series(pred), errors="coerce")
    if pred_num.isna().any() or not set(pred_num.astype(int).unique()).issubset({0,1}):
        raise ValueError(f"Expected binary DOWN=0 / UP=1 predictions; got {pd.Series(pred).unique()[:10]}")
    sig = pd.Series(np.where(pred_num.astype(int).to_numpy() == 1, 1, -1), index=times, name="signal")
    return backtest(bars, sig, target=target, config=config, event_labels=event_labels,
                    sizer=sizer, initial_capital=initial_capital)
