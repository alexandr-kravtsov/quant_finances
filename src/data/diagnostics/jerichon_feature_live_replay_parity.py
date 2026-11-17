"""
JERICHON — LIVE ↔ REPLAY FEATURE PARITY AUDIT
==============================================

Цель
----
Проверить 475 признаков не только статическим поиском leakage, а через
фактическое наблюдение за live-пайплайном:

    LIVE MARKET
        ↓
    local AlgoPack history files (as they arrive)
        ↓
    exact 475-feature reconstruction
        ↓
    immutable journal: one row per completed dollar bar

После рынка:

    FINAL HISTORY
        ↓
    exact same reconstruction
        ↓
    compare with LIVE JOURNAL

Если для одного и того же dollar-bar timestamp значения 475 признаков
совпадают (с заданным tolerance), это сильное подтверждение того, что
replay не использует данные, которых не было в live-срезе.

ВАЖНО
------
Это НЕ математическое доказательство отсутствия leakage. Оно проверяет
главное эмпирическое свойство: live/replay parity. Отдельно остаются:
- правильность временной семантики SYSTIME / availability;
- корректность самого источника данных;
- потенциальные leakage внутри внешних builder-классов;
- изменения/исправления исторических данных после live-съёма.

Использование Windows CMD
-------------------------
1) Во время рынка:
   python jerichon_feature_live_replay_parity.py capture --ticker SMLT

2) После рынка / на полной истории:
   python jerichon_feature_live_replay_parity.py replay --ticker SMLT

3) Можно указать другой ROOT:
   python jerichon_feature_live_replay_parity.py capture --ticker SMLT --root "C:\\...\\JERICHON\\data"

Файлы journal сохраняются в:
    <root>/for_ml/storage/diagnostics/live_feature_parity/<ticker>/

По умолчанию сохраняется только последний новый dollar bar на каждом poll.
Это делает журнал небольшим и при этом позволяет сравнить весь live-путь.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Defaults — совпадают с текущим causal notebook.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = PROJECT_ROOT / "data"
DOLLAR_THRESHOLD = 13_000_000
POLL_SECONDS = 5.0
FEATURE_TOL_ABS = 1e-9
FEATURE_TOL_REL = 1e-7

JOURNAL_COLUMNS_FIRST = [
    "bar_ts",
    "captured_at_utc",
    "source_latest_ts",
    "source_rows",
    "feature_count",
]


# -----------------------------------------------------------------------------
# Imports from the real project.
# -----------------------------------------------------------------------------
def load_project_imports(root: Path):
    root = Path(root)
    project_root = root.parent if root.name.lower() == "data" else root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data.builders.dollar_bars import InformationDrivenBars
    from src.data.builders.micro_features import MicroFeatures
    from src.data.builders.build_features import FeatureBuilder

    return InformationDrivenBars, MicroFeatures, FeatureBuilder


# -----------------------------------------------------------------------------
# Raw AlgoPack local history loader.
# This intentionally uses the same storage contract as the current notebook.
# -----------------------------------------------------------------------------
def available_dates(root: Path, ticker: str) -> list[str]:
    dates = set()
    for kind in ["realtime", "supercandles"]:
        base = root / "for_ml" / "storage" / "history" / kind / ticker
        if base.exists():
            for p in base.iterdir():
                if p.is_dir():
                    dates.add(p.name)
    return sorted(dates)


def load_history(root: Path, ticker: str) -> dict[str, pd.DataFrame]:
    result = {
        "candles": [],
        "trades": [],
        "orderbook": [],
        "tradestats": [],
        "obstats": [],
        "orderstats": [],
    }

    rt_base = root / "for_ml" / "storage" / "history" / "realtime"
    sc_base = root / "for_ml" / "storage" / "history" / "supercandles"

    for date in available_dates(root, ticker):
        rt_dir = rt_base / ticker / date
        sc_dir = sc_base / ticker / date

        for key in ["candles", "trades", "orderbook"]:
            path = rt_dir / f"{key}.parquet"
            if path.exists():
                try:
                    x = pd.read_parquet(path)
                    if not x.empty:
                        x["date"] = date
                        result[key].append(x)
                except Exception as exc:
                    logging.warning("Cannot read %s: %s", path, exc)

        for key in ["tradestats", "obstats", "orderstats"]:
            path = sc_dir / f"{key}.parquet"
            if path.exists():
                try:
                    x = pd.read_parquet(path)
                    if not x.empty:
                        x["date"] = date
                        result[key].append(x)
                except Exception as exc:
                    logging.warning("Cannot read %s: %s", path, exc)

    out = {}
    for key, frames in result.items():
        if frames:
            out[key] = pd.concat(frames, ignore_index=True)
        else:
            out[key] = pd.DataFrame()

        if "ts" in out[key].columns and not out[key].empty:
            out[key]["ts"] = pd.to_datetime(
                out[key]["ts"], errors="coerce", utc=True
            ).dt.tz_convert("Europe/Moscow")

    return out


# -----------------------------------------------------------------------------
# Exact helper functions copied from the current feature construction logic.
# -----------------------------------------------------------------------------
def prep_exact(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    x = df.copy()
    if "ts" not in x.columns:
        return pd.DataFrame()

    x["ts"] = pd.to_datetime(x["ts"], errors="coerce")

    for c in x.columns:
        if c not in {"ts", "ticker", "secid", "super_kind", "date", "SYSTIME"}:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    return x.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def prepare_ts(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "ts" not in df.columns:
        return pd.DataFrame()

    x = df.copy()
    x["ts"] = pd.to_datetime(x["ts"], errors="coerce")

    if x["ts"].dt.tz is None:
        x["ts"] = x["ts"].dt.tz_localize("Europe/Moscow")
    else:
        x["ts"] = x["ts"].dt.tz_convert("Europe/Moscow")

    return x.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def safe_div(a, b, eps=1e-12):
    return a / (b.abs() + eps)


def imb(a, b, eps=1e-12):
    return (a - b) / (a + b).abs().add(eps)


def add_dyn(df, cols, windows=(3, 6, 12, 24)):
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        out[f"{c}_chg"] = s.diff()
        out[f"{c}_ret"] = s.pct_change().replace([np.inf, -np.inf], np.nan)
        for w in windows:
            minp = max(2, min(w, 3))
            mean = s.rolling(w, min_periods=minp).mean()
            std = s.rolling(w, min_periods=minp).std()
            out[f"{c}_ma{w}"] = mean
            out[f"{c}_z{w}"] = (s - mean) / (std + 1e-12)
    return out


def causal_merge(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if base is None or base.empty:
        return pd.DataFrame()
    if extra is None or extra.empty:
        return base.copy()

    b = base.copy()
    e = extra.copy()
    b["ts"] = pd.to_datetime(b["ts"], errors="coerce")
    e["ts"] = pd.to_datetime(e["ts"], errors="coerce")
    b = b.dropna(subset=["ts"]).sort_values("ts")
    e = e.dropna(subset=["ts"]).sort_values("ts")

    cols = [
        c for c in e.columns
        if c != "ts" and pd.api.types.is_numeric_dtype(e[c])
    ]
    if not cols:
        return b

    e = e[["ts"] + cols].copy()
    e = e.drop(columns=[c for c in cols if c in b.columns], errors="ignore")
    return pd.merge_asof(b, e, on="ts", direction="backward")


# -----------------------------------------------------------------------------
# Dollar bars — same contract as causal notebook.
# -----------------------------------------------------------------------------
def build_dollar_bars(data: dict[str, pd.DataFrame], InformationDrivenBars):
    trades = data.get("trades", pd.DataFrame()).copy()
    if trades.empty:
        return pd.DataFrame()

    required = ["PRICE", "QUANTITY", "VALUE", "ts"]
    missing = [c for c in required if c not in trades.columns]
    if missing:
        raise RuntimeError(f"trades missing required columns: {missing}")

    trades["ts"] = pd.to_datetime(trades["ts"], errors="coerce")
    for c in ["PRICE", "QUANTITY", "VALUE"]:
        trades[c] = pd.to_numeric(trades[c], errors="coerce")

    trades = trades.dropna(subset=required)
    trades = trades[
        (trades["PRICE"] > 0)
        & (trades["QUANTITY"] > 0)
        & (trades["VALUE"] > 0)
    ].copy()
    if trades.empty:
        return pd.DataFrame()

    trades_for_bars = pd.DataFrame({
        "ts": trades["ts"],
        "price": trades["PRICE"],
        "quantity": trades["VALUE"] / trades["PRICE"],
    }).dropna()

    trades_for_bars = trades_for_bars[
        (trades_for_bars["price"] > 0)
        & (trades_for_bars["quantity"] > 0)
    ]

    builder = InformationDrivenBars()
    bars = builder.build_dollar(
        trades_for_bars,
        threshold=DOLLAR_THRESHOLD,
    )
    if bars is None or bars.empty:
        return pd.DataFrame()

    bars = bars.copy()
    bars["ts"] = pd.to_datetime(bars["close_time"], errors="coerce")
    if bars["ts"].dt.tz is None:
        bars["ts"] = bars["ts"].dt.tz_localize("Europe/Moscow")
    else:
        bars["ts"] = bars["ts"].dt.tz_convert("Europe/Moscow")

    bars = bars.dropna(subset=["ts"])
    bars = bars.set_index("ts").sort_index()
    bars = bars[~bars.index.duplicated(keep="first")]
    bars = bars.rename(columns={
        "open": "OPEN",
        "high": "HIGH",
        "low": "LOW",
        "close": "CLOSE",
        "volume": "VOLUME",
    })
    return bars


# -----------------------------------------------------------------------------
# Exact 73 + V2 feature construction, producing the full numeric matrix.
# -----------------------------------------------------------------------------
def build_475_features(data, dollar_bars, MicroFeatures, FeatureBuilder):
    if dollar_bars.empty:
        return pd.DataFrame()

    # ---- 73 base features ---------------------------------------------------
    bars_lower = dollar_bars.rename(columns={
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "VOLUME": "volume",
    })

    mf = MicroFeatures(
        spread_window=20,
        volatility_window=20,
        intensity_window=20,
    )
    bars_with_micro = mf.build_all_features(bars_lower)

    fb = FeatureBuilder(
        windows=[5, 10, 20],
        lags=[1, 2, 3],
        include_micro=False,
    )
    bars_featured = fb.build_all(bars_with_micro)
    base_numeric = bars_featured.select_dtypes(include=[np.number]).copy()

    # Keep exact index -> ts representation.
    if isinstance(base_numeric.index, pd.DatetimeIndex):
        base_numeric = base_numeric.copy()
        base_numeric.index = pd.to_datetime(base_numeric.index)
        base_numeric.index.name = "ts"
        base_numeric = base_numeric.reset_index()
    elif "ts" not in base_numeric.columns:
        raise RuntimeError("FeatureBuilder output has neither DatetimeIndex nor ts")

    # ---- V2 source features -------------------------------------------------
    ts0 = prep_exact(data.get("tradestats", pd.DataFrame()))
    os0 = prep_exact(data.get("orderstats", pd.DataFrame()))
    obs0 = prepare_ts(data.get("obstats", pd.DataFrame()))

    frames = []

    # GROUP 10 — tradestats
    if not ts0.empty:
        tsf = ts0[["ts"]].copy()
        tsf["ts_trade_imbalance"] = imb(ts0["trades_b"], ts0["trades_s"])
        tsf["ts_volume_imbalance"] = imb(ts0["vol_b"], ts0["vol_s"])
        tsf["ts_value_imbalance"] = imb(ts0["val_b"], ts0["val_s"])
        tsf["ts_buy_trade_share"] = safe_div(ts0["trades_b"], ts0["trades_b"] + ts0["trades_s"])
        tsf["ts_buy_volume_share"] = safe_div(ts0["vol_b"], ts0["vol_b"] + ts0["vol_s"])
        tsf["ts_buy_value_share"] = safe_div(ts0["val_b"], ts0["val_b"] + ts0["val_s"])
        tsf["ts_vwap_b_s_diff"] = ts0["pr_vwap_b"] - ts0["pr_vwap_s"]
        tsf["ts_vwap_b_s_bps"] = (
            (ts0["pr_vwap_b"] - ts0["pr_vwap_s"])
            / ((ts0["pr_vwap_b"] + ts0["pr_vwap_s"]) / 2 + 1e-12)
            * 1e4
        )
        tsf["ts_buy_vwap_displacement"] = (ts0["pr_vwap_b"] - ts0["pr_vwap"]) / (ts0["pr_vwap"] + 1e-12)
        tsf["ts_sell_vwap_displacement"] = (ts0["pr_vwap_s"] - ts0["pr_vwap"]) / (ts0["pr_vwap"] + 1e-12)
        tsf["ts_value_per_trade"] = safe_div(ts0["val"], ts0["trades"])
        tsf["ts_volume_per_trade"] = safe_div(ts0["vol"], ts0["trades"])
        tsf["ts_flow_x_return"] = ts0["disb"] * ts0["pr_change"]
        tsf["ts_flow_x_volatility"] = ts0["disb"] * ts0["pr_std"]
        dyn = [
            "ts_trade_imbalance", "ts_volume_imbalance", "ts_value_imbalance",
            "ts_buy_volume_share", "ts_buy_value_share", "ts_vwap_b_s_bps",
            "ts_value_per_trade", "ts_flow_x_return",
        ]
        tsf = add_dyn(tsf, dyn, windows=(3, 6, 12, 24))
        frames.append(tsf)

    # GROUP 11 — orderstats
    if not os0.empty:
        osf = os0[["ts"]].copy()
        osf["os_put_orders_imb"] = imb(os0["put_orders_b"], os0["put_orders_s"])
        osf["os_put_volume_imb"] = imb(os0["put_vol_b"], os0["put_vol_s"])
        osf["os_put_value_imb"] = imb(os0["put_val_b"], os0["put_val_s"])
        osf["os_put_buy_share"] = safe_div(os0["put_vol_b"], os0["put_vol_b"] + os0["put_vol_s"])
        osf["os_put_value_buy_share"] = safe_div(os0["put_val_b"], os0["put_val_b"] + os0["put_val_s"])
        osf["os_cancel_orders_imb"] = imb(os0["cancel_orders_b"], os0["cancel_orders_s"])
        osf["os_cancel_volume_imb"] = imb(os0["cancel_vol_b"], os0["cancel_vol_s"])
        osf["os_cancel_value_imb"] = imb(os0["cancel_val_b"], os0["cancel_val_s"])
        osf["os_cancel_buy_share"] = safe_div(os0["cancel_vol_b"], os0["cancel_vol_b"] + os0["cancel_vol_s"])
        osf["os_cancel_to_put_orders"] = safe_div(os0["cancel_orders"], os0["put_orders"])
        osf["os_cancel_to_put_volume"] = safe_div(os0["cancel_vol"], os0["put_vol"])
        osf["os_cancel_to_put_value"] = safe_div(os0["cancel_val"], os0["put_val"])
        osf["os_net_orders"] = os0["put_orders"] - os0["cancel_orders"]
        osf["os_net_volume"] = os0["put_vol"] - os0["cancel_vol"]
        osf["os_net_value"] = os0["put_val"] - os0["cancel_val"]
        osf["os_net_volume_imb"] = imb(os0["put_vol_b"] - os0["cancel_vol_b"], os0["put_vol_s"] - os0["cancel_vol_s"])
        osf["os_net_value_imb"] = imb(os0["put_val_b"] - os0["cancel_val_b"], os0["put_val_s"] - os0["cancel_val_s"])
        osf["os_put_vwap_b_s_bps"] = (
            (os0["put_vwap_b"] - os0["put_vwap_s"])
            / ((os0["put_vwap_b"] + os0["put_vwap_s"]) / 2 + 1e-12) * 1e4
        )
        osf["os_cancel_vwap_b_s_bps"] = (
            (os0["cancel_vwap_b"] - os0["cancel_vwap_s"])
            / ((os0["cancel_vwap_b"] + os0["cancel_vwap_s"]) / 2 + 1e-12) * 1e4
        )
        osf["os_cancel_vs_put_vwap_b"] = (os0["cancel_vwap_b"] - os0["put_vwap_b"]) / (os0["put_vwap_b"] + 1e-12)
        osf["os_cancel_vs_put_vwap_s"] = (os0["cancel_vwap_s"] - os0["put_vwap_s"]) / (os0["put_vwap_s"] + 1e-12)
        dyn = [
            "os_put_orders_imb", "os_put_volume_imb", "os_put_value_imb",
            "os_cancel_orders_imb", "os_cancel_volume_imb", "os_cancel_value_imb",
            "os_cancel_to_put_orders", "os_cancel_to_put_volume", "os_cancel_to_put_value",
            "os_net_volume", "os_net_value", "os_net_volume_imb", "os_net_value_imb",
            "os_put_vwap_b_s_bps", "os_cancel_vwap_b_s_bps",
        ]
        osf = add_dyn(osf, dyn, windows=(3, 6, 12, 24))
        frames.append(osf)

    # GROUP 12 — obstats
    if not obs0.empty:
        obsf = pd.DataFrame({"ts": obs0["ts"]})
        obsf["ob_total_volume"] = obs0["vol_b"].fillna(0) + obs0["vol_s"].fillna(0)
        obsf["ob_total_value"] = obs0["val_b"].fillna(0) + obs0["val_s"].fillna(0)
        obsf["ob_vol_imb_bbo_vs_deep"] = obs0["imbalance_vol_bbo"] - obs0["imbalance_vol"]
        obsf["ob_val_imb_bbo_vs_deep"] = obs0["imbalance_val_bbo"] - obs0["imbalance_val"]
        obsf["ob_bid_value_share"] = obs0["val_b"] / (obs0["val_b"] + obs0["val_s"] + 1e-12)
        obsf["ob_bid_volume_share"] = obs0["vol_b"] / (obs0["vol_b"] + obs0["vol_s"] + 1e-12)
        obsf["ob_bid_level_share"] = obs0["levels_b"] / (obs0["levels_b"] + obs0["levels_s"] + 1e-12)
        obsf["ob_lv10_to_bbo_spread_ratio"] = obs0["spread_lv10"] / (obs0["spread_bbo"] + 1e-12)
        obsf["ob_1mio_to_bbo_spread_ratio"] = obs0["spread_1mio"] / (obs0["spread_bbo"] + 1e-12)
        obsf["ob_vwap_b_s_bps"] = (obs0["vwap_b"] - obs0["vwap_s"]) / (obs0["vwap_s"] + 1e-12) * 1e4
        obsf["ob_vwap_1mio_b_s_bps"] = (obs0["vwap_b_1mio"] - obs0["vwap_s_1mio"]) / (obs0["vwap_s_1mio"] + 1e-12) * 1e4
        obsf["ob_1mio_liquidity_cost_bps"] = (obs0["spread_1mio"] - obs0["spread_bbo"]) / (obs0["spread_bbo"] + 1e-12) * 1e4

        px = dollar_bars.reset_index()[["ts", "CLOSE"]].rename(columns={"CLOSE": "close"})
        px["ts"] = pd.to_datetime(px["ts"], errors="coerce")
        # VWAP fields belong to the raw obstats source (obs0), not obsf:
        # obsf contains only derived features, so selecting VWAP columns from
        # obsf raises KeyError. Build the causal price/VWAP join from obs0.
        vwap_cols = ["vwap_b", "vwap_s", "vwap_b_1mio", "vwap_s_1mio"]
        missing_vwap = [c for c in vwap_cols if c not in obs0.columns]
        if missing_vwap:
            raise RuntimeError(
                "AlgoPack obstats is missing required VWAP columns: "
                f"{missing_vwap}. Available columns: {obs0.columns.tolist()}"
            )
        tmp = pd.merge_asof(
            obs0[["ts"] + vwap_cols].sort_values("ts"),
            px.sort_values("ts"),
            on="ts", direction="backward"
        )
        obsf["ob_px_vs_vwap_b_bps"] = (tmp["close"] - tmp["vwap_b"]) / (tmp["vwap_b"] + 1e-12) * 1e4
        obsf["ob_px_vs_vwap_s_bps"] = (tmp["close"] - tmp["vwap_s"]) / (tmp["vwap_s"] + 1e-12) * 1e4
        obsf["ob_px_vs_vwap_b_1mio_bps"] = (tmp["close"] - tmp["vwap_b_1mio"]) / (tmp["vwap_b_1mio"] + 1e-12) * 1e4
        obsf["ob_px_vs_vwap_s_1mio_bps"] = (tmp["close"] - tmp["vwap_s_1mio"]) / (tmp["vwap_s_1mio"] + 1e-12) * 1e4

        dyn = [
            "ob_total_volume", "ob_total_value", "ob_vol_imb_bbo_vs_deep",
            "ob_val_imb_bbo_vs_deep", "ob_bid_value_share", "ob_bid_volume_share",
            "ob_bid_level_share", "ob_lv10_to_bbo_spread_ratio", "ob_1mio_to_bbo_spread_ratio",
            "ob_vwap_b_s_bps", "ob_vwap_1mio_b_s_bps", "ob_1mio_liquidity_cost_bps",
        ]
        obsf = add_dyn(obsf, dyn, windows=(3, 6, 12, 24))
        frames.append(obsf)

    # Cross-source divergence — exact formulas from current notebook.
    cross = None
    for frame in frames:
        if cross is None:
            cross = frame.copy()
        else:
            cross = pd.merge_asof(
                cross.sort_values("ts"), frame.sort_values("ts"),
                on="ts", direction="backward"
            )

    if cross is not None and not cross.empty:
        if "ts_volume_imbalance" in cross and "os_put_volume_imb" in cross:
            cross["x_trade_vs_placement_imb"] = cross["ts_volume_imbalance"] - cross["os_put_volume_imb"]
        if "ts_volume_imbalance" in cross and "os_cancel_volume_imb" in cross:
            cross["x_trade_vs_cancel_imb"] = cross["ts_volume_imbalance"] - cross["os_cancel_volume_imb"]
        if "ts_volume_imbalance" in cross and "os_net_volume_imb" in cross:
            cross["x_trade_vs_net_order_imb"] = cross["ts_volume_imbalance"] - cross["os_net_volume_imb"]
        if "ts_volume_imbalance" in cross and "ob_vol_imb_bbo_vs_deep" in cross:
            cross["x_trade_vs_book_depth_divergence"] = cross["ts_volume_imbalance"] - cross["ob_vol_imb_bbo_vs_deep"]
        if "ts_value_per_trade" in cross and "os_cancel_to_put_value" in cross:
            cross["x_trade_size_x_cancel_pressure"] = (
                np.log1p(cross["ts_value_per_trade"].clip(lower=0))
                * np.log1p(cross["os_cancel_to_put_value"].clip(lower=0))
            )
        xcols = [c for c in cross.columns if c.startswith("x_")]
        if xcols:
            frames.append(cross[["ts"] + xcols])

    # ---- Merge all V2 sources into the dollar-bar matrix --------------------
    matrix = dollar_bars.reset_index()[["ts"] + [c for c in dollar_bars.columns if c != "ts"]].copy()
    matrix["ts"] = pd.to_datetime(matrix["ts"], errors="coerce")
    before = set(matrix.columns)
    for frame in frames:
        matrix = causal_merge(matrix, frame)

    # ---- Add 73 base features by timestamp -------------------------------
    base_numeric["ts"] = pd.to_datetime(base_numeric["ts"], errors="coerce")
    base_numeric = base_numeric.dropna(subset=["ts"]).sort_values("ts")
    matrix = pd.merge_asof(
        matrix.sort_values("ts"),
        base_numeric.sort_values("ts"),
        on="ts", direction="backward", suffixes=("", "_base")
    )

    numeric = [
        c for c in matrix.columns
        if c != "ts" and pd.api.types.is_numeric_dtype(matrix[c])
    ]

    # Preserve deterministic order. The existing notebook ultimately selects
    # all numeric columns from v2_matrix_full.
    matrix = matrix[["ts"] + numeric].copy()
    matrix = matrix.drop_duplicates("ts", keep="first").sort_values("ts")

    return matrix


# -----------------------------------------------------------------------------
# Journal I/O
# -----------------------------------------------------------------------------
def journal_dir(root: Path, ticker: str) -> Path:
    p = root / "for_ml" / "storage" / "diagnostics" / "live_feature_parity" / ticker
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_capture_row(root: Path, ticker: str, row: pd.Series):
    out = journal_dir(root, ticker) / "live_features.parquet"
    row_df = row.to_frame().T
    if out.exists():
        old = pd.read_parquet(out)
        # Do not duplicate a dollar-bar timestamp.
        if "bar_ts" in old.columns and pd.Timestamp(row["bar_ts"]) in pd.to_datetime(old["bar_ts"]).tolist():
            return False
        combined = pd.concat([old, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_parquet(out, index=False)
    return True


def save_meta(root: Path, ticker: str, meta: dict):
    p = journal_dir(root, ticker) / "capture_meta.json"
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_journal(root: Path, ticker: str) -> pd.DataFrame:
    p = journal_dir(root, ticker) / "live_features.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Live journal not found: {p}")
    x = pd.read_parquet(p)
    x["bar_ts"] = pd.to_datetime(x["bar_ts"], errors="coerce")
    return x.dropna(subset=["bar_ts"]).sort_values("bar_ts").drop_duplicates("bar_ts", keep="last")


# -----------------------------------------------------------------------------
# Capture mode.
# -----------------------------------------------------------------------------
def capture(root: Path, ticker: str, poll_seconds: float):
    InformationDrivenBars, MicroFeatures, FeatureBuilder = load_project_imports(root)

    print("=" * 100)
    print("JERICHON — LIVE FEATURE PARITY CAPTURE")
    print("=" * 100)
    print(f"Ticker       : {ticker}")
    print(f"Root         : {root}")
    print(f"Poll         : {poll_seconds:g}s")
    print(f"Dollar thresh: {DOLLAR_THRESHOLD:,} RUB")
    print("STOP: Ctrl+C")
    print()

    while True:
        started = time.time()
        try:
            data = load_history(root, ticker)
            bars = build_dollar_bars(data, InformationDrivenBars)
            if bars.empty:
                print(f"[{datetime.now():%H:%M:%S}] no dollar bars yet")
            else:
                matrix = build_475_features(data, bars, MicroFeatures, FeatureBuilder)
                if matrix.empty:
                    print(f"[{datetime.now():%H:%M:%S}] feature matrix empty")
                else:
                    row = matrix.iloc[-1].copy()
                    bar_ts = pd.Timestamp(row["ts"])
                    feature_cols = [c for c in matrix.columns if c != "ts"]
                    # Fail closed: never create a misleading "475-feature parity"
                    # journal from a partial reconstruction.
                    if len(feature_cols) != 475:
                        raise RuntimeError(
                            f"Feature reconstruction is INCOMPLETE: got {len(feature_cols)} "
                            "features, expected 475. No live row was saved. "
                            "The current script's feature builder must be aligned with "
                            "the notebook's exact v2_matrix_full/X475 construction before "
                            "a valid parity test can run."
                        )
                    feature_values = row[feature_cols]

                    # Replace non-finite values with NaN before journal write.
                    feature_values = feature_values.replace([np.inf, -np.inf], np.nan)

                    source_latest = []
                    source_rows = {}
                    for name, df in data.items():
                        if df is not None and not df.empty and "ts" in df.columns:
                            ts = pd.to_datetime(df["ts"], errors="coerce").dropna()
                            if len(ts):
                                source_latest.append(ts.max())
                            source_rows[name] = int(len(df))

                    latest_source = max(source_latest) if source_latest else pd.NaT

                    journal_row = pd.Series({
                        "bar_ts": bar_ts,
                        "captured_at_utc": pd.Timestamp.now(tz="UTC"),
                        "source_latest_ts": latest_source,
                        "source_rows": json.dumps(source_rows, ensure_ascii=False),
                        "feature_count": len(feature_cols),
                    })
                    for c in feature_cols:
                        journal_row[c] = feature_values[c]

                    inserted = save_capture_row(root, ticker, journal_row)

                    if inserted:
                        print(
                            f"[{datetime.now():%H:%M:%S}] NEW BAR {bar_ts} | "
                            f"features={len(feature_cols)} | "
                            f"source_latest={latest_source}"
                        )
                    else:
                        print(
                            f"[{datetime.now():%H:%M:%S}] heartbeat | "
                            f"last_bar={bar_ts} | features={len(feature_cols)}"
                        )

                    save_meta(root, ticker, {
                        "ticker": ticker,
                        "root": str(root),
                        "dollar_threshold": DOLLAR_THRESHOLD,
                        "feature_count_last_snapshot": len(feature_cols),
                        "last_bar_ts": str(bar_ts),
                        "updated_at_utc": str(pd.Timestamp.now(tz="UTC")),
                        "poll_seconds": poll_seconds,
                    })

        except KeyboardInterrupt:
            print("\nCapture stopped.")
            return
        except Exception as exc:
            logging.exception("Capture error")

        elapsed = time.time() - started
        time.sleep(max(0.0, poll_seconds - elapsed))


# -----------------------------------------------------------------------------
# Replay comparison.
# -----------------------------------------------------------------------------
def replay(root: Path, ticker: str, max_rows: int | None = None):
    InformationDrivenBars, MicroFeatures, FeatureBuilder = load_project_imports(root)

    live = load_journal(root, ticker)
    if max_rows:
        live = live.tail(max_rows).copy()

    print("=" * 100)
    print("JERICHON — LIVE ↔ REPLAY FEATURE PARITY")
    print("=" * 100)
    print(f"Ticker: {ticker}")
    print(f"Live rows in journal: {len(live):,}")

    data = load_history(root, ticker)
    bars = build_dollar_bars(data, InformationDrivenBars)
    replay_matrix = build_475_features(data, bars, MicroFeatures, FeatureBuilder)

    if replay_matrix.empty:
        raise RuntimeError("Replay feature matrix is empty")

    live_feature_cols = [
        c for c in live.columns
        if c not in JOURNAL_COLUMNS_FIRST
    ]
    replay_feature_cols = [c for c in replay_matrix.columns if c != "ts"]

    missing = sorted(set(live_feature_cols) - set(replay_feature_cols))
    extra = sorted(set(replay_feature_cols) - set(live_feature_cols))

    print(f"Replay feature count: {len(replay_feature_cols):,}")
    print(f"Live feature count  : {len(live_feature_cols):,}")

    if missing:
        print(f"\n❌ Missing in replay: {len(missing)}")
        print(missing[:50])
    if extra:
        print(f"\nℹ Extra replay columns: {len(extra)}")
        print(extra[:50])

    common_features = [c for c in live_feature_cols if c in replay_feature_cols]

    replay = replay_matrix.copy()
    replay["bar_ts"] = pd.to_datetime(replay["ts"], errors="coerce")
    replay = replay.drop(columns=["ts"])

    merged = live[["bar_ts"] + common_features].merge(
        replay[["bar_ts"] + common_features],
        on="bar_ts",
        how="inner",
        suffixes=("_live", "_replay"),
    )

    print(f"\nMatched timestamps: {len(merged):,}")
    print(f"Unmatched live timestamps: {len(live) - len(merged):,}")

    if merged.empty:
        raise RuntimeError("No matching timestamps between live journal and replay")

    mismatch_rows = []
    exact_equal = 0
    compared = 0

    for feature in common_features:
        a = pd.to_numeric(merged[f"{feature}_live"], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(merged[f"{feature}_replay"], errors="coerce").to_numpy(dtype=float)

        finite = np.isfinite(a) & np.isfinite(b)
        both_nan = np.isnan(a) & np.isnan(b)
        valid = finite | both_nan

        close = np.zeros_like(valid, dtype=bool)
        close[both_nan] = True
        if finite.any():
            close[finite] = np.isclose(
                a[finite], b[finite],
                rtol=FEATURE_TOL_REL,
                atol=FEATURE_TOL_ABS,
                equal_nan=True,
            )

        compared += int(valid.sum())
        exact_equal += int(close[valid].sum())

        bad = np.where(valid & ~close)[0]
        if len(bad):
            delta = np.abs(a[bad] - b[bad])
            mismatch_rows.append({
                "feature": feature,
                "mismatch_cells": int(len(bad)),
                "max_abs_diff": float(np.nanmax(delta)) if len(delta) else np.nan,
                "mean_abs_diff": float(np.nanmean(delta)) if len(delta) else np.nan,
                "first_bad_ts": str(merged.iloc[bad[0]]["bar_ts"]),
            })

    mismatch_df = pd.DataFrame(mismatch_rows).sort_values(
        ["mismatch_cells", "max_abs_diff"], ascending=False
    ) if mismatch_rows else pd.DataFrame()

    cell_total = max(compared, 1)
    parity_pct = 100.0 * exact_equal / cell_total

    print("\n" + "=" * 100)
    print("PARITY RESULT")
    print("=" * 100)
    print(f"Common features       : {len(common_features):,}")
    print(f"Compared finite/NaN cells: {compared:,}")
    print(f"Matching cells        : {exact_equal:,}")
    print(f"Parity                : {parity_pct:.6f}%")
    print(f"Features with mismatch: {len(mismatch_df):,}")

    out_dir = journal_dir(root, ticker)
    mismatch_path = out_dir / "parity_mismatches.csv"
    summary_path = out_dir / "parity_summary.json"

    if not mismatch_df.empty:
        mismatch_df.to_csv(mismatch_path, index=False, encoding="utf-8-sig")
        print(f"\nMismatch report: {mismatch_path}")
        print("\nTOP MISMATCHES:")
        print(mismatch_df.head(30).to_string(index=False))
    else:
        pd.DataFrame(columns=[
            "feature", "mismatch_cells", "max_abs_diff", "mean_abs_diff", "first_bad_ts"
        ]).to_csv(mismatch_path, index=False, encoding="utf-8-sig")
        print("\n✅ NO FEATURE MISMATCHES WITHIN TOLERANCE")

    summary = {
        "ticker": ticker,
        "live_rows": int(len(live)),
        "matched_timestamps": int(len(merged)),
        "common_features": int(len(common_features)),
        "compared_cells": int(compared),
        "matching_cells": int(exact_equal),
        "parity_pct": float(parity_pct),
        "mismatching_features": int(len(mismatch_df)),
        "tolerance_abs": FEATURE_TOL_ABS,
        "tolerance_rel": FEATURE_TOL_REL,
        "replay_last_bar": str(replay["bar_ts"].max()),
        "live_last_bar": str(live["bar_ts"].max()),
        "generated_at_utc": str(pd.Timestamp.now(tz="UTC")),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if len(common_features) != 475:
        print(
            f"\n⚠️ WARNING: current reconstructed matrix has {len(common_features)} "
            "common features, not 475. Do not interpret this as a clean parity test "
            "until the project environment reproduces exactly 475 columns."
        )
    elif not mismatch_df.empty:
        print(
            "\n❌ PARITY FAILED: at least one of the 475 features changes between "
            "the live snapshot and final replay. Investigate the first mismatch "
            "before any ML optimization."
        )
    else:
        print(
            "\n✅ 475-FEATURE LIVE/REPLAY PARITY PASSED within tolerance."
        )

    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="JERICHON live/replay parity audit for the 475-feature matrix"
    )
    parser.add_argument("mode", choices=["capture", "replay"])
    parser.add_argument("--ticker", default="SMLT")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--poll", type=float, default=POLL_SECONDS)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.mode == "capture":
        capture(args.root, args.ticker, args.poll)
    else:
        replay(args.root, args.ticker, args.max_rows)


if __name__ == "__main__":
    main()
