# -*- coding: utf-8 -*-
"""
JERICHON — Trade Flow Stage 3: Visual Market Regimes & Extremes
================================================================

Purpose
-------
Create price-chart visualizations of four Trade Flow regimes:
    1) high volume pressure / high count activity
    2) high volume pressure / low count activity
    3) low volume pressure / high count activity
    4) low volume pressure / low count activity

The sign of TFI is retained separately:
    positive -> BUYSELL=B encoded flow
    negative -> BUYSELL=S encoded flow
This is NOT labelled aggressive buying/selling unless the source semantics
have independently been verified.

Also marks:
    - prior 10-session high/low levels (calculated only from completed days)
    - intraday local extrema candidates
    - flow/price disagreement ("absorption proxy")
    - forward price response after 1 / 3 / 5 distinct transaction-price changes

Important limitations
---------------------
* Transaction price is not mid-price; bid/ask bounce remains possible.
* TFI is signed executed-trade imbalance, not order-book OFI.
* The four regimes are descriptive bins, not a trading strategy.
* The "absorption" flag is a proxy: strong |TFI| + small price displacement.
* Prior 10-session levels use completed sessions only (no same-day leakage).
* This script does not infer aggressor side from BUYSELL.
* Data are processed one ticker at a time to reduce RAM usage.

Expected raw AlgoPack trades fields (case-insensitive):
    TRADENO, TRADETIME or ts, PRICE, QUANTITY, BUYSELL,
    TRADE_SESSION_DATE / TRADEDATE (optional if date can be inferred)

Expected folder layout:
    <PROJECT_ROOT>/data/for_ml/storage/history/realtime/<TICKER>/<YYYY-MM-DD>/trades.parquet

Run:
    python JERICHON_Trade_Flow_Stage_3_Visual.py

Dependencies:
    pip install pandas numpy pyarrow plotly

Outputs:
    trade_flow_stage3_visual/
        <TICKER>/
            <TICKER>_YYYY-MM-DD.html
            <TICKER>_regime_summary.csv
            <TICKER>_extreme_events.csv
            <TICKER>_daily_levels.csv
        stage3_run_report.txt

If the project folder is not the script's parent, set PROJECT_ROOT below.
"""

from __future__ import annotations

from pathlib import Path
import warnings
import traceback
import numpy as np
import pandas as pd

# ============================ USER CONFIG ============================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Example override:
# PROJECT_ROOT = Path(r"C:\Users\YourName\Desktop\Morgan Co\JERICHON")

HISTORY_ROOT = PROJECT_ROOT / "data" / "for_ml" / "storage" / "history"
OUTPUT_ROOT = PROJECT_ROOT / "research" / "tradeflow" / "mark_4" / "results" / "trade_flow_stage3_visual"

TICKERS = ("SMLT", "VTBR", "GAZP", "ROSN", "SBER")

# Main rolling window. 25 is the conservative baseline from Stage 2.
FLOW_WINDOW = 25

# Use a longer window only as a sensitivity check, not as the primary chart.
SECONDARY_WINDOW = 50

# Regime cutoffs are calculated per trading day from the distribution of
# |TFI| values. 0.50 = median. This keeps bins interpretable across tickers.
PRESSURE_QUANTILE = 0.50
ACTIVITY_QUANTILE = 0.50

# Price chart point density. 1 = every trade; 5/10 is recommended for HTML size.
PLOT_EVERY_N_TRADES = 5

# Local extreme candidates: compare current price to rolling past/future-free
# intraday range. This is a visual candidate marker, not a confirmed pivot.
LOCAL_EXTREME_LOOKBACK_TRADES = 500
LOCAL_EXTREME_DISTANCE_BPS = 8.0

# Prior completed-session levels: high/low over previous N completed sessions.
LEVEL_LOOKBACK_SESSIONS = 10
LEVEL_NEAR_BPS = 15.0

# Absorption proxy: high absolute volume TFI and small price displacement
# over the same rolling window. The threshold is a within-day quantile.
ABSORPTION_TFI_QUANTILE = 0.80
ABSORPTION_MOVE_QUANTILE = 0.30

# Produce charts for every day by default. Set integer (e.g. 3) to limit
# charts per ticker to the most recent N sessions after all days are loaded.
MAX_CHART_DAYS_PER_TICKER = 5

# Optional explicit data directory if history is stored elsewhere:
# HISTORY_ROOT = Path(r"D:\JERICHON\data\for_ml\storage\history")

# ====================================================================

REGIME_NAMES = {
    (1, 1): "HIGH PRESSURE / HIGH ACTIVITY",
    (1, 0): "HIGH PRESSURE / LOW ACTIVITY",
    (0, 1): "LOW PRESSURE / HIGH ACTIVITY",
    (0, 0): "LOW PRESSURE / LOW ACTIVITY",
}


def pick_col(columns, candidates, required=True):
    lookup = {str(c).lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    if required:
        raise KeyError(f"Missing any of {candidates}; available={list(columns)}")
    return None


def discover_days(ticker: str):
    ticker_dir = HISTORY_ROOT / "realtime" / ticker.upper()
    if not ticker_dir.exists():
        raise FileNotFoundError(f"Ticker folder not found: {ticker_dir}")
    days = sorted(
        p.name for p in ticker_dir.iterdir()
        if p.is_dir() and len(p.name) == 10 and p.name[4] == "-" and p.name[7] == "-"
        and (p / "trades.parquet").exists()
    )
    if not days:
        raise FileNotFoundError(f"No daily trades.parquet files in {ticker_dir}")
    return ticker_dir, days


def load_one_day(path: Path, ticker: str, day: str) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    if raw.empty:
        return pd.DataFrame()

    c_id = pick_col(raw.columns, ["TRADENO", "trade_no", "trade_id"], required=False)
    c_ts = pick_col(raw.columns, ["ts", "timestamp", "TRADETIME", "tradetime"])
    c_price = pick_col(raw.columns, ["PRICE", "price"])
    c_size = pick_col(raw.columns, ["QUANTITY", "quantity", "size", "volume"])
    c_value = pick_col(raw.columns, ["VALUE", "value", "notional"], required=False)
    c_side = pick_col(raw.columns, ["BUYSELL", "buysell", "sign", "side"], required=False)
    c_date = pick_col(raw.columns, ["TRADE_SESSION_DATE", "TRADEDATE", "trade_date", "date"],
                      required=False)

    out = pd.DataFrame()
    out["trade_id"] = raw[c_id].to_numpy() if c_id else np.arange(len(raw), dtype=np.int64)
    ts = pd.to_datetime(raw[c_ts], errors="coerce")
    # TRADETIME can be a time-only string; combine it with the folder date.
    if ts.isna().mean() > 0.5:
        time_only = pd.to_datetime(raw[c_ts].astype(str), format="%H:%M:%S", errors="coerce")
        ts = pd.to_datetime(day) + (time_only - time_only.dt.normalize())
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
    out["ts"] = ts
    out["price"] = pd.to_numeric(raw[c_price], errors="coerce")
    out["size"] = pd.to_numeric(raw[c_size], errors="coerce")
    out["notional"] = (
        pd.to_numeric(raw[c_value], errors="coerce").abs()
        if c_value else out["price"] * out["size"]
    )
    if c_side is None:
        out["sign"] = np.nan
    else:
        side_text = raw[c_side].astype("string").str.strip().str.upper()
        if side_text.isin(["B", "S"]).any():
            out["sign"] = side_text.map({"B": 1.0, "S": -1.0})
        else:
            numeric = pd.to_numeric(raw[c_side], errors="coerce")
            out["sign"] = numeric.where(numeric.isin([-1, 1]), np.nan)

    if c_date:
        parsed_date = pd.to_datetime(raw[c_date], errors="coerce")
        out["trade_date"] = parsed_date.dt.strftime("%Y-%m-%d")
    else:
        out["trade_date"] = day

    out["ticker"] = ticker
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["ts", "price", "size"])
    out = out[(out["price"] > 0) & (out["size"] > 0)]
    out["trade_date"] = out["trade_date"].fillna(day).astype(str)
    out["notional"] = out["notional"].fillna(out["price"] * out["size"])
    out["sign"] = out["sign"].where(out["sign"].isin([-1.0, 1.0]), np.nan)
    out = out.sort_values(["ts", "trade_id"], kind="mergesort")
    out = out.drop_duplicates("trade_id", keep="last").reset_index(drop=True)
    out["signed_size"] = out["size"] * out["sign"]
    out["signed_notional"] = out["notional"] * out["sign"]
    return out


def add_flow_features(day: pd.DataFrame) -> pd.DataFrame:
    g = day.sort_values(["ts", "trade_id"], kind="mergesort").copy()
    w = FLOW_WINDOW
    minp = w

    def roll_sum(col):
        return g[col].rolling(w, min_periods=minp).sum()

    g["tfi_volume"] = roll_sum("signed_size") / roll_sum("size").replace(0, np.nan)
    g["tfi_count"] = (
        g["sign"].rolling(w, min_periods=minp).sum()
        / g["sign"].rolling(w, min_periods=minp).count().replace(0, np.nan)
    )
    g["tfi_notional"] = (
        roll_sum("signed_notional") / roll_sum("notional").replace(0, np.nan)
    )
    g["flow_intensity_per_sec"] = (
        (w - 1) / (g["ts"] - g["ts"].shift(w - 1)).dt.total_seconds().replace(0, np.nan)
    )

    # Price displacement over the formation window; only past/current data.
    g["formation_move_bps"] = 1e4 * np.log(g["price"] / g["price"].shift(w - 1))
    g["abs_tfi_volume"] = g["tfi_volume"].abs()
    g["abs_tfi_count"] = g["tfi_count"].abs()

    # Four quadrants: magnitude of volume pressure × actual trade arrival intensity.
    # TFI count remains a separate breadth/directional-confirmation feature.
    p_cut = g["abs_tfi_volume"].quantile(PRESSURE_QUANTILE)
    a_cut = g["flow_intensity_per_sec"].quantile(ACTIVITY_QUANTILE)
    g["pressure_high"] = g["abs_tfi_volume"] >= p_cut
    g["activity_high"] = g["flow_intensity_per_sec"] >= a_cut
    g["regime"] = [
        REGIME_NAMES[(int(p), int(a))]
        for p, a in zip(g["pressure_high"].fillna(False), g["activity_high"].fillna(False))
    ]
    g.loc[g[["tfi_volume", "tfi_count"]].isna().any(axis=1), "regime"] = "WARMUP / UNKNOWN"

    # Flow direction is kept as a separate dimension, not baked into the regime.
    g["flow_direction"] = np.select(
        [g["tfi_volume"] > 0, g["tfi_volume"] < 0],
        ["BUYSELL-B POSITIVE", "BUYSELL-S NEGATIVE"],
        default="NEUTRAL / UNKNOWN",
    )

    # A simple proxy for absorption / price-flow disagreement.
    move_abs = g["formation_move_bps"].abs()
    high_pressure = g["abs_tfi_volume"] >= g["abs_tfi_volume"].quantile(ABSORPTION_TFI_QUANTILE)
    low_move = move_abs <= move_abs.quantile(ABSORPTION_MOVE_QUANTILE)
    g["absorption_proxy"] = high_pressure & low_move & g["tfi_volume"].notna()

    # Forward transaction-price returns after 1/3/5 distinct price changes.
    # Calculated within this session only. These are outcomes, not features.
    p = g["price"].to_numpy(float)
    n = len(p)
    next_change = np.full(n, -1, dtype=np.int64)
    last = -1
    for i in range(n - 2, -1, -1):
        if p[i + 1] != p[i]:
            last = i + 1
        next_change[i] = last if last >= 0 and p[last] != p[i] else -1

    for k in (1, 3, 5):
        target = np.full(n, np.nan)
        for i in range(n):
            j = next_change[i]
            step = 1
            while j >= 0 and step < k:
                j = next_change[j]
                step += 1
            if j >= 0 and step == k:
                target[i] = 1e4 * (p[j] / p[i] - 1.0)
        g[f"future_{k}_price_changes_bps"] = target

    # Prior completed sessions' rolling high/low levels are added later.
    return g


def build_daily_levels(daily_stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, row in daily_stats.iterrows():
        prior = daily_stats.iloc[max(0, i - LEVEL_LOOKBACK_SESSIONS):i]
        rows.append({
            "trade_date": row["trade_date"],
            "prior_10d_high": prior["high"].max() if len(prior) else np.nan,
            "prior_10d_low": prior["low"].min() if len(prior) else np.nan,
            "prior_sessions_used": len(prior),
        })
    return pd.DataFrame(rows)


def add_level_distance(g: pd.DataFrame, level_row: pd.Series) -> pd.DataFrame:
    g = g.copy()
    hi = level_row.get("prior_10d_high", np.nan)
    lo = level_row.get("prior_10d_low", np.nan)
    g["prior_10d_high"] = hi
    g["prior_10d_low"] = lo
    g["dist_prior_high_bps"] = 1e4 * np.log(g["price"] / hi) if pd.notna(hi) and hi > 0 else np.nan
    g["dist_prior_low_bps"] = 1e4 * np.log(g["price"] / lo) if pd.notna(lo) and lo > 0 else np.nan
    g["near_prior_high"] = g["dist_prior_high_bps"].abs() <= LEVEL_NEAR_BPS
    g["near_prior_low"] = g["dist_prior_low_bps"].abs() <= LEVEL_NEAR_BPS
    return g


def make_chart(g: pd.DataFrame, ticker: str, day: str, out_file: Path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plot = g.iloc[::max(1, PLOT_EVERY_N_TRADES)].copy()
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        row_heights=[0.58, 0.22, 0.20],
        subplot_titles=(
            f"{ticker} — transaction price + Trade Flow regimes — {day}",
            "TFI: signed volume pressure and signed count imbalance",
            "Price-flow diagnostics: formation move / future response",
        ),
    )

    fig.add_trace(go.Scatter(
        x=plot["ts"], y=plot["price"], mode="lines",
        name="Transaction price", line=dict(color="#D7DEE8", width=1.2),
        hovertemplate="%{x}<br>Price: %{y}<extra></extra>",
    ), row=1, col=1)

    regime_colors = {
        "HIGH PRESSURE / HIGH ACTIVITY": "#F97316",
        "HIGH PRESSURE / LOW ACTIVITY": "#EAB308",
        "LOW PRESSURE / HIGH ACTIVITY": "#38BDF8",
        "LOW PRESSURE / LOW ACTIVITY": "#94A3B8",
    }
    for regime, color in regime_colors.items():
        z = plot[plot["regime"] == regime]
        if z.empty:
            continue
        fig.add_trace(go.Scatter(
            x=z["ts"], y=z["price"], mode="markers", name=regime,
            marker=dict(color=color, size=4, opacity=0.78),
            customdata=np.column_stack([
                z["tfi_volume"], z["tfi_count"], z["flow_direction"],
                z["flow_intensity_per_sec"], z["formation_move_bps"],
                z["future_3_price_changes_bps"]
            ]),
            hovertemplate=(
                "%{x}<br>Price=%{y}<br>TFI volume=%{customdata[0]:.3f}"
                "<br>TFI count=%{customdata[1]:.3f}<br>%{customdata[2]}"
                "<br>Trade intensity=%{customdata[3]:.2f} trades/sec"
                "<br>Formation move=%{customdata[4]:.2f} bps"
                "<br>Future after 3 price changes=%{customdata[5]:.2f} bps"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    # Direction markers: same regime, but distinguish sign of the flow.
    for label, mask, color, symbol in [
        ("Positive TFI volume", plot["tfi_volume"] > 0, "#22C55E", "triangle-up"),
        ("Negative TFI volume", plot["tfi_volume"] < 0, "#EF4444", "triangle-down"),
    ]:
        z = plot[mask]
        if not z.empty:
            fig.add_trace(go.Scatter(
                x=z["ts"], y=z["price"], mode="markers", name=label,
                marker=dict(color=color, size=5, symbol=symbol, opacity=0.65),
                showlegend=True,
            ), row=1, col=1)

    # Prior completed 10-session levels.
    for col, label, color in [
        ("prior_10d_high", "Prior 10-session high", "#FB7185"),
        ("prior_10d_low", "Prior 10-session low", "#60A5FA"),
    ]:
        value = g[col].iloc[0] if col in g and len(g) else np.nan
        if pd.notna(value):
            fig.add_hline(y=float(value), line_dash="dash", line_color=color,
                          annotation_text=label, row=1, col=1)

    # Mark absorption proxy events and local extremes.
    absorp = plot[plot["absorption_proxy"]]
    if not absorp.empty:
        fig.add_trace(go.Scatter(
            x=absorp["ts"], y=absorp["price"], mode="markers",
            name="Absorption proxy", marker=dict(color="#C084FC", size=9, symbol="x"),
            hovertemplate="Absorption proxy<br>%{x}<br>Price=%{y}<extra></extra>",
        ), row=1, col=1)

    # TFI series. Lines are signed values, so direction is visible.
    fig.add_trace(go.Scatter(
        x=plot["ts"], y=plot["tfi_volume"], mode="lines",
        name="TFI volume (pressure)", line=dict(color="#F59E0B", width=1),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=plot["ts"], y=plot["tfi_count"], mode="lines",
        name="TFI count (activity imbalance)", line=dict(color="#38BDF8", width=1),
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="gray", line_width=1, row=2, col=1)

    fig.add_trace(go.Scatter(
        x=plot["ts"], y=plot["formation_move_bps"], mode="lines",
        name=f"Past {FLOW_WINDOW}-trade move (bps)",
        line=dict(color="#A78BFA", width=1),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=plot["ts"], y=plot["future_3_price_changes_bps"], mode="lines",
        name="Future return after 3 distinct price changes (bps)",
        line=dict(color="#34D399", width=1),
    ), row=3, col=1)
    fig.add_hline(y=0, line_color="gray", line_width=1, row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=1050, width=1500,
        title=f"{ticker} | Trade Flow Stage 3 visual study | {day}",
        hovermode="x unified", legend=dict(orientation="h", y=-0.12),
        margin=dict(l=60, r=30, t=90, b=100),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="TFI [-1, +1]", range=[-1.05, 1.05], row=2, col=1)
    fig.update_yaxes(title_text="Basis points", row=3, col=1)
    fig.write_html(out_file, include_plotlyjs="cdn")


def summarize_day(g: pd.DataFrame, ticker: str, day: str):
    rows = []
    for regime, z in g.groupby("regime", observed=True):
        row = {
            "ticker": ticker, "trade_date": day, "regime": regime,
            "observations": len(z),
            "mean_tfi_volume": z["tfi_volume"].mean(),
            "mean_tfi_count": z["tfi_count"].mean(),
            "mean_trade_intensity_per_sec": z["flow_intensity_per_sec"].mean(),
            "positive_flow_share": (z["tfi_volume"] > 0).mean(),
            "mean_formation_move_bps": z["formation_move_bps"].mean(),
            "mean_future_1_change_bps": z["future_1_price_changes_bps"].mean(),
            "mean_future_3_changes_bps": z["future_3_price_changes_bps"].mean(),
            "mean_future_5_changes_bps": z["future_5_price_changes_bps"].mean(),
            "absorption_proxy_share": z["absorption_proxy"].mean(),
        }
        rows.append(row)
    return rows


def find_local_extreme_candidates(g: pd.DataFrame, ticker: str, day: str):
    # Candidate extrema use a trailing window only, so no future values leak
    # into the event timestamp. The label describes proximity to a past range.
    p = g["price"]
    roll_hi = p.rolling(LOCAL_EXTREME_LOOKBACK_TRADES, min_periods=100).max()
    roll_lo = p.rolling(LOCAL_EXTREME_LOOKBACK_TRADES, min_periods=100).min()
    near_hi = 1e4 * np.log(p / roll_hi).abs() <= LOCAL_EXTREME_DISTANCE_BPS
    near_lo = 1e4 * np.log(p / roll_lo).abs() <= LOCAL_EXTREME_DISTANCE_BPS
    mask = near_hi | near_lo
    z = g.loc[mask, [
        "ts", "price", "tfi_volume", "tfi_count", "regime", "flow_direction",
        "formation_move_bps", "future_1_price_changes_bps",
        "future_3_price_changes_bps", "future_5_price_changes_bps",
        "absorption_proxy", "near_prior_high", "near_prior_low",
    ]].copy()
    if z.empty:
        return []
    z.insert(0, "trade_date", day)
    z.insert(0, "ticker", ticker)
    z["candidate_type"] = np.where(near_hi.loc[z.index], "NEAR TRAILING INTRADAY HIGH",
                                   "NEAR TRAILING INTRADAY LOW")
    return z.to_dict("records")


def run_ticker(ticker: str):
    ticker_dir, days = discover_days(ticker)
    print(f"\n[{ticker}] {len(days)} sessions: {days[0]} -> {days[-1]}")
    daily_stats = []

    # Pass 1: inspect one session at a time; retain only compact daily statistics.
    # Do NOT keep millions of raw trades from all days in RAM simultaneously.
    for day in days:
        path = ticker_dir / day / "trades.parquet"
        raw_day = load_one_day(path, ticker, day)
        if raw_day.empty:
            continue
        daily_stats.append({
            "trade_date": day,
            "high": raw_day["price"].max(),
            "low": raw_day["price"].min(),
            "trades": len(raw_day),
        })
        del raw_day

    if not daily_stats:
        raise RuntimeError(f"{ticker}: no usable trades after normalization")

    daily = pd.DataFrame(daily_stats).sort_values("trade_date").reset_index(drop=True)
    levels = build_daily_levels(daily)
    level_map = levels.set_index("trade_date")

    out_dir = OUTPUT_ROOT / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    regime_rows, extreme_rows = [], []
    chart_days = set(days[-MAX_CHART_DAYS_PER_TICKER:]) if MAX_CHART_DAYS_PER_TICKER else set(days)

    for day in days:
        if day not in level_map.index:
            continue
        path = ticker_dir / day / "trades.parquet"
        raw_day = load_one_day(path, ticker, day)
        if raw_day.empty:
            continue
        g = add_flow_features(raw_day)
        del raw_day
        if day in level_map.index:
            g = add_level_distance(g, level_map.loc[day])
        else:
            g["prior_10d_high"] = np.nan
            g["prior_10d_low"] = np.nan
            g["near_prior_high"] = False
            g["near_prior_low"] = False

        regime_rows.extend(summarize_day(g, ticker, day))
        extreme_rows.extend(find_local_extreme_candidates(g, ticker, day))

        if day in chart_days:
            chart_path = out_dir / f"{ticker}_{day}.html"
            make_chart(g, ticker, day, chart_path)
            print(f"  chart: {chart_path.name}")

    pd.DataFrame(regime_rows).to_csv(
        out_dir / f"{ticker}_regime_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(extreme_rows).to_csv(
        out_dir / f"{ticker}_extreme_events.csv", index=False, encoding="utf-8-sig"
    )
    levels.to_csv(out_dir / f"{ticker}_daily_levels.csv", index=False, encoding="utf-8-sig")
    print(f"  summary rows: {len(regime_rows):,}; extreme candidates: {len(extreme_rows):,}")
    return {
        "ticker": ticker, "sessions": len(days), "trades": int(daily["trades"].sum()),
        "charts_written": len(chart_days), "output": str(out_dir),
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report = []
    failures = []
    for ticker in TICKERS:
        try:
            report.append(run_ticker(ticker))
        except Exception as exc:
            failures.append((ticker, repr(exc)))
            print(f"\nERROR [{ticker}]: {exc}")
            traceback.print_exc()

    report_path = OUTPUT_ROOT / "stage3_run_report.txt"
    lines = ["JERICHON Trade Flow Stage 3 — run report", "=" * 48]
    lines += [str(x) for x in report]
    if failures:
        lines += ["", "FAILURES:"] + [f"{t}: {e}" for t, e in failures]
    lines += [
        "", "Interpretation guardrails:",
        "- Positive/negative TFI refers only to the encoded BUYSELL/sign field.",
        "- TFI is not OFI and BUYSELL is not asserted to be aggressor side.",
        "- Price is transaction price, not mid-price.",
        "- Absorption is a proxy, not confirmed order-book absorption.",
        "- Four regimes = |TFI volume| pressure × trade arrival intensity (trades/sec).",
        "- TFI count is shown separately as signed-count breadth/confirmation, not raw activity.",
        "- Regime thresholds are calculated per session; do not treat them as universal cutoffs.",
        "- Future returns are outcomes for visual diagnosis, never input features.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRun report: {report_path}")
    if failures:
        print("Some tickers failed; inspect report and traceback above.")


if __name__ == "__main__":
    main()
