# -*- coding: utf-8 -*-
"""
JERICHON — Stage 3 compact results exporter
===========================================

Reads ONLY the small Stage 3 summary CSVs and the large extreme_events CSVs.
It does NOT read raw trades.parquet, HTML charts, or any other large source data.

Place this file in the JERICHON project root and run:
    python JERICHON_Trade_Flow_Stage_3_Aggregate.py

Or specify the project root:
    python JERICHON_Trade_Flow_Stage_3_Aggregate.py --project-root "D:\\JERICHON"

Output: exactly two CSV files in trade_flow_stage3_visual/:
    stage3_compact_regimes.csv
    stage3_compact_extremes.csv

The large extreme_events CSV is processed in chunks to keep RAM usage low.
Original files are never modified or deleted.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd


TICKERS = ("SMLT", "VTBR", "GAZP", "ROSN", "SBER")
OUTPUT_FOLDER = "research/tradeflow/mark_4/results/trade_flow_stage3_visual"
CHUNK_SIZE = 200_000

REGIME_FILE = "{ticker}_regime_summary.csv"
LEVELS_FILE = "{ticker}_daily_levels.csv"
EXTREMES_FILE = "{ticker}_extreme_events.csv"

REGIME_METRICS = [
    "mean_tfi_volume",
    "mean_tfi_count",
    "mean_trade_intensity_per_sec",
    "positive_flow_share",
    "mean_formation_move_bps",
    "mean_future_1_change_bps",
    "mean_future_3_changes_bps",
    "mean_future_5_changes_bps",
    "absorption_proxy_share",
]

EXTREME_METRICS = [
    "tfi_volume",
    "tfi_count",
    "formation_move_bps",
    "future_1_price_changes_bps",
    "future_3_price_changes_bps",
    "future_5_price_changes_bps",
]


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read Stage 3 CSVs, tolerating UTF-8 BOM."""
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    values = series.astype("string").str.strip().str.lower()
    return values.isin({"true", "1", "yes", "y", "да"})


def weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    values = pd.to_numeric(frame[value_col], errors="coerce")
    weights = pd.to_numeric(frame[weight_col], errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def build_regime_export(root: Path) -> pd.DataFrame:
    daily_frames = []
    overall_frames = []

    for ticker in TICKERS:
        ticker_dir = root / ticker
        regime_path = ticker_dir / REGIME_FILE.format(ticker=ticker)
        levels_path = ticker_dir / LEVELS_FILE.format(ticker=ticker)

        if not regime_path.exists():
            print(f"[WARN] Missing: {regime_path}")
            continue

        regimes = read_csv(regime_path)
        if regimes.empty:
            print(f"[WARN] Empty: {regime_path}")
            continue

        regimes["ticker"] = ticker
        regimes["trade_date"] = regimes["trade_date"].astype(str)
        regimes["observations"] = pd.to_numeric(
            regimes["observations"], errors="coerce"
        ).fillna(0).astype("int64")

        if levels_path.exists():
            levels = read_csv(levels_path)
            if not levels.empty and "trade_date" in levels.columns:
                levels["trade_date"] = levels["trade_date"].astype(str)
                regimes = regimes.merge(levels, on="trade_date", how="left",
                                        suffixes=("", "_levels"))

        session_totals = regimes.groupby("trade_date")["observations"].sum()
        regimes["session_observations"] = regimes["trade_date"].map(session_totals)
        regimes["session_share"] = np.where(
            regimes["session_observations"] > 0,
            regimes["observations"] / regimes["session_observations"],
            np.nan,
        )
        regimes["row_type"] = "TICKER_DAY_REGIME"
        daily_frames.append(regimes)

        # One compact ticker-level row per regime. Means are weighted by the
        # number of observations in each daily regime.
        rows = []
        for regime, part in regimes.groupby("regime", dropna=False):
            row = {
                "ticker": ticker,
                "trade_date": "ALL",
                "regime": regime,
                "row_type": "TICKER_TOTAL_REGIME",
                "observations": int(part["observations"].sum()),
                "session_observations": np.nan,
                "session_share": (
                    float(part["observations"].sum())
                    / float(regimes.groupby("trade_date")["observations"].sum().sum())
                    if regimes["observations"].sum() > 0 else np.nan
                ),
            }
            for metric in REGIME_METRICS:
                if metric in part.columns:
                    row[metric] = weighted_mean(part, metric, "observations")
            if "prior_10d_high" in regimes.columns:
                latest = regimes.sort_values("trade_date").iloc[-1]
                row["latest_prior_10d_high"] = latest.get("prior_10d_high", np.nan)
                row["latest_prior_10d_low"] = latest.get("prior_10d_low", np.nan)
            rows.append(row)
        overall_frames.extend(rows)

    if not daily_frames:
        raise FileNotFoundError(
            f"No {TICKER}_regime_summary.csv files found under {root}"
        )

    result = pd.concat(daily_frames, ignore_index=True, sort=False)
    if overall_frames:
        result = pd.concat(
            [result, pd.DataFrame(overall_frames)],
            ignore_index=True, sort=False
        )

    first = ["row_type", "ticker", "trade_date", "regime",
             "observations", "session_observations", "session_share"]
    cols = first + [c for c in result.columns if c not in first]
    return result[cols].sort_values(
        ["ticker", "row_type", "trade_date", "regime"],
        kind="stable"
    ).reset_index(drop=True)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_extreme_export(root: Path) -> pd.DataFrame:
    """
    Stream extreme_events CSVs and aggregate by ticker/day/candidate/regime/flow.
    These are candidate OBSERVATIONS, not deduplicated swing-high/low events.
    """
    keys = ["ticker", "trade_date", "candidate_type", "regime", "flow_direction"]
    partials = []

    for ticker in TICKERS:
        path = root / ticker / EXTREMES_FILE.format(ticker=ticker)
        if not path.exists():
            print(f"[WARN] Missing: {path}")
            continue

        print(f"[READ] {path.name} in chunks of {CHUNK_SIZE:,}")
        try:
            reader = pd.read_csv(
                path, encoding="utf-8-sig", chunksize=CHUNK_SIZE,
                low_memory=False
            )
            found_rows = 0
            for chunk in reader:
                if chunk.empty:
                    continue
                found_rows += len(chunk)
                chunk["ticker"] = ticker
                chunk["trade_date"] = chunk["trade_date"].astype(str)

                required = keys + ["price"]
                missing = [c for c in required if c not in chunk.columns]
                if missing:
                    raise ValueError(
                        f"{path.name}: missing required columns: {missing}"
                    )

                for col in EXTREME_METRICS:
                    if col in chunk.columns:
                        chunk[col] = _numeric(chunk[col])

                for col in ("absorption_proxy", "near_prior_high", "near_prior_low"):
                    if col in chunk.columns:
                        chunk[col] = to_bool(chunk[col]).astype("int8")
                    else:
                        chunk[col] = 0

                agg_spec = {
                    "n_observations": ("price", "count"),
                    "absorption_n": ("absorption_proxy", "sum"),
                    "near_prior_high_n": ("near_prior_high", "sum"),
                    "near_prior_low_n": ("near_prior_low", "sum"),
                }
                for col in EXTREME_METRICS:
                    if col in chunk.columns:
                        agg_spec[f"{col}__sum"] = (col, "sum")
                        agg_spec[f"{col}__n"] = (col, "count")
                        if col.startswith("future_"):
                            agg_spec[f"{col}__positive_n"] = (
                                col, lambda s: int((s > 0).sum())
                            )
                            agg_spec[f"{col}__negative_n"] = (
                                col, lambda s: int((s < 0).sum())
                            )

                part = chunk.groupby(keys, dropna=False).agg(**agg_spec).reset_index()
                partials.append(part)

            print(f"       scanned {found_rows:,} candidate observations")
        except pd.errors.EmptyDataError:
            print(f"[WARN] Empty CSV: {path}")
        except Exception as exc:
            raise RuntimeError(f"Failed processing {path}: {exc}") from exc

    if not partials:
        return pd.DataFrame(columns=keys + ["n_observations"])

    # Only grouped partials are concatenated; raw extreme rows are discarded.
    compact = pd.concat(partials, ignore_index=True)
    sum_cols = [c for c in compact.columns if c not in keys]
    compact = compact.groupby(keys, dropna=False)[sum_cols].sum().reset_index()

    compact["absorption_share"] = (
        compact["absorption_n"] / compact["n_observations"].replace(0, np.nan)
    )
    compact["near_prior_high_share"] = (
        compact["near_prior_high_n"] / compact["n_observations"].replace(0, np.nan)
    )
    compact["near_prior_low_share"] = (
        compact["near_prior_low_n"] / compact["n_observations"].replace(0, np.nan)
    )

    for col in EXTREME_METRICS:
        sum_col, n_col = f"{col}__sum", f"{col}__n"
        if sum_col in compact.columns:
            compact[f"mean_{col}"] = (
                compact[sum_col] / compact[n_col].replace(0, np.nan)
            )
            if col.startswith("future_"):
                pos = compact.get(f"{col}__positive_n", 0)
                neg = compact.get(f"{col}__negative_n", 0)
                compact[f"positive_share_{col}"] = (
                    pos / compact[n_col].replace(0, np.nan)
                )
                compact[f"negative_share_{col}"] = (
                    neg / compact[n_col].replace(0, np.nan)
                )

    compact["row_type"] = "TICKER_DAY_CANDIDATE_GROUP"

    # Add ticker-wide totals for each candidate/regime/flow combination.
    ticker_keys = ["ticker", "candidate_type", "regime", "flow_direction"]
    total = compact.groupby(ticker_keys, dropna=False)[sum_cols].sum().reset_index()
    total["trade_date"] = "ALL"
    total["row_type"] = "TICKER_TOTAL_CANDIDATE_GROUP"

    for frame in (total,):
        frame["absorption_share"] = (
            frame["absorption_n"] / frame["n_observations"].replace(0, np.nan)
        )
        frame["near_prior_high_share"] = (
            frame["near_prior_high_n"] / frame["n_observations"].replace(0, np.nan)
        )
        frame["near_prior_low_share"] = (
            frame["near_prior_low_n"] / frame["n_observations"].replace(0, np.nan)
        )
        for col in EXTREME_METRICS:
            sum_col, n_col = f"{col}__sum", f"{col}__n"
            if sum_col in frame.columns:
                frame[f"mean_{col}"] = (
                    frame[sum_col] / frame[n_col].replace(0, np.nan)
                )
                if col.startswith("future_"):
                    frame[f"positive_share_{col}"] = (
                        frame[f"{col}__positive_n"] / frame[n_col].replace(0, np.nan)
                    )
                    frame[f"negative_share_{col}"] = (
                        frame[f"{col}__negative_n"] / frame[n_col].replace(0, np.nan)
                    )

    out = pd.concat([compact, total], ignore_index=True, sort=False)
    # Keep audit counts/sums as well as human-readable means and rates.
    return out.sort_values(
        ["ticker", "row_type", "trade_date", "candidate_type", "regime",
         "flow_direction"], kind="stable"
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Create two small CSV summaries from Stage 3 outputs."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2],
        help="JERICHON project root (folder containing research/tradeflow/mark_4/results/trade_flow_stage3_visual)"
    )
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve() / OUTPUT_FOLDER
    if not root.exists():
        raise FileNotFoundError(
            f"Stage 3 output folder not found: {root}\n"
            "Pass --project-root with the JERICHON project folder."
        )

    print("=" * 72)
    print("JERICHON Stage 3 — compact aggregation")
    print(f"Source: {root}")
    print("Raw trades / Parquet / HTML will NOT be opened.")
    print("=" * 72)

    regimes = build_regime_export(root)
    extremes = build_extreme_export(root)

    regime_out = root / "stage3_compact_regimes.csv"
    extreme_out = root / "stage3_compact_extremes.csv"

    regimes.to_csv(regime_out, index=False, encoding="utf-8-sig")
    extremes.to_csv(extreme_out, index=False, encoding="utf-8-sig")

    print("\nDONE")
    for path in (regime_out, extreme_out):
        size = path.stat().st_size
        print(f"{path.name}: {size:,} bytes ({size / 1024:.1f} KB)")
    print("\nInterpretation notes:")
    print("- TFI sign is the encoded BUYSELL/sign direction, not verified aggressor side.")
    print("- Price outcomes are transaction-price returns, not mid-price returns.")
    print("- Absorption is the Stage 3 proxy, not confirmed order-book absorption.")
    print("- Extreme rows count candidate observations, not unique pivot events.")
    print("- Means are aggregated from Stage 3 summaries; raw trade data is not reprocessed.")


if __name__ == "__main__":
    main()
