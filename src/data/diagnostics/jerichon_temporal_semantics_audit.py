# -*- coding: utf-8 -*-
"""JERICHON — temporal semantics and availability-aware alignment audit.

Does NOT rebuild the 475-feature matrix and does NOT modify production files.
It compares the current ts-based asof selection with selection by SYSTIME-derived
availability timestamp for Super Candles, and separately audits orderbook UPDATETIME.

Run from JERICHON\\data:
    python for_ml\\models\\jerichon_temporal_semantics_audit.py --ticker VTBR

Outputs under:
    for_ml/storage/diagnostics/temporal_semantics/<TICKER>/
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = PROJECT_ROOT / "data"
TZ = "Europe/Moscow"
SOURCES = {
    "trades": ("realtime", "trades"),
    "candles": ("realtime", "candles"),
    "orderbook": ("realtime", "orderbook"),
    "tradestats": ("supercandles", "tradestats"),
    "obstats": ("supercandles", "obstats"),
    "orderstats": ("supercandles", "orderstats"),
}
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def available_dates(root: Path, ticker: str) -> list[str]:
    dates = set()
    for kind in ("realtime", "supercandles"):
        base = root / "for_ml" / "storage" / "history" / kind / ticker
        if base.exists():
            dates.update(p.name for p in base.iterdir() if p.is_dir())
    return sorted(dates)


def load_history(root: Path, ticker: str) -> dict[str, pd.DataFrame]:
    frames = {k: [] for k in SOURCES}
    for date in available_dates(root, ticker):
        for name, (kind, filename) in SOURCES.items():
            path = root / "for_ml" / "storage" / "history" / kind / ticker / date / f"{filename}.parquet"
            if not path.exists():
                continue
            try:
                df = pd.read_parquet(path)
                if not df.empty:
                    df = df.copy()
                    df["_source_file_date"] = date
                    frames[name].append(df)
            except Exception as exc:
                logging.warning("Cannot read %s: %s", path, exc)
    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in frames.items()}


def parse_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def systime_to_utc(s: pd.Series) -> pd.Series:
    """AlgoPack docs examples use exchange-local wall time; interpret naive SYSTIME as MSK.
    If values are already timezone-aware, preserve their represented instant.
    """
    parsed = pd.to_datetime(s, errors="coerce", format="mixed")
    if getattr(parsed.dt, "tz", None) is not None:
        return parsed.dt.tz_convert("UTC")
    return parsed.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


def build_bars(root: Path, trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        raise RuntimeError("trades.parquet missing or empty; cannot build dollar bars")
    required = ["PRICE", "QUANTITY", "VALUE", "ts"]
    missing = [c for c in required if c not in trades.columns]
    if missing:
        raise RuntimeError(f"trades missing required columns: {missing}")
    project_root = root.parent if root.name.lower() == "data" else root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from src.data.builders.dollar_bars import InformationDrivenBars
    except Exception as exc:
        raise RuntimeError("Cannot import InformationDrivenBars from src.data.builders.dollar_bars") from exc
    x = trades.copy()
    x["ts"] = parse_utc(x["ts"])
    for c in ("PRICE", "QUANTITY", "VALUE"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=required)
    x = x[(x.PRICE > 0) & (x.QUANTITY > 0) & (x.VALUE > 0)]
    inp = pd.DataFrame({"ts": x["ts"], "price": x["PRICE"], "quantity": x["VALUE"] / x["PRICE"]}).dropna()
    bars = InformationDrivenBars().build_dollar(inp, threshold=13_000_000)
    if bars is None or bars.empty:
        raise RuntimeError("Dollar bar builder returned no bars")
    bars = bars.copy()
    bars["bar_ts"] = parse_utc(bars["close_time"])
    return bars.dropna(subset=["bar_ts"])[["bar_ts"]].drop_duplicates().sort_values("bar_ts").reset_index(drop=True)


def timestamp_summary(name: str, df: pd.DataFrame) -> dict:
    out = {"source": name, "rows": len(df), "has_ts": "ts" in df.columns,
           "has_SYSTIME": "SYSTIME" in df.columns, "has_UPDATETIME": "UPDATETIME" in df.columns}
    if df.empty or "ts" not in df.columns:
        return out
    ts = parse_utc(df["ts"]).dropna().sort_values()
    delta = ts.drop_duplicates().diff().dt.total_seconds().dropna()
    out.update({
        "valid_ts_rows": int(parse_utc(df["ts"]).notna().sum()),
        "unique_ts": int(ts.nunique()),
        "ts_min_utc": str(ts.min()) if len(ts) else "",
        "ts_max_utc": str(ts.max()) if len(ts) else "",
        "median_unique_ts_step_sec": float(delta.median()) if len(delta) else np.nan,
        "p95_unique_ts_step_sec": float(delta.quantile(.95)) if len(delta) else np.nan,
        "share_ts_on_5min_boundary_pct": float((((ts.dt.minute % 5 == 0) & (ts.dt.second == 0)).mean()) * 100) if len(ts) else np.nan,
    })
    if "SYSTIME" in df.columns:
        av = systime_to_utc(df["SYSTIME"])
        # Row-aligned Series; calculate the lag against the corresponding source ts.
        event = parse_utc(df["ts"])
        lag = (av - event).dt.total_seconds().replace([np.inf, -np.inf], np.nan).dropna()
        out.update({
            "systime_parse_pct": float(av.notna().mean() * 100),
            "median_systime_minus_ts_sec_msk": float(lag.median()) if len(lag) else np.nan,
            "p95_systime_minus_ts_sec_msk": float(lag.quantile(.95)) if len(lag) else np.nan,
            "max_systime_minus_ts_sec_msk": float(lag.max()) if len(lag) else np.nan,
            "negative_systime_lag_rows": int((lag < 0).sum()),
            "systime_after_ts_rows": int((lag > 0).sum()),
        })
    if "super_kind" in df.columns:
        out["super_kind_values"] = "; ".join(df["super_kind"].dropna().astype(str).value_counts().head(10).index.tolist())
    return out


def availability_alignment(name: str, df: pd.DataFrame, bars: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    base = {"source": name, "bars": len(bars), "has_ts": "ts" in df.columns, "has_SYSTIME": "SYSTIME" in df.columns}
    if df.empty or "ts" not in df.columns or "SYSTIME" not in df.columns:
        base.update({"matched_by_ts": 0, "matched_by_availability": 0, "current_ts_selection_late": np.nan,
                     "selection_diff_pct": np.nan, "median_source_age_sec_availability_join": np.nan})
        return base, pd.DataFrame()

    x = df.copy()
    x["source_ts"] = parse_utc(x["ts"])
    x["available_ts"] = systime_to_utc(x["SYSTIME"])
    x = x.dropna(subset=["source_ts", "available_ts"]).sort_values("source_ts")
    if x.empty:
        base.update({"matched_by_ts": 0, "matched_by_availability": 0, "current_ts_selection_late": np.nan,
                     "selection_diff_pct": np.nan, "median_source_age_sec_availability_join": np.nan})
        return base, pd.DataFrame()

    # Match the current pipeline's duplicate policy: last row for a duplicate timestamp.
    by_ts = x.sort_values(["source_ts", "available_ts"]).drop_duplicates("source_ts", keep="last")
    by_av = x.sort_values(["available_ts", "source_ts"]).drop_duplicates("available_ts", keep="last")

    left = bars[["bar_ts"]].sort_values("bar_ts").copy()
    cur = pd.merge_asof(left, by_ts[["source_ts", "available_ts"]].sort_values("source_ts"),
                        left_on="bar_ts", right_on="source_ts", direction="backward")
    cur = cur.rename(columns={"source_ts": "ts_selected_source_ts", "available_ts": "ts_selected_available_ts"})
    av = pd.merge_asof(left, by_av[["source_ts", "available_ts"]].sort_values("available_ts"),
                       left_on="bar_ts", right_on="available_ts", direction="backward")
    av = av.rename(columns={"source_ts": "av_selected_source_ts", "available_ts": "av_selected_available_ts"})
    joined = cur.merge(av, on="bar_ts", how="left")
    joined["current_ts_selection_late"] = joined["ts_selected_available_ts"] > joined["bar_ts"]
    joined["selection_changed"] = (
        joined["ts_selected_source_ts"].isna() != joined["av_selected_source_ts"].isna()
    ) | (joined["ts_selected_source_ts"] != joined["av_selected_source_ts"])
    joined["source_age_sec_availability_join"] = (joined["bar_ts"] - joined["av_selected_source_ts"]).dt.total_seconds()
    joined["source_age_sec_ts_join"] = (joined["bar_ts"] - joined["ts_selected_source_ts"]).dt.total_seconds()
    late = joined["current_ts_selection_late"].fillna(False)
    changed = joined["selection_changed"].fillna(True)
    age = joined["source_age_sec_availability_join"].dropna()
    base.update({
        "matched_by_ts": int(joined["ts_selected_source_ts"].notna().sum()),
        "matched_by_availability": int(joined["av_selected_source_ts"].notna().sum()),
        "current_ts_selection_late": int(late.sum()),
        "current_ts_selection_late_pct": float(late.mean() * 100),
        "selection_changed_count": int(changed.sum()),
        "selection_diff_pct": float(changed.mean() * 100),
        "median_source_age_sec_availability_join": float(age.median()) if len(age) else np.nan,
        "p95_source_age_sec_availability_join": float(age.quantile(.95)) if len(age) else np.nan,
    })
    # Only save a compact set of rows where current selection is late or changes.
    examples = joined.loc[late | changed].copy()
    examples.insert(1, "source", name)
    examples = examples.head(250)
    return base, examples


def audit_orderbook(df: pd.DataFrame, bars: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    out = {"source": "orderbook", "rows": len(df), "has_UPDATETIME": "UPDATETIME" in df.columns,
           "has_SEQNUM": "SEQNUM" in df.columns}
    if df.empty or "UPDATETIME" not in df.columns or "_source_file_date" not in df.columns:
        out.update({"valid_update_rows": 0, "unique_snapshots": 0, "matched_bars": 0})
        return out, pd.DataFrame()
    x = df.copy()
    # UPDATETIME is documented as a time-of-day; combine it with the saved source-file date.
    wall = pd.to_datetime(x["_source_file_date"].astype(str) + " " + x["UPDATETIME"].astype(str), errors="coerce")
    x["update_ts"] = wall.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    x = x.dropna(subset=["update_ts"])
    key = ["_source_file_date", "UPDATETIME"]
    if "SEQNUM" in x.columns:
        key.append("SEQNUM")
    snaps = x.groupby(key, dropna=False, as_index=False).agg(update_ts=("update_ts", "max"), levels=("PRICE", "size") if "PRICE" in x.columns else ("update_ts", "size"))
    snaps = snaps.sort_values("update_ts")
    matched = pd.merge_asof(bars.sort_values("bar_ts"), snaps[["update_ts"]].sort_values("update_ts"),
                            left_on="bar_ts", right_on="update_ts", direction="backward")
    age = (matched["bar_ts"] - matched["update_ts"]).dt.total_seconds().dropna()
    out.update({"valid_update_rows": len(x), "unique_snapshots": len(snaps),
                "matched_bars": int(matched["update_ts"].notna().sum()),
                "median_snapshot_age_sec": float(age.median()) if len(age) else np.nan,
                "p95_snapshot_age_sec": float(age.quantile(.95)) if len(age) else np.nan,
                "max_snapshot_age_sec": float(age.max()) if len(age) else np.nan})
    return out, matched


def main():
    parser = argparse.ArgumentParser(description="Audit timestamp semantics and availability-aware joins.")
    parser.add_argument("--ticker", default="VTBR")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    logging.info("JERICHON temporal semantics audit | ticker=%s | root=%s", args.ticker, root)
    data = load_history(root, args.ticker)
    bars = build_bars(root, data.get("trades", pd.DataFrame()))
    out_dir = root / "for_ml" / "storage" / "diagnostics" / "temporal_semantics" / args.ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_summary = pd.DataFrame([timestamp_summary(k, v) for k, v in data.items()])
    ts_summary.to_csv(out_dir / "timestamp_semantics_summary.csv", index=False, encoding="utf-8-sig")

    alignment_rows, examples = [], []
    for name in ("tradestats", "obstats", "orderstats"):
        result, sample = availability_alignment(name, data.get(name, pd.DataFrame()), bars)
        alignment_rows.append(result)
        if not sample.empty:
            examples.append(sample)
    alignment = pd.DataFrame(alignment_rows)
    alignment.to_csv(out_dir / "availability_alignment_summary.csv", index=False, encoding="utf-8-sig")
    if examples:
        pd.concat(examples, ignore_index=True).to_csv(out_dir / "availability_alignment_examples.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(out_dir / "availability_alignment_examples.csv", index=False, encoding="utf-8-sig")

    ob_summary, ob_details = audit_orderbook(data.get("orderbook", pd.DataFrame()), bars)
    pd.DataFrame([ob_summary]).to_csv(out_dir / "orderbook_updatetime_summary.csv", index=False, encoding="utf-8-sig")
    if not ob_details.empty:
        ob_details.to_csv(out_dir / "orderbook_bar_matches.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(f"JERICHON TEMPORAL SEMANTICS AUDIT | {args.ticker} | bars={len(bars):,}")
    print(f"Period: {bars.bar_ts.min()} — {bars.bar_ts.max()}")
    print("\n1) Timestamp semantics / observed delays (naive SYSTIME interpreted as Moscow time):")
    cols = [c for c in ["source", "rows", "unique_ts", "median_unique_ts_step_sec", "share_ts_on_5min_boundary_pct",
                        "median_systime_minus_ts_sec_msk", "p95_systime_minus_ts_sec_msk", "max_systime_minus_ts_sec_msk"] if c in ts_summary]
    print(ts_summary[cols].to_string(index=False))
    print("\n2) Current ts-join vs availability-time join:")
    print(alignment.to_string(index=False))
    print("\n3) Orderbook UPDATETIME:")
    print(pd.DataFrame([ob_summary]).to_string(index=False))
    print("\nInterpretation:")
    print("- A row counted as current_ts_selection_late has SYSTIME after bar close, despite ts <= bar close.")
    print("- selection_diff_pct estimates how often an availability-time join would select a different source record.")
    print("- This is a source-level alignment audit, not proof that every feature in X475 leaks.")
    print("- UPDATETIME-derived dates assume the file date is the correct trading date; inspect overnight/session edge cases.")
    print(f"\nSaved compact reports to: {out_dir}")

if __name__ == "__main__":
    main()
