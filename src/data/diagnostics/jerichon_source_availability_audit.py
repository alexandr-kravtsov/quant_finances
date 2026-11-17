# -*- coding: utf-8 -*-
"""
JERICHON — SOURCE DATA AVAILABILITY AUDIT
========================================
Цель: проверить, были ли исходные записи AlgoPack доступны к закрытию
каждого dollar bar, не перестраивая 475 признаков.

Запуск из Windows CMD (из папки data или с полным путём):
    python jerichon_source_availability_audit.py --ticker VTBR

Результаты:
    for_ml/storage/diagnostics/source_availability/<TICKER>/
        summary_by_source.csv
        bar_source_audit.csv
        raw_timestamp_schema.csv

Важно:
- ts трактуется так же, как в текущем pipeline: UTC -> Europe/Moscow.
- SYSTIME анализируется в трёх вариантах: как есть, naive как MSK,
  naive как UTC. Это намеренно: без подтверждённого контракта нельзя
  молча выбирать одну трактовку.
- Если SYSTIME отсутствует/непарсится, доступность нельзя подтвердить.
- Это аудит доступности исходных записей, а не самостоятельное доказательство
  отсутствия любого look-ahead внутри всех feature builders.
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
                logging.warning("Не удалось прочитать %s: %s", path, exc)
    out = {}
    for name, parts in frames.items():
        out[name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out


def parse_event_ts(series: pd.Series) -> pd.Series:
    # Внутри аудита все timestamps нормализуются к UTC, чтобы избежать
    # несовпадения timezone dtype (UTC+03:00 vs Europe/Moscow).
    return pd.to_datetime(series, errors="coerce", utc=True)


def parse_systime_variants(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return raw parsed values plus MSK-assumed and UTC-assumed UTC timestamps.

    Every comparison-ready output is normalized to datetime64[ns, UTC].
    This deliberately avoids pandas' tz-naive vs tz-aware comparison error.
    """
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    tz = getattr(parsed.dt, "tz", None)

    if tz is not None:
        aware_utc = parsed.dt.tz_convert("UTC")
        # Already-aware timestamps have one unambiguous instant; both variants
        # therefore represent the same instant.
        return aware_utc, aware_utc, aware_utc

    # Naive SYSTIME has ambiguous timezone semantics. Keep two interpretations,
    # but make both comparison-ready UTC-aware series.
    as_msk_utc = (
        parsed.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )
    as_utc = parsed.dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")

    # Defensive normalization for pandas versions / mixed input edge cases.
    as_msk_utc = pd.to_datetime(as_msk_utc, errors="coerce", utc=True)
    as_utc = pd.to_datetime(as_utc, errors="coerce", utc=True)
    return parsed, as_msk_utc, as_utc


def load_dollar_bars(root: Path, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    trades = data.get("trades", pd.DataFrame()).copy()
    if trades.empty:
        raise RuntimeError("trades.parquet не найден или пуст: невозможно построить dollar bars.")
    required = ["PRICE", "QUANTITY", "VALUE", "ts"]
    missing = [c for c in required if c not in trades.columns]
    if missing:
        raise RuntimeError(f"В trades отсутствуют обязательные колонки: {missing}")

    project_root = root.parent if root.name.lower() == "data" else root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from src.data.builders.dollar_bars import InformationDrivenBars
    except Exception as exc:
        raise RuntimeError(
            "Не удалось импортировать InformationDrivenBars из for_ml.builders.dollar_bars. "
            "Запускай скрипт в текущем проекте JERICHON и проверь структуру папок."
        ) from exc

    trades["ts"] = parse_event_ts(trades["ts"])
    for col in ("PRICE", "QUANTITY", "VALUE"):
        trades[col] = pd.to_numeric(trades[col], errors="coerce")
    trades = trades.dropna(subset=required)
    trades = trades[(trades.PRICE > 0) & (trades.QUANTITY > 0) & (trades.VALUE > 0)]
    if trades.empty:
        raise RuntimeError("После очистки в trades не осталось валидных записей.")

    input_trades = pd.DataFrame({
        "ts": trades["ts"],
        "price": trades["PRICE"],
        "quantity": trades["VALUE"] / trades["PRICE"],
    }).dropna()
    bars = InformationDrivenBars().build_dollar(input_trades, threshold=13_000_000)
    if bars is None or bars.empty:
        raise RuntimeError("InformationDrivenBars не построил dollar bars.")
    bars = bars.copy()
    bars["bar_ts"] = pd.to_datetime(bars["close_time"], errors="coerce", utc=True)
    bars = bars.dropna(subset=["bar_ts"]).sort_values("bar_ts")
    return bars[["bar_ts"]].drop_duplicates().reset_index(drop=True)


def make_schema_report(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in data.items():
        if df.empty:
            rows.append({"source": name, "rows": 0, "columns": "", "has_ts": False,
                         "has_SYSTIME": False, "ts_dtype": "", "SYSTIME_dtype": "",
                         "SYSTIME_parse_rate": np.nan, "SYSTIME_naive": ""})
            continue
        has_ts = "ts" in df.columns
        has_st = "SYSTIME" in df.columns
        parsed_rate = np.nan
        naive = ""
        if has_st:
            raw = pd.to_datetime(df["SYSTIME"], errors="coerce")
            parsed_rate = float(raw.notna().mean())
            naive = "yes" if getattr(raw.dt, "tz", None) is None else "no"
        rows.append({
            "source": name, "rows": len(df), "columns": " | ".join(map(str, df.columns)),
            "has_ts": has_ts, "has_SYSTIME": has_st,
            "ts_dtype": str(df["ts"].dtype) if has_ts else "",
            "SYSTIME_dtype": str(df["SYSTIME"].dtype) if has_st else "",
            "SYSTIME_parse_rate": parsed_rate, "SYSTIME_naive": naive,
        })
    return pd.DataFrame(rows)


def audit_source(name: str, df: pd.DataFrame, bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df.empty or "ts" not in df.columns:
        detail = pd.DataFrame({
            "bar_ts": bars["bar_ts"], "source": name,
            "status": "SOURCE_EMPTY_OR_NO_TS",
        })
        return detail, {"source": name, "rows": len(df), "bars": len(bars),
                        "matched_event_ts": 0, "late_by_systime_msk": 0,
                        "late_by_systime_utc": 0, "missing_systime": len(bars),
                        "median_systime_minus_ts_sec": np.nan,
                        "p95_systime_minus_ts_sec": np.nan,
                        "max_systime_minus_ts_sec": np.nan}

    x = df.copy()
    x["_event_ts"] = parse_event_ts(x["ts"])
    x = x.dropna(subset=["_event_ts"]).sort_values("_event_ts").reset_index(drop=True)
    if x.empty:
        detail = pd.DataFrame({"bar_ts": bars["bar_ts"], "source": name, "status": "NO_VALID_TS"})
        return detail, {"source": name, "rows": len(df), "bars": len(bars),
                        "matched_event_ts": 0, "late_by_systime_msk": 0,
                        "late_by_systime_utc": 0, "missing_systime": len(bars),
                        "median_systime_minus_ts_sec": np.nan,
                        "p95_systime_minus_ts_sec": np.nan,
                        "max_systime_minus_ts_sec": np.nan}

    if "SYSTIME" in x.columns:
        raw, st_msk, st_utc = parse_systime_variants(x["SYSTIME"])
        x["_systime_msk"] = st_msk
        x["_systime_utc"] = st_utc
        x["_systime_parse_ok"] = st_msk.notna() & st_utc.notna()
        # SYSTIME minus event ts in seconds, under each naive-time interpretation.
        x["_lag_msk_sec"] = (x["_systime_msk"] - x["_event_ts"]).dt.total_seconds()
        x["_lag_utc_sec"] = (x["_systime_utc"] - x["_event_ts"]).dt.total_seconds()
    else:
        x["_systime_msk"] = pd.NaT
        x["_systime_utc"] = pd.NaT
        x["_systime_parse_ok"] = False
        x["_lag_msk_sec"] = np.nan
        x["_lag_utc_sec"] = np.nan

    # For each bar, select the exact record the current ts-based backward merge
    # would use. Then test whether its SYSTIME was already <= bar close.
    left = bars[["bar_ts"]].sort_values("bar_ts").copy()
    right = x[["_event_ts", "_systime_msk", "_systime_utc",
               "_systime_parse_ok", "_lag_msk_sec", "_lag_utc_sec"]].copy()
    right = right.rename(columns={"_event_ts": "matched_event_ts"})
    joined = pd.merge_asof(
        left, right.sort_values("matched_event_ts"),
        left_on="bar_ts", right_on="matched_event_ts", direction="backward"
    )
    joined["source"] = name
    joined["event_age_sec"] = (joined["bar_ts"] - joined["matched_event_ts"]).dt.total_seconds()
    joined["systime_missing_or_unparseable"] = ~joined["_systime_parse_ok"].fillna(False)
    # Boolean comparisons can be nullable when SYSTIME is missing.
    # Normalize both sides to UTC-aware datetime dtype before comparing.
    joined["bar_ts"] = pd.to_datetime(joined["bar_ts"], errors="coerce", utc=True)
    joined["_systime_msk"] = pd.to_datetime(joined["_systime_msk"], errors="coerce", utc=True)
    joined["_systime_utc"] = pd.to_datetime(joined["_systime_utc"], errors="coerce", utc=True)
    joined["late_by_systime_msk"] = (
        joined["_systime_msk"].notna() & (joined["_systime_msk"] > joined["bar_ts"])
    )
    joined["late_by_systime_utc"] = (
        joined["_systime_utc"].notna() & (joined["_systime_utc"] > joined["bar_ts"])
    )

    def status(row):
        if pd.isna(row["matched_event_ts"]):
            return "NO_PRIOR_SOURCE_RECORD"
        if row["systime_missing_or_unparseable"]:
            return "CANNOT_VERIFY_SYSTIME"
        if row["late_by_systime_msk"] and row["late_by_systime_utc"]:
            return "LATE_UNDER_BOTH_TIMEZONES"
        if row["late_by_systime_msk"]:
            return "LATE_IF_SYSTIME_IS_MSK"
        if row["late_by_systime_utc"]:
            return "LATE_IF_SYSTIME_IS_UTC"
        return "SYSTIME_NOT_LATER_THAN_BAR"
    joined["status"] = joined.apply(status, axis=1)

    lags = x["_lag_msk_sec"].replace([np.inf, -np.inf], np.nan).dropna()
    summary = {
        "source": name,
        "rows": len(df),
        "bars": len(bars),
        "matched_event_ts": int(joined["matched_event_ts"].notna().sum()),
        "late_by_systime_msk": int(joined["late_by_systime_msk"].sum()),
        "late_by_systime_utc": int(joined["late_by_systime_utc"].sum()),
        "missing_systime": int(joined["systime_missing_or_unparseable"].sum()),
        "median_systime_minus_ts_sec": float(lags.median()) if len(lags) else np.nan,
        "p95_systime_minus_ts_sec": float(lags.quantile(.95)) if len(lags) else np.nan,
        "max_systime_minus_ts_sec": float(lags.max()) if len(lags) else np.nan,
    }
    keep = ["bar_ts", "source", "matched_event_ts", "event_age_sec",
            "_systime_msk", "_systime_utc", "_lag_msk_sec", "_lag_utc_sec",
            "late_by_systime_msk", "late_by_systime_utc",
            "systime_missing_or_unparseable", "status"]
    return joined[keep], summary


def main():
    parser = argparse.ArgumentParser(description="Audit AlgoPack source-data availability at dollar-bar close.")
    parser.add_argument("--ticker", default="VTBR", help="Ticker, e.g. VTBR or SMLT")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="JERICHON data root")
    args = parser.parse_args()
    root = Path(args.root)

    logging.info("JERICHON source availability audit | ticker=%s", args.ticker)
    logging.info("Root: %s", root)
    data = load_history(root, args.ticker)
    schema = make_schema_report(data)
    bars = load_dollar_bars(root, data)

    out_dir = root / "for_ml" / "storage" / "diagnostics" / "source_availability" / args.ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    schema.to_csv(out_dir / "raw_timestamp_schema.csv", index=False, encoding="utf-8-sig")

    details, summaries = [], []
    for name, df in data.items():
        detail, summary = audit_source(name, df, bars)
        details.append(detail)
        summaries.append(summary)
    detail_df = pd.concat(details, ignore_index=True, sort=False)
    summary_df = pd.DataFrame(summaries)
    detail_df.to_csv(out_dir / "bar_source_audit.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "summary_by_source.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 88)
    print("SOURCE DATA AVAILABILITY AUDIT — RESULT")
    print("=" * 88)
    print(f"Ticker: {args.ticker} | dollar bars: {len(bars):,}")
    print(f"Период: {bars['bar_ts'].min()} — {bars['bar_ts'].max()}")
    print("\nСводка по источникам:")
    cols = ["source", "rows", "matched_event_ts", "late_by_systime_msk",
            "late_by_systime_utc", "missing_systime",
            "median_systime_minus_ts_sec", "p95_systime_minus_ts_sec",
            "max_systime_minus_ts_sec"]
    print(summary_df[cols].to_string(index=False))
    print("\nИнтерпретация:")
    print("- LATE_* > 0: ts-based merge выбрал бы запись, чья SYSTIME позже закрытия бара.")
    print("- CANNOT_VERIFY_SYSTIME: доступность нельзя подтвердить по имеющимся меткам.")
    print("- Нулевые LATE_* не доказывают отсутствие leakage, если семантика SYSTIME не подтверждена.")
    print(f"\nОтчёты сохранены: {out_dir}")


if __name__ == "__main__":
    main()
