# -*- coding: utf-8 -*-
"""Compact Trade Flow research exporter for JERICHON.

Place in project root beside:
  trade_flow_outputs_SMLT/
  trade_flow_outputs_VTBR/
  trade_flow_outputs_GAZP/

Run: python export_trade_flow_report.py

Creates exactly three files in trade_flow_report/:
  research_summary.md, key_findings.csv, distributions.csv

Requires: pandas, numpy, pyarrow.
"""
from pathlib import Path
import math
import warnings
import numpy as np
import pandas as pd

TICKERS = ("SMLT", "VTBR", "GAZP", "ROSN", "SBER")
CSV_NAMES = {
    "bucket": "{t}_bucket_results.csv",
    "bootstrap": "{t}_day_block_bootstrap.csv",
    "momentum": "{t}_momentum_control.csv",
    "overlap": "{t}_overlap_report.csv",
}
PQ_NAMES = {
    "features": "{t}_trade_flow_features.parquet",
    "targets": "{t}_trade_price_targets.parquet",
    "uniqueness": "{t}_event_uniqueness.parquet",
    "trades": "{t}_trades_normalized.parquet",
}
BATCH = 50_000
SAMPLE_MAX = 100_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "research" / "tradeflow" / "mark_2" / "results" / "trade_flow_report"


def read_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"{path.name}: {exc}")
        return pd.DataFrame()


def parquet_info(path):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    return pf.metadata.num_rows, pf.metadata.num_row_groups, pf.schema_arrow.names


def profile_parquet(path, ticker, kind):
    """Full-pass moments/extrema/nulls; sampled quantiles, bounded memory."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        warnings.warn("Нет pyarrow. Установи: pip install pyarrow")
        return []
    try:
        pf = pq.ParquetFile(path)
        cols = [
            f.name for f in pf.schema_arrow
            if any(x in str(f.type).lower() for x in ("int", "float", "double", "decimal"))
            and f.name not in ("trade_id",)
        ]
        stats = {
            c: {"n": 0, "null": 0, "sum": 0.0, "sum2": 0.0,
                "min": np.inf, "max": -np.inf, "sample": np.empty(0), "seen": 0}
            for c in cols
        }
        rng = np.random.default_rng(20261106)
        for batch in pf.iter_batches(batch_size=BATCH, columns=cols):
            df = batch.to_pandas()
            for c in cols:
                a = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                a = a[np.isfinite(a)]
                s = stats[c]
                # Null count includes NaN and non-finite values.
                s["null"] += len(df) - len(a)
                if not len(a):
                    continue
                s["n"] += len(a)
                s["sum"] += float(a.sum(dtype=np.float64))
                s["sum2"] += float(np.dot(a, a))
                s["min"] = min(s["min"], float(a.min()))
                s["max"] = max(s["max"], float(a.max()))
                # Reservoir sampling (uniform, deterministic).
                sample = s["sample"]
                for v in a:
                    s["seen"] += 1
                    if len(sample) < SAMPLE_MAX:
                        sample = np.append(sample, v)
                    else:
                        j = rng.integers(0, s["seen"])
                        if j < SAMPLE_MAX:
                            sample[j] = v
                s["sample"] = sample
        rows = []
        for c, s in stats.items():
            n = s["n"]
            if not n:
                continue
            mean = s["sum"] / n
            std = math.sqrt(max(0.0, (s["sum2"] - n * mean * mean) / max(1, n - 1)))
            q = np.quantile(s["sample"], [0, .01, .05, .25, .5, .75, .95, .99, 1])
            rows.append({
                "ticker": ticker, "dataset": kind, "column": c,
                "valid_n": n, "null_n": s["null"],
                "null_share": s["null"] / max(1, n + s["null"]),
                "mean": mean, "std": std, "min_full": s["min"], "max_full": s["max"],
                "q00_sample": q[0], "q01_sample": q[1], "q05_sample": q[2],
                "q25_sample": q[3], "q50_sample": q[4], "q75_sample": q[5],
                "q95_sample": q[6], "q99_sample": q[7], "q100_sample": q[8],
                "quantile_sample_n": len(s["sample"]),
            })
        return rows
    except Exception as exc:
        warnings.warn(f"Не удалось профилировать {path.name}: {exc}")
        return []


def col(df, name):
    return name if name in df.columns else None


def findings_for(ticker, bucket, boot):
    out = []
    if not bucket.empty and {"signal", "bucket"}.issubset(bucket.columns):
        mean_cols = [c for c in bucket.columns if c.lower().endswith("_mean")]
        for target_col in mean_cols:
            for signal, g in bucket.groupby("signal"):
                g = g.sort_values("bucket")
                if len(g) < 2:
                    continue
                lo, hi = g.iloc[0], g.iloc[-1]
                lv, hv = pd.to_numeric(lo[target_col], errors="coerce"), pd.to_numeric(hi[target_col], errors="coerce")
                if pd.isna(lv) or pd.isna(hv):
                    continue
                out.append({
                    "ticker": ticker, "method": "bucket_high_minus_low",
                    "signal": signal, "target": target_col[:-5],
                    "effect": float(hv - lv),
                    "low_bucket": lo["bucket"], "low_bucket_mean": float(lv),
                    "high_bucket": hi["bucket"], "high_bucket_mean": float(hv),
                    "low_n": lo.get("observations", np.nan), "high_n": hi.get("observations", np.nan),
                    "ci_low": np.nan, "ci_high": np.nan,
                    "ci_excludes_zero": np.nan,
                })
    if not boot.empty and {"signal", "target", "spread_bps"}.issubset(boot.columns):
        for _, r in boot.iterrows():
            e = pd.to_numeric(r["spread_bps"], errors="coerce")
            if pd.isna(e):
                continue
            l = pd.to_numeric(r.get("ci_low", np.nan), errors="coerce")
            h = pd.to_numeric(r.get("ci_high", np.nan), errors="coerce")
            out.append({
                "ticker": ticker, "method": "day_block_bootstrap",
                "signal": r["signal"], "target": r["target"], "effect": float(e),
                "low_bucket": np.nan, "low_bucket_mean": np.nan,
                "high_bucket": np.nan, "high_bucket_mean": np.nan,
                "low_n": np.nan, "high_n": np.nan,
                "ci_low": l, "ci_high": h,
                "ci_excludes_zero": bool(l > 0 or h < 0) if pd.notna(l) and pd.notna(h) else np.nan,
            })
    return out


def main():
    root = PROJECT_ROOT
    OUT.mkdir(exist_ok=True)
    all_findings, all_dist = [], []
    report = [
        "# JERICHON — Trade Flow: compact research export", "",
        f"Сформировано: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}", "",
        "Квантили parquet рассчитаны по детерминированной равномерной выборке до 100 000 значений на колонку.",
        "Count/null/min/max/mean/std рассчитаны по полному файлу. CSV-результаты читаются целиком.", "",
    ]

    for ticker in TICKERS:
        folder = root / f"trade_flow_outputs_{ticker}"
        report += [f"## {ticker}", ""]
        if not folder.is_dir():
            report += [f"Папка не найдена: `{folder.name}`", ""]
            continue

        for kind, pattern in PQ_NAMES.items():
            p = folder / pattern.format(t=ticker)
            if not p.exists():
                continue
            size_mb = p.stat().st_size / 1024**2
            try:
                nrows, ngroups, names = parquet_info(p)
                report.append(f"- `{p.name}` — {nrows:,} строк, {len(names)} колонок, {size_mb:.1f} MB")
            except Exception as exc:
                report.append(f"- `{p.name}` — {size_mb:.1f} MB; metadata error: {exc}")
                continue
            if kind in ("features", "targets", "uniqueness"):
                all_dist.extend(profile_parquet(p, ticker, kind))

        bucket = read_csv(folder / CSV_NAMES["bucket"].format(t=ticker))
        boot = read_csv(folder / CSV_NAMES["bootstrap"].format(t=ticker))
        momentum = read_csv(folder / CSV_NAMES["momentum"].format(t=ticker))
        overlap = read_csv(folder / CSV_NAMES["overlap"].format(t=ticker))

        if not overlap.empty:
            report += ["", "### Event overlap / uniqueness", ""]
            keep = [c for c in ["event_type", "valid_events", "mean_uniqueness", "median_uniqueness", "max_concurrency"] if c in overlap]
            report.append(overlap[keep].to_string(index=False) if keep else overlap.head(10).to_string(index=False))

        if not bucket.empty:
            report += ["", f"### Bucket results ({len(bucket):,} rows)", ""]
            if {"signal", "bucket"}.issubset(bucket.columns):
                means = [c for c in bucket.columns if c.lower().endswith("_mean")]
                edges = []
                for _, g in bucket.groupby("signal"):
                    g = g.sort_values("bucket")
                    if len(g):
                        edges.extend([g.iloc[0], g.iloc[-1]])
                if edges:
                    edge_df = pd.DataFrame(edges)
                    keep = [c for c in ["signal", "bucket", "observations"] if c in edge_df] + means[:6]
                    report.append(edge_df[keep].to_string(index=False))
            else:
                report.append(bucket.head(12).to_string(index=False))

        if not boot.empty:
            report += ["", f"### Day-block bootstrap ({len(boot):,} tests)", ""]
            if "spread_bps" in boot:
                b = boot.copy()
                b["_abs"] = pd.to_numeric(b["spread_bps"], errors="coerce").abs()
                keep = [c for c in ["signal", "target", "spread_bps", "ci_low", "ci_high", "days", "bootstrap_valid"] if c in b]
                report.append(b.sort_values("_abs", ascending=False)[keep].head(15).to_string(index=False))
            else:
                report.append(boot.head(10).to_string(index=False))

        if not momentum.empty:
            report += ["", f"### Momentum control ({len(momentum):,} rows)", ""]
            report.append(momentum.head(10).to_string(index=False))

        all_findings.extend(findings_for(ticker, bucket, boot))
        report.append("")

    fdf = pd.DataFrame(all_findings)
    if not fdf.empty:
        fdf["_abs"] = pd.to_numeric(fdf["effect"], errors="coerce").abs()
        fdf = fdf.sort_values(["ticker", "method", "_abs"], ascending=[True, True, False]).drop(columns="_abs")
    ddf = pd.DataFrame(all_dist)

    report += [
        "## Interpretation notes", "",
        "- `bucket_high_minus_low`: среднее значение target в верхнем bucket минус нижний; это описательный эффект, не тест значимости.",
        "- `day_block_bootstrap`: разница верхнего/нижнего квинтилей и доверительный интервал bootstrap по торговым дням.",
        "- CI, не пересекающий 0, — предварительный статистический сигнал, но при малом числе дней он нестабилен.",
        "- `distributions.csv`: моменты и экстремумы рассчитаны по всем строкам; квантили — по выборке до 100 тыс. наблюдений на колонку.",
        "- `trades_normalized.parquet` учитывается по размеру/числу строк, но не профилируется, чтобы не раздувать время и отчёт.",
        "- Эти данные сами по себе не доказывают HFT-альфу. Нужно подтвердить смысл BUYSELL, порядок событий, bid/ask bounce и реалистичность исполнения/издержек.", "",
        "## Состав экспорта", "",
        "1. `research_summary.md` — обзор по тикерам и наиболее заметные результаты.",
        "2. `key_findings.csv` — все bucket spreads и day-block bootstrap оценки.",
        "3. `distributions.csv` — статистики/квантили признаков, target и uniqueness.", "",
    ]

    (OUT / "research_summary.md").write_text("\n".join(report), encoding="utf-8")
    fdf.to_csv(OUT / "key_findings.csv", index=False, encoding="utf-8-sig")
    ddf.to_csv(OUT / "distributions.csv", index=False, encoding="utf-8-sig")

    print("Экспорт завершён. Созданы три файла:")
    for name in ("research_summary.md", "key_findings.csv", "distributions.csv"):
        p = OUT / name
        print(f"  {p} — {p.stat().st_size / 1024:.1f} KB")
    if fdf.empty:
        print("ВНИМАНИЕ: key_findings.csv пуст — проверь формат bucket/bootstrap CSV.")
    if ddf.empty:
        print("ВНИМАНИЕ: distributions.csv пуст — проверь parquet и установку pyarrow.")


if __name__ == "__main__":
    main()
