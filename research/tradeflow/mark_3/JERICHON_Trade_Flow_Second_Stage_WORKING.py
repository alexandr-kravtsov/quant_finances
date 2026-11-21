# -*- coding: utf-8 -*-
"""
JERICHON — Trade Flow second-stage research (no ML).

Place anywhere inside the JERICHON project. The script searches for the
Stage-1 feature parquet recursively in the project directory and its parent.
Preferred layout:
  trade_flow_outputs_SMLT/SMLT_trade_flow_features.parquet
  trade_flow_outputs_VTBR/VTBR_trade_flow_features.parquet
  trade_flow_outputs_GAZP/GAZP_trade_flow_features.parquet

Run:
  python JERICHON_Trade_Flow_Second_Stage.py

Requires:
  pip install pandas numpy pyarrow

Two focused studies:
1) Data-integrity audit + rebuild TFI from primitive sign/size/notional.
2) Intraday, momentum-conditioned price response: does trade flow add information
   beyond recent transaction-price movement? Compares volume/notional/count TFI.

Reads only compact columns from the existing Parquet files. Does not modify inputs.
Writes three compact CSVs and one short Markdown report into
research/tradeflow/mark_3/results/.

Important:
- The stored signed_size / tfi_volume fields are audited, not trusted.
- All rolling windows and forward targets reset at trade_date boundaries.
- Targets remain transaction-price responses, NOT mid-price responses.
- BUYSELL/sign aggressor semantics are not assumed or certified by this script.
"""
from pathlib import Path
import os
import warnings
import numpy as np
import pandas as pd

TICKERS = ("SMLT", "VTBR", "GAZP", "SBER", "ROSN")
WINDOWS = (25, 50, 100, 1000)
TARGETS = (
    "ret_next_5_trades_bps",
    "ret_next_10_trades_bps",
    "ret_after_1_distinct_price_changes_bps",
    "ret_after_3_distinct_price_changes_bps",
    "ret_after_5_distinct_price_changes_bps",
)
BOOT_REPS = 300
SEED = 20261106
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "tradeflow" / "mark_3" / "results" / "trade_flow_second_stage"
OUT.mkdir(exist_ok=True)


def locate_feature_file(ticker):
    """
    Locate the Stage-1 feature parquet even if its output folder is nested
    or the script was copied to a different project subfolder.
    """
    filename = f"{ticker}_trade_flow_features.parquet"
    foldername = f"trade_flow_outputs_{ticker}"

    # Fast, common layouts first.
    roots = [ROOT]
    candidates = []
    for base in roots:
        candidates.extend([
            base / foldername / filename,
            base / "trade_flow_outputs" / ticker / filename,
            base / "data" / foldername / filename,
            base / "data" / "trade_flow_outputs" / ticker / filename,
        ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Search beneath the project directory, pruning common bulky/unrelated dirs.
    skip_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules",
                 ".mypy_cache", ".pytest_cache"}
    for search_root in roots:
        if not search_root.is_dir():
            continue
        for current, dirs, files in os.walk(search_root):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            if filename in files:
                return Path(current) / filename

    # Case-insensitive fallback for minor filename casing differences.
    target = filename.lower()
    for search_root in roots:
        if not search_root.is_dir():
            continue
        for current, dirs, files in os.walk(search_root):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for name in files:
                if name.lower() == target:
                    return Path(current) / name

    # Helpful diagnostic: show folders/files that resemble this ticker's Stage-1 output.
    found = []
    for search_root in roots:
        if not search_root.is_dir():
            continue
        for current, dirs, files in os.walk(search_root):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for name in files:
                low = name.lower()
                if ticker.lower() in low and ("trade_flow" in low or "features" in low):
                    found.append(str(Path(current) / name))
                    if len(found) >= 12:
                        break
            if len(found) >= 12:
                break
    details = "\n".join(f"  - {p}" for p in found) if found else "  Похожие файлы не найдены."
    raise FileNotFoundError(
        f"\nНе найден файл признаков первого этапа для {ticker}:\n"
        f"  {filename}\n"
        f"Искал в папках:\n"
        f"  {ROOT}\n  {ROOT.parent}\n"
        f"Похожие файлы:\n{details}\n\n"
        "Положи parquet в папку trade_flow_outputs_<TICKER> рядом со скриптом "
        "или проверь, что Stage 1 завершился и создал этот файл."
    )


def read_inputs(ticker):
    fp = locate_feature_file(ticker)
    try:
        import pyarrow.parquet as pq
        available = set(pq.ParquetFile(fp).schema_arrow.names)
    except Exception:
        # pandas will provide the detailed parquet read error if metadata inspection fails.
        available = None

    required = ["trade_date", "price", "size", "notional", "sign"]
    optional = [
        "signed_size",
        "tfi_volume_25", "tfi_volume_50", "tfi_volume_100",
        "tfi_notional_25", "tfi_notional_50", "tfi_notional_100",
        "tfi_count_25", "tfi_count_50", "tfi_count_100",
    ]
    if available is not None:
        missing = [c for c in required if c not in available]
        if missing:
            raise ValueError(
                f"{ticker}: найден parquet {fp}, но в нём отсутствуют обязательные "
                f"колонки: {missing}. Доступные колонки: {sorted(available)}"
            )
        columns = required + [c for c in optional if c in available]
    else:
        columns = required + optional

    print(f"  Input: {fp}")
    return pd.read_parquet(fp, columns=columns)


def make_intraday_targets(price):
    """Build forward transaction-price returns without crossing trade_date."""
    n = len(price)
    out = {}
    for h in (5, 10):
        future = np.full(n, np.nan)
        if n > h:
            future[:-h] = price[h:]
        out[f"ret_next_{h}_trades_bps"] = 1e4 * (future / price - 1.0)

    # Index of the next trade with a different transaction price.
    next_change = np.full(n, -1, dtype=np.int64)
    nxt = -1
    for i in range(n - 2, -1, -1):
        if price[i + 1] != price[i]:
            nxt = i + 1
        next_change[i] = nxt if nxt >= 0 and price[nxt] != price[i] else -1

    for k in (1, 3, 5):
        target = np.full(n, np.nan)
        for i in range(n):
            j = next_change[i]
            step = 1
            while j >= 0 and step < k:
                j = next_change[j]
                step += 1
            if j >= 0 and step == k:
                target[i] = 1e4 * (price[j] / price[i] - 1.0)
        out[f"ret_after_{k}_distinct_price_changes_bps"] = target
    return out


def add_intraday_features(f):
    """Rebuild bounded TFI from primitive columns, resetting at each session."""
    frames = []
    for day, g0 in f.groupby("trade_date", sort=False, dropna=False):
        g = g0.copy()
        price = pd.to_numeric(g["price"], errors="coerce").to_numpy(float)
        size = pd.to_numeric(g["size"], errors="coerce").to_numpy(float)
        notional = pd.to_numeric(g["notional"], errors="coerce").to_numpy(float)
        sign = pd.to_numeric(g["sign"], errors="coerce").to_numpy(float)

        # Reject malformed sides; valid encoded side should be exactly +/-1.
        sign = np.where(np.isin(sign, (-1.0, 1.0)), sign, np.nan)
        signed_size = sign * size
        signed_notional = sign * notional
        for w in WINDOWS:
            den_v = pd.Series(size).rolling(w, min_periods=w).sum().to_numpy()
            num_v = pd.Series(signed_size).rolling(w, min_periods=w).sum().to_numpy()
            den_n = pd.Series(notional).rolling(w, min_periods=w).sum().to_numpy()
            num_n = pd.Series(signed_notional).rolling(w, min_periods=w).sum().to_numpy()
            den_c = pd.Series(sign).rolling(w, min_periods=w).count().to_numpy()
            num_c = pd.Series(sign).rolling(w, min_periods=w).sum().to_numpy()
            g[f"tfi_volume_{w}_clean"] = np.divide(
                num_v, den_v, out=np.full(len(g), np.nan), where=den_v != 0)
            g[f"tfi_notional_{w}_clean"] = np.divide(
                num_n, den_n, out=np.full(len(g), np.nan), where=den_n != 0)
            g[f"tfi_count_{w}_clean"] = np.divide(
                num_c, den_c, out=np.full(len(g), np.nan), where=den_c != 0)

        for h in (5, 20):
            past = np.full(len(g), np.nan)
            if len(g) > h:
                past[h:] = 1e4 * np.log(price[h:] / price[:-h])
            g[f"past_return_{h}_trades_bps_clean"] = past

        targets = make_intraday_targets(price)
        for name, values in targets.items():
            g[name + "_intraday"] = values
        frames.append(g)

    return pd.concat(frames, axis=0).sort_index()


def quantile_bins(x, q=3):
    s = pd.Series(x)
    valid = s.notna()
    out = pd.Series(np.nan, index=s.index)
    if valid.sum() < q * 100:
        return out
    # Rank first avoids qcut failure when many observations are tied at zero.
    out.loc[valid] = pd.qcut(s.loc[valid].rank(method="first"), q=q, labels=False)
    return out


def bootstrap_day_spread(df, signal, target, reps=BOOT_REPS, seed=SEED):
    d = df[["trade_date", signal, target]].dropna()
    if d.empty or d["trade_date"].nunique() < 5:
        return np.nan, np.nan, np.nan, int(d["trade_date"].nunique())
    lo, hi = d[signal].quantile([.2, .8])
    low = d.loc[d[signal] <= lo, target]
    high = d.loc[d[signal] >= hi, target]
    point = float(high.mean() - low.mean())
    # Aggregate once by day; bootstrap only compact daily sums/counts,
    # never concatenate millions of trade rows inside each replication.
    d = d.copy()
    d["_low_sum"] = np.where(d[signal] <= lo, d[target], 0.0)
    d["_low_n"] = (d[signal] <= lo).astype(np.int64)
    d["_high_sum"] = np.where(d[signal] >= hi, d[target], 0.0)
    d["_high_n"] = (d[signal] >= hi).astype(np.int64)
    daily = d.groupby("trade_date", sort=False)[
        ["_low_sum", "_low_n", "_high_sum", "_high_n"]
    ].sum()
    days = len(daily)
    rng = np.random.default_rng(seed)
    vals = []
    a = daily.to_numpy(dtype=float)
    for _ in range(reps):
        ix = rng.integers(0, days, size=days)
        z = a[ix].sum(axis=0)
        if z[1] > 0 and z[3] > 0:
            vals.append(float(z[2] / z[3] - z[0] / z[1]))
    if not vals:
        return point, np.nan, np.nan, days
    return point, float(np.quantile(vals, .025)), float(np.quantile(vals, .975)), days


def main():
    integrity_rows, conditional_rows, bootstrap_rows = [], [], []
    report = [
        "# JERICHON — Trade Flow second-stage research", "",
        "Два исследования: проверка целостности TFI и intraday momentum-conditioned response.", "",
        "Все rolling-признаки и forward targets пересчитываются внутри trade_date, без перехода через ночь.",
        "Цены — transaction price, не mid-price. Направление BUYSELL/sign не считается подтверждённой стороной агрессора.", "",
    ]

    for ticker in TICKERS:
        print(f"Processing {ticker} ...")
        f = read_inputs(ticker)
        g = add_intraday_features(f)

        # Integrity: per-trade signed size must equal sign * size and TFI must be bounded.
        sign = pd.to_numeric(g["sign"], errors="coerce")
        size = pd.to_numeric(g["size"], errors="coerce")
        expected_signed_size = sign * size
        if "signed_size" in g.columns:
            stored_signed = pd.to_numeric(g["signed_size"], errors="coerce")
            signed_match = np.isclose(
                stored_signed, expected_signed_size,
                rtol=1e-7, atol=1e-8, equal_nan=True
            )
            signed_match_share = float(np.mean(signed_match))
        else:
            signed_match_share = np.nan
        row = {
            "ticker": ticker,
            "rows": len(g),
            "days": g["trade_date"].nunique(),
            "signed_size_match_share": signed_match_share,
        }
        for w in WINDOWS:
            for kind in ("volume", "notional", "count"):
                clean = g[f"tfi_{kind}_{w}_clean"]
                row[f"clean_tfi_{kind}_{w}_outside_[-1,1]"] = int(((clean < -1.000001) | (clean > 1.000001)).sum())
                stored_col = f"tfi_{kind}_{w}"
                if stored_col in g:
                    old = pd.to_numeric(g[stored_col], errors="coerce")
                    valid = old.notna() & clean.notna()
                    row[f"stored_{kind}_{w}_outside_[-1,1]"] = int(((old < -1.000001) | (old > 1.000001)).sum())
                    row[f"stored_{kind}_{w}_vs_clean_corr"] = (
                        float(old[valid].corr(clean[valid])) if valid.sum() > 100 else np.nan)
        integrity_rows.append(row)

        # Study 1: TFI adds information after conditioning on recent price movement.
        # Study 2: compare volume/notional/count definitions and response horizons.
        for w in WINDOWS:
            for kind in ("volume", "notional", "count"):
                signal = f"tfi_{kind}_{w}_clean"
                for mom_h in (5, 20):
                    mb = quantile_bins(g[f"past_return_{mom_h}_trades_bps_clean"], q=3)
                    for target in TARGETS:
                        target_clean = target + "_intraday"
                        tmp = g[["trade_date", signal, target_clean]].copy()
                        tmp["momentum_bucket"] = mb
                        tmp = tmp.dropna()
                        if len(tmp) < 1000:
                            continue
                        # Within each past-return tercile, compare bottom/top TFI quintiles.
                        for m, mg in tmp.groupby("momentum_bucket"):
                            if len(mg) < 500:
                                continue
                            qlo, qhi = mg[signal].quantile([.2, .8])
                            low = mg.loc[mg[signal] <= qlo, target_clean]
                            high = mg.loc[mg[signal] >= qhi, target_clean]
                            conditional_rows.append({
                                "ticker": ticker, "signal": signal,
                                "momentum_control": f"past_return_{mom_h}_trades_bps",
                                "momentum_tercile": int(m), "target": target,
                                "n": len(mg), "low_flow_mean_bps": float(low.mean()),
                                "high_flow_mean_bps": float(high.mean()),
                                "high_minus_low_bps": float(high.mean() - low.mean()),
                                "low_flow_up_probability": float((low > 0).mean()),
                                "high_flow_up_probability": float((high > 0).mean()),
                            })

                # Day-block bootstrap without momentum stratification: descriptive
                # benchmark for each clean flow definition and target.
                for target in TARGETS:
                    target_clean = target + "_intraday"
                    tmp = g[["trade_date", signal, target_clean]].rename(
                        columns={target_clean: "target"}).copy()
                    eff, lo, hi, days = bootstrap_day_spread(
                        tmp, signal, "target", seed=SEED + len(bootstrap_rows))
                    bootstrap_rows.append({
                        "ticker": ticker, "signal": signal, "target": target,
                        "spread_high_minus_low_bps": eff, "ci_low": lo,
                        "ci_high": hi, "days": days,
                        "ci_excludes_zero": bool(lo > 0 or hi < 0) if pd.notna(lo) and pd.notna(hi) else False,
                    })

        # Simple headline diagnostics per ticker.
        report += [
            f"## {ticker}", "",
            f"- Строк: {len(g):,}; торговых дат: {g['trade_date'].nunique():,}.",
            f"- Доля совпадения stored signed_size с sign × size: "
            f"{row['signed_size_match_share']:.3%}."
            if pd.notna(row["signed_size_match_share"])
            else "- Колонка signed_size отсутствует; проверка пропущена.",
        ]
        bad = []
        for w in WINDOWS:
            nbad = row.get(f"stored_volume_{w}_outside_[-1,1]", 0)
            if nbad:
                bad.append(f"tfi_volume_{w}: {nbad:,} значений вне [-1,1]")
        report.append("- Проверка сохранённых TFI: " + ("; ".join(bad) if bad else "нарушений диапазона не найдено."))
        report.append("")

    idf = pd.DataFrame(integrity_rows)
    cdf = pd.DataFrame(conditional_rows)
    bdf = pd.DataFrame(bootstrap_rows)

    idf.to_csv(OUT / "integrity_audit.csv", index=False, encoding="utf-8-sig")
    cdf.to_csv(OUT / "momentum_conditioned_results.csv", index=False, encoding="utf-8-sig")
    bdf.to_csv(OUT / "clean_tfi_day_bootstrap.csv", index=False, encoding="utf-8-sig")

    report += [
        "## Как читать", "",
        "- `integrity_audit.csv`: проверка signed_size и того, что TFI находится в [-1, 1].",
        "- `momentum_conditioned_results.csv`: сравнение верхнего и нижнего квинтилей потока внутри каждого терциля прошлой доходности.",
        "- `clean_tfi_day_bootstrap.csv`: разница верхнего/нижнего квинтилей с bootstrap по целым торговым дням.",
        "- Если исходный `tfi_volume_*` выходит за [-1, 1], это ошибка/несовпадение формулы или версии данных; для второго этапа используется пересчёт из sign × size.",
        "- Это conditional association, не доказательство причинности и не готовая торговая стратегия.", "",
    ]
    (OUT / "second_stage_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\nDone. Compact results:")
    for name in ("second_stage_report.md", "integrity_audit.csv",
                 "momentum_conditioned_results.csv", "clean_tfi_day_bootstrap.csv"):
        q = OUT / name
        print(f"  {q.name}: {q.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
