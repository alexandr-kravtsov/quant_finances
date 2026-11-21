from pathlib import Path
import sys, importlib, warnings
import pandas as pd

_F = Path(__file__).resolve()
ROOT = next(p for p in [_F, *_F.parents] if (p / ".git").exists() or ((p / "src").is_dir() and (p / "research").is_dir()))
DATA_ROOT = ROOT / "data"
_hr = {p.parent for p in DATA_ROOT.rglob("realtime") if p.is_dir()}
if not _hr:
    raise FileNotFoundError(f"no realtime/ dirs under {DATA_ROOT}")
HISTORY_ROOT = min(_hr, key=lambda d: "sample_data" in d.parts)
for _p in (str(ROOT), str(DATA_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_JFP = next(ROOT.rglob("jerichon_feature_pipeline.py"), None)
if _JFP is None:
    raise FileNotFoundError(f"jerichon_feature_pipeline.py not found under {ROOT}")
jfp = importlib.import_module(".".join(_JFP.relative_to(ROOT).with_suffix("").parts))

SOURCE_KEYS = ("trades", "tradestats", "obstats", "orderstats")


def available_dates(ticker, history_root=HISTORY_ROOT):
    dates = set()
    for kind in ("realtime", "supercandles"):
        p = history_root / kind / ticker
        if p.exists():
            dates.update(x.name for x in p.iterdir() if x.is_dir())
    return sorted(d for d in dates if len(d) == 10 and d[4] == "-" and d[7] == "-")


def discover_tickers(history_root=HISTORY_ROOT):
    root = history_root / "realtime"
    if not root.exists():
        return []
    good = []
    for ticker in sorted(p.name for p in root.iterdir() if p.is_dir()):
        dates = available_dates(ticker, history_root)
        has_trades = any((history_root / "realtime" / ticker / d / "trades.parquet").exists() for d in dates)
        has_sc = all(any((history_root / "supercandles" / ticker / d / f"{k}.parquet").exists() for d in dates) for k in ("tradestats", "obstats", "orderstats"))
        if has_trades and has_sc:
            good.append(ticker)
    return good


def load_local_history(ticker, history_root=HISTORY_ROOT):
    frames = {k: [] for k in ("candles", "trades", "orderbook", "tradestats", "obstats", "orderstats")}
    dates = available_dates(ticker, history_root)
    if not dates:
        raise FileNotFoundError(f"no saved dates for {ticker}")
    for day in dates:
        for key in ("candles", "trades", "orderbook"):
            path = history_root / "realtime" / ticker / day / f"{key}.parquet"
            if path.exists():
                try:
                    x = pd.read_parquet(path)
                    if not x.empty:
                        x = x.copy()
                        x["file_date"] = day
                        frames[key].append(x)
                except Exception as exc:
                    print(f"WARNING read {path}: {exc}")
        for key in ("tradestats", "obstats", "orderstats"):
            path = history_root / "supercandles" / ticker / day / f"{key}.parquet"
            if path.exists():
                try:
                    x = pd.read_parquet(path)
                    if not x.empty:
                        x = x.copy()
                        x["file_date"] = day
                        frames[key].append(x)
                except Exception as exc:
                    print(f"WARNING read {path}: {exc}")
    data = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in frames.items()}
    missing = [k for k in SOURCE_KEYS if data[k].empty]
    if missing:
        raise RuntimeError(f"{ticker}: missing required sources {missing}")
    return data, dates


def load_all_trades(ticker, history_root=HISTORY_ROOT):
    ticker_dir = history_root / "realtime" / ticker.upper()
    if not ticker_dir.exists():
        raise FileNotFoundError(f"ticker dir not found: {ticker_dir}")
    dates = sorted(d.name for d in ticker_dir.iterdir() if d.is_dir() and len(d.name) == 10 and d.name[4] == "-" and d.name[7] == "-")
    if not dates:
        raise FileNotFoundError(f"no saved dates for {ticker} in {ticker_dir}")
    frames = []
    for day in dates:
        p = ticker_dir / day / "trades.parquet"
        if p.exists():
            try:
                df = pd.read_parquet(p)
                if not df.empty:
                    df = df.copy()
                    df["file_date"] = day
                    frames.append(df)
            except Exception as e:
                print(f"WARNING: could not read {p}: {e}")
    if not frames:
        raise FileNotFoundError(f"no trades.parquet found for {ticker}")
    out = pd.concat(frames, ignore_index=True)
    print(f"collected {len(dates)} dates: {dates[0]} -> {dates[-1]}, rows: {len(out):,}")
    return out


def discover_trade_file(ticker, root=DATA_ROOT):
    if not root.exists():
        return None
    found = []
    for pat in ("**/*trades*.parquet", "**/*trade*.parquet", "**/*trades*.csv", "**/*trade*.csv"):
        found.extend(root.glob(pat))
    found = [p for p in found if ticker.upper() in str(p).upper()]
    found.sort(key=lambda p: (ticker.upper() not in p.name.upper(), len(str(p))))
    return found[0] if found else None


def load_frame(path):
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported file: {path}")


def build_feature_dataset(ticker, data, threshold=10_000_000, expected_base=73, expected_ml=474):
    missing = [n for n in ("build_dollar_bars_from_trades", "_base", "_build_sc_features", "_attach_source") if not hasattr(jfp, n)]
    if missing:
        raise RuntimeError(f"jerichon_feature_pipeline.py missing helpers: {missing}")
    bars = jfp.build_dollar_bars_from_trades(data["trades"], threshold=threshold)
    if bars is None or bars.empty:
        raise RuntimeError(f"{ticker}: Dollar Bar builder returned empty")
    bars = bars.copy()
    if "bar_ts" not in bars.columns:
        if "close_time" in bars.columns:
            bars["bar_ts"] = pd.to_datetime(bars["close_time"], errors="coerce", utc=True)
        elif "ts" in bars.columns:
            bars["bar_ts"] = pd.to_datetime(bars["ts"], errors="coerce", utc=True)
        else:
            raise RuntimeError(f"{ticker}: no bar_ts/close_time/ts. cols: {list(bars.columns)}")
    else:
        bars["bar_ts"] = pd.to_datetime(bars["bar_ts"], errors="coerce", utc=True)
    bars = bars.dropna(subset=["bar_ts"]).sort_values("bar_ts").reset_index(drop=True)
    aliases = {"open": ("OPEN", "open_price", "OPEN_PRICE"), "high": ("HIGH", "high_price", "HIGH_PRICE"), "low": ("LOW", "low_price", "LOW_PRICE"), "close": ("CLOSE", "close_price", "CLOSE_PRICE"), "volume": ("VOLUME", "qty", "volume_qty")}
    for target, options in aliases.items():
        if target not in bars.columns:
            source = next((c for c in options if c in bars.columns), None)
            if source:
                bars[target] = bars[source]
    for col in ("open", "high", "low", "close"):
        if col not in bars.columns:
            raise RuntimeError(f"{ticker}: no {col} in Dollar Bars; cols={list(bars.columns)}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="DataFrame is highly fragmented", category=pd.errors.PerformanceWarning)
        base_frame = jfp._base(bars.copy())
    base_frame = base_frame.copy()
    if "bar_ts" not in base_frame.columns:
        if len(base_frame) != len(bars):
            raise RuntimeError(f"{ticker}: _base without bar_ts changed row count")
        base_frame["bar_ts"] = bars["bar_ts"].to_numpy()
    base_cols = [c for c in base_frame.columns if c.startswith("base__")]
    if not base_cols:
        excluded = {"bar_ts", "bar_ts_utc", "ts", "ts_utc", "close_time"}
        base_cols = [c for c in base_frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(base_frame[c])]
    if len(base_cols) != expected_base:
        raise RuntimeError(f"{ticker}: base inventory {len(base_cols)}, expected {expected_base}")
    bars_for_sc = bars.copy()
    if "ts" not in bars_for_sc.columns:
        bars_for_sc["ts"] = bars_for_sc["bar_ts"]
    if "CLOSE" not in bars_for_sc.columns:
        bars_for_sc["CLOSE"] = bars_for_sc["close"]
    sc_frames = jfp._build_sc_features({k: data[k] for k in ("tradestats", "obstats", "orderstats")}, bars_for_sc)
    if not isinstance(sc_frames, dict):
        raise TypeError(f"{ticker}: _build_sc_features returned {type(sc_frames)}, expected dict")
    causal = bars[["bar_ts"]].copy()
    for source in ("tradestats", "obstats", "orderstats"):
        if source not in sc_frames or sc_frames[source] is None or sc_frames[source].empty:
            raise RuntimeError(f"{ticker}: no feature frame for {source}")
        causal = jfp._attach_source(causal, sc_frames[source], data[source], source)
        if causal is None or causal.empty:
            raise RuntimeError(f"{ticker}: _attach_source empty for {source}")
    base_frame["bar_ts"] = pd.to_datetime(base_frame["bar_ts"], errors="coerce", utc=True)
    causal["bar_ts"] = pd.to_datetime(causal["bar_ts"], errors="coerce", utc=True)
    causal = causal.merge(base_frame, on="bar_ts", how="left", validate="one_to_one")
    provenance_cols = [c for c in causal.columns if c.endswith("__event_ts") or c.endswith("__available_at")]
    sc_prefixes = ("tradestats__", "obstats__", "orderstats__")
    sc_cols = [c for c in causal.columns if c.startswith(sc_prefixes) and c not in provenance_cols and pd.api.types.is_numeric_dtype(causal[c])]
    feature_cols = base_cols + sc_cols
    if len(feature_cols) != expected_ml:
        raise RuntimeError(f"{ticker}: ML inventory {len(feature_cols)}, expected {expected_ml}")
    dataset = causal[["bar_ts"] + feature_cols + provenance_cols].copy()
    dataset = dataset.sort_values("bar_ts").reset_index(drop=True)
    market = bars[["bar_ts", "open", "high", "low", "close"]].copy()
    market["bar_ts"] = pd.to_datetime(market["bar_ts"], utc=True)
    market = market.merge(dataset[["bar_ts"] + feature_cols], on="bar_ts", how="inner", validate="one_to_one")
    market = market.sort_values("bar_ts").reset_index(drop=True)
    return bars, market, dataset, feature_cols, threshold


def get_features(ticker, threshold=10_000_000, history_root=HISTORY_ROOT):
    data, dates = load_local_history(ticker, history_root)
    bars, market, dataset, feature_cols, thr = build_feature_dataset(ticker, data, threshold)
    return bars, market, dataset, feature_cols, dates