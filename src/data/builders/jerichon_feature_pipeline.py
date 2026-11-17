"""Canonical causal JERICHON feature pipeline.

Raw AlgoPack -> Dollar Bars -> existing Base FeatureBuilder -> causal backward
Super-Candle joins -> audit -> FINAL_DATASET. Labels are deliberately absent.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable
import numpy as np
import pandas as pd
from src.data.providers.algopack_provider import AlgoPackProvider
from src.data.builders.dollar_bars import InformationDrivenBars
from src.data.builders.build_features import FeatureBuilder
from src.data.builders.micro_features import MicroFeatures

BASE_FEATURE_COUNT=73
SUPERCANDLE_FEATURE_COUNT=401
FULL_FEATURE_COUNT=474
SOURCES=("tradestats","obstats","orderstats")
PROVENANCE=tuple(f"{s}__{k}" for s in SOURCES for k in ("event_ts","available_at"))

def _utc(s): return pd.to_datetime(s, errors="coerce", utc=True)
def _col(df,names:Iterable[str]):
    m={str(c).lower():c for c in df.columns}
    for n in names:
        if n.lower() in m: return m[n.lower()]
    return None

def _event(df):
    c=_col(df,("event_ts","ts","timestamp","datetime"))
    if c is None: raise ValueError("Super Candle source has no event timestamp")
    return _utc(df[c])

def _available(df):
    c=_col(df,("available_at","availability_ts","availability","SYSTIME","UPDATETIME","system_time","update_time"))
    if c is None: raise ValueError("Super Candle source has no available_at/SYSTIME/UPDATETIME")
    return _utc(df[c])



# Exact Super Candle feature engineering ported from jerichon_mark_63(3).ipynb.
def _prep_exact(df):
    if df is None or df.empty or "ts" not in df.columns:
        return pd.DataFrame()
    x = df.copy()
    x["ts"] = pd.to_datetime(x["ts"], errors="coerce", utc=True)
    for c in x.columns:
        if c not in {"ts", "ticker", "secid", "super_kind", "file_date", "date", "SYSTIME"}:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

def _prepare_ts(df):
    if df is None or df.empty or "ts" not in df.columns:
        return pd.DataFrame()
    x = df.copy()
    x["ts"] = pd.to_datetime(x["ts"], errors="coerce", utc=True)
    return x.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

def _safe_div(a, b, eps=1e-12):
    return a / (b.abs() + eps)

def _imb(a, b, eps=1e-12):
    return (a-b) / (a+b).abs().add(eps)

def _add_dyn(df, cols, windows=(3, 6, 12, 24)):
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        out[f"{c}_chg"] = s.diff()
        out[f"{c}_ret"] = s.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        for w in windows:
            minp = max(2, min(w, 3))
            mean = s.rolling(w, min_periods=minp).mean()
            std = s.rolling(w, min_periods=minp).std()
            out[f"{c}_z{w}"] = (s - mean) / (std + 1e-12)
            out[f"{c}_ma{w}"] = mean
    return out

def _build_sc_features(raw_sources, bars_for_features):
    feature_frames = {}
    feature_groups = {}
    # --- tradestats: target 94 ---
    ts0 = _prep_exact(raw_sources["tradestats"])
    if not ts0.empty:
        tsf = ts0[["ts"]].copy()
        tsf["ts_trade_imbalance"] = _imb(ts0["trades_b"],ts0["trades_s"])
        tsf["ts_volume_imbalance"] = _imb(ts0["vol_b"],ts0["vol_s"])
        tsf["ts_value_imbalance"] = _imb(ts0["val_b"],ts0["val_s"])
        tsf["ts_buy_trade_share"] = _safe_div(ts0["trades_b"],ts0["trades_b"]+ts0["trades_s"])
        tsf["ts_buy_volume_share"] = _safe_div(ts0["vol_b"],ts0["vol_b"]+ts0["vol_s"])
        tsf["ts_buy_value_share"] = _safe_div(ts0["val_b"],ts0["val_b"]+ts0["val_s"])
        tsf["ts_vwap_b_s_diff"] = ts0["pr_vwap_b"]-ts0["pr_vwap_s"]
        tsf["ts_vwap_b_s_bps"] = (ts0["pr_vwap_b"]-ts0["pr_vwap_s"])/((ts0["pr_vwap_b"]+ts0["pr_vwap_s"])/2+1e-12)*1e4
        tsf["ts_buy_vwap_displacement"] = (ts0["pr_vwap_b"]-ts0["pr_vwap"])/(ts0["pr_vwap"]+1e-12)
        tsf["ts_sell_vwap_displacement"] = (ts0["pr_vwap_s"]-ts0["pr_vwap"])/(ts0["pr_vwap"]+1e-12)
        tsf["ts_value_per_trade"] = _safe_div(ts0["val"],ts0["trades"])
        tsf["ts_volume_per_trade"] = _safe_div(ts0["vol"],ts0["trades"])
        tsf["ts_flow_x_return"] = ts0["disb"]*ts0["pr_change"]
        tsf["ts_flow_x_volatility"] = ts0["disb"]*ts0["pr_std"]
        dyn=["ts_trade_imbalance","ts_volume_imbalance","ts_value_imbalance","ts_buy_volume_share",
             "ts_buy_value_share","ts_vwap_b_s_bps","ts_value_per_trade","ts_flow_x_return"]
        tsf=_add_dyn(tsf,dyn)
        feature_frames["tradestats_pressure"]=tsf
        feature_groups["tradestats_pressure"]=[c for c in tsf.columns if c!="ts"]

    # --- orderstats: target 171 ---
    os0 = _prep_exact(raw_sources["orderstats"])
    if not os0.empty:
        osf=os0[["ts"]].copy()
        osf["os_put_orders_imb"]=_imb(os0["put_orders_b"],os0["put_orders_s"])
        osf["os_put_volume_imb"]=_imb(os0["put_vol_b"],os0["put_vol_s"])
        osf["os_put_value_imb"]=_imb(os0["put_val_b"],os0["put_val_s"])
        osf["os_put_buy_share"]=_safe_div(os0["put_vol_b"],os0["put_vol_b"]+os0["put_vol_s"])
        osf["os_put_value_buy_share"]=_safe_div(os0["put_val_b"],os0["put_val_b"]+os0["put_val_s"])
        osf["os_cancel_orders_imb"]=_imb(os0["cancel_orders_b"],os0["cancel_orders_s"])
        osf["os_cancel_volume_imb"]=_imb(os0["cancel_vol_b"],os0["cancel_vol_s"])
        osf["os_cancel_value_imb"]=_imb(os0["cancel_val_b"],os0["cancel_val_s"])
        osf["os_cancel_buy_share"]=_safe_div(os0["cancel_vol_b"],os0["cancel_vol_b"]+os0["cancel_vol_s"])
        osf["os_cancel_to_put_orders"]=_safe_div(os0["cancel_orders"],os0["put_orders"])
        osf["os_cancel_to_put_volume"]=_safe_div(os0["cancel_vol"],os0["put_vol"])
        osf["os_cancel_to_put_value"]=_safe_div(os0["cancel_val"],os0["put_val"])
        osf["os_net_orders"]=os0["put_orders"]-os0["cancel_orders"]
        osf["os_net_volume"]=os0["put_vol"]-os0["cancel_vol"]
        osf["os_net_value"]=os0["put_val"]-os0["cancel_val"]
        osf["os_net_volume_imb"]=_imb(os0["put_vol_b"]-os0["cancel_vol_b"],os0["put_vol_s"]-os0["cancel_vol_s"])
        osf["os_net_value_imb"]=_imb(os0["put_val_b"]-os0["cancel_val_b"],os0["put_val_s"]-os0["cancel_val_s"])
        osf["os_put_vwap_b_s_bps"]=(os0["put_vwap_b"]-os0["put_vwap_s"])/((os0["put_vwap_b"]+os0["put_vwap_s"])/2+1e-12)*1e4
        osf["os_cancel_vwap_b_s_bps"]=(os0["cancel_vwap_b"]-os0["cancel_vwap_s"])/((os0["cancel_vwap_b"]+os0["cancel_vwap_s"])/2+1e-12)*1e4
        osf["os_cancel_vs_put_vwap_b"]=(os0["cancel_vwap_b"]-os0["put_vwap_b"])/(os0["put_vwap_b"]+1e-12)
        osf["os_cancel_vs_put_vwap_s"]=(os0["cancel_vwap_s"]-os0["put_vwap_s"])/(os0["put_vwap_s"]+1e-12)
        dyn=["os_put_orders_imb","os_put_volume_imb","os_put_value_imb","os_cancel_orders_imb",
             "os_cancel_volume_imb","os_cancel_value_imb","os_cancel_to_put_orders","os_cancel_to_put_volume",
             "os_cancel_to_put_value","os_net_volume","os_net_value","os_net_volume_imb","os_net_value_imb",
             "os_put_vwap_b_s_bps","os_cancel_vwap_b_s_bps"]
        osf=_add_dyn(osf,dyn)
        feature_frames["orderstats_pressure"]=osf
        feature_groups["orderstats_pressure"]=[c for c in osf.columns if c!="ts"]

    # --- obstats: 136 features ---
    obs0 = _prepare_ts(raw_sources["obstats"])
    if not obs0.empty:
        obsf=pd.DataFrame({"ts":obs0["ts"]})
        obsf["ob_total_volume"]=obs0["vol_b"].fillna(0)+obs0["vol_s"].fillna(0)
        obsf["ob_total_value"]=obs0["val_b"].fillna(0)+obs0["val_s"].fillna(0)
        obsf["ob_vol_imb_bbo_vs_deep"]=obs0["imbalance_vol_bbo"]-obs0["imbalance_vol"]
        obsf["ob_val_imb_bbo_vs_deep"]=obs0["imbalance_val_bbo"]-obs0["imbalance_val"]
        obsf["ob_bid_value_share"]=obs0["val_b"]/(obs0["val_b"]+obs0["val_s"]+1e-12)
        obsf["ob_bid_volume_share"]=obs0["vol_b"]/(obs0["vol_b"]+obs0["vol_s"]+1e-12)
        obsf["ob_bid_level_share"]=obs0["levels_b"]/(obs0["levels_b"]+obs0["levels_s"]+1e-12)
        obsf["ob_lv10_to_bbo_spread_ratio"]=obs0["spread_lv10"]/(obs0["spread_bbo"]+1e-12)
        obsf["ob_1mio_to_bbo_spread_ratio"]=obs0["spread_1mio"]/(obs0["spread_bbo"]+1e-12)
        obsf["ob_vwap_b_s_bps"]=(obs0["vwap_b"]-obs0["vwap_s"])/(obs0["vwap_s"]+1e-12)*1e4
        obsf["ob_vwap_1mio_b_s_bps"]=(obs0["vwap_b_1mio"]-obs0["vwap_s_1mio"])/(obs0["vwap_s_1mio"]+1e-12)*1e4
        obsf["ob_1mio_liquidity_cost_bps"]=(obs0["spread_1mio"]-obs0["spread_bbo"])/(obs0["spread_bbo"]+1e-12)*1e4
        px=bars_for_features[["ts","CLOSE"]].rename(columns={"CLOSE":"close"}).sort_values("ts")
        tmp=pd.merge_asof(obs0[["ts","vwap_b","vwap_s","vwap_b_1mio","vwap_s_1mio"]].sort_values("ts"),
                          px,on="ts",direction="backward")
        obsf["ob_px_vs_vwap_b_bps"]=(tmp["close"]-tmp["vwap_b"])/(tmp["vwap_b"]+1e-12)*1e4
        obsf["ob_px_vs_vwap_s_bps"]=(tmp["close"]-tmp["vwap_s"])/(tmp["vwap_s"]+1e-12)*1e4
        obsf["ob_px_vs_vwap_b_1mio_bps"]=(tmp["close"]-tmp["vwap_b_1mio"])/(tmp["vwap_b_1mio"]+1e-12)*1e4
        obsf["ob_px_vs_vwap_s_1mio_bps"]=(tmp["close"]-tmp["vwap_s_1mio"])/(tmp["vwap_s_1mio"]+1e-12)*1e4
        dyn=["ob_total_volume","ob_total_value","ob_vol_imb_bbo_vs_deep","ob_val_imb_bbo_vs_deep",
             "ob_bid_value_share","ob_bid_volume_share","ob_bid_level_share","ob_lv10_to_bbo_spread_ratio",
             "ob_1mio_to_bbo_spread_ratio","ob_vwap_b_s_bps","ob_vwap_1mio_b_s_bps","ob_1mio_liquidity_cost_bps"]
        obsf=_add_dyn(obsf,dyn)
        feature_frames["obstats_topology"]=obsf
        feature_groups["obstats_topology"]=[c for c in obsf.columns if c!="ts"]


    expected = {"tradestats_pressure": 94, "orderstats_pressure": 171, "obstats_topology": 136}
    actual = {k: len(v) for k, v in feature_groups.items()}
    for group, count in expected.items():
        if actual.get(group, 0) != count:
            raise RuntimeError(f"Super Candle feature inventory mismatch: {group} expected {count}, got {actual.get(group, 0)}. Source columns may not match jerichon_mark_63(3).")
    return {"tradestats": feature_frames["tradestats_pressure"], "orderstats": feature_frames["orderstats_pressure"], "obstats": feature_frames["obstats_topology"]}

def _choose_availability(df, source_name):
    if df is None or df.empty:
        return pd.DataFrame(columns=["event_ts", "available_at"])
    if "ts" not in df.columns:
        raise RuntimeError(f"{source_name}: missing ts event timestamp")
    c = _col(df, ("SYSTIME", "UPDATETIME", "available_at", "availability_ts"))
    if c is None:
        raise RuntimeError(f"{source_name}: no SYSTIME/UPDATETIME availability timestamp; cannot safely causal-join")
    x = df.copy()
    x["event_ts"] = _utc(x["ts"])
    syst = pd.to_datetime(x[c], errors="coerce")
    if getattr(syst.dt, "tz", None) is not None:
        av_utc = syst.dt.tz_convert("UTC")
        av_msk = syst.dt.tz_convert("Europe/Moscow").dt.tz_convert("UTC")
    else:
        av_utc = syst.dt.tz_localize("UTC")
        av_msk = syst.dt.tz_localize("Europe/Moscow").dt.tz_convert("UTC")
    lag_u = (av_utc - x["event_ts"]).dt.total_seconds()
    lag_m = (av_msk - x["event_ts"]).dt.total_seconds()
    def score(lag):
        z = lag.dropna()
        if z.empty: return (1e9, 1e9, 1e9)
        return ((z < 0).mean(), abs(z.median() - 15), z.quantile(.95))
    x["available_at"] = av_utc if score(lag_u) <= score(lag_m) else av_msk
    x = x.dropna(subset=["event_ts", "available_at"])
    # Do not repair future/negative lags by shifting timestamps.
    x = x[x["event_ts"] <= x["available_at"]]
    return x[["event_ts", "available_at"]].drop_duplicates("event_ts", keep="last")

def _attach_source(bars, feature_frame, raw_source, source):
    timing = _choose_availability(raw_source, source)
    if feature_frame is None or feature_frame.empty or timing.empty:
        raise RuntimeError(f"{source}: no engineered features or validated availability rows")
    f = feature_frame.copy()
    f["event_ts"] = _utc(f["ts"])
    f = f.drop(columns=["ts"]).merge(timing, on="event_ts", how="inner")
    rename = {c: f"{source}__{c}" for c in f.columns if c not in ("event_ts", "available_at")}
    f = f.rename(columns=rename).rename(columns={"event_ts": f"{source}__event_ts", "available_at": f"{source}__available_at"})
    f = f.sort_values(f"{source}__available_at").drop_duplicates(f"{source}__available_at", keep="last")
    out = pd.merge_asof(bars.sort_values("bar_ts"), f, left_on="bar_ts", right_on=f"{source}__available_at", direction="backward", allow_exact_matches=True)
    ev, av = f"{source}__event_ts", f"{source}__available_at"
    if (out[av].notna() & (out[av] > out["bar_ts"])).any() or (out[ev].notna() & (out[ev] > out["bar_ts"])).any():
        raise RuntimeError(f"Causal violation in {source}")
    return out

def _trades_for_bars(trades):
    if trades is None or trades.empty: raise ValueError("AlgoPack trades are empty")
    req=["ts","PRICE","VALUE"]; miss=[c for c in req if c not in trades.columns]
    if miss: raise ValueError(f"trades missing required columns: {miss}")
    x=trades.copy(); x["ts"]=_utc(x["ts"]); x["PRICE"]=pd.to_numeric(x["PRICE"],errors="coerce"); x["VALUE"]=pd.to_numeric(x["VALUE"],errors="coerce")
    x=x.dropna(subset=req); x=x[(x.PRICE>0)&(x.VALUE>0)].copy()
    # AlgoPack VALUE is monetary turnover: quantity = VALUE / PRICE.
    return pd.DataFrame({"ts":x.ts,"price":x.PRICE,"quantity":x.VALUE/x.PRICE}).sort_values("ts").reset_index(drop=True)

def build_dollar_bars_from_trades(trades, threshold=13_000_000):
    b=InformationDrivenBars().build_dollar(_trades_for_bars(trades), threshold=threshold)
    if b is None or b.empty or "close_time" not in b.columns: raise RuntimeError("Dollar Bar builder returned no usable bars")
    b=b.copy(); b["bar_ts"]=_utc(b["close_time"]); return b.dropna(subset=["bar_ts"]).sort_values("bar_ts").drop_duplicates("bar_ts").reset_index(drop=True)

def _base(bars):
    """Build the 73 base features using the exact reference-notebook sequence.

    Reference: jerichon_mark_63(3).ipynb, cell 4:
      MicroFeatures(spread_window=20, volatility_window=20, intensity_window=20)
      -> FeatureBuilder(windows=[5, 10, 20], lags=[1, 2, 3], include_micro=False)

    The bar timestamp must be a DatetimeIndex before FeatureBuilder runs;
    otherwise its time features are silently omitted.
    """
    x = bars.copy()
    if "bar_ts" in x.columns:
        ts = _utc(x["bar_ts"])
    elif "close_time" in x.columns:
        ts = _utc(x["close_time"])
    elif isinstance(x.index, pd.DatetimeIndex):
        ts = _utc(pd.Series(x.index, index=x.index))
    else:
        raise RuntimeError("Base feature input has no bar_ts/close_time timestamp")

    x = x.drop(columns=[c for c in ("bar_ts", "close_time", "ts", "datetime", "timestamp") if c in x.columns])
    # Reference cell renames OHLCV to lowercase before the micro/feature builders.
    mp = {u: l for u, l in (("OPEN", "open"), ("HIGH", "high"), ("LOW", "low"), ("CLOSE", "close"), ("VOLUME", "volume")) if u in x.columns and l not in x.columns}
    x = x.rename(columns=mp)
    x.index = pd.DatetimeIndex(ts)
    x.index.name = "ts"
    x = x.loc[~x.index.isna()].sort_index()
    x = x.loc[~x.index.duplicated(keep="first")]

    micro = MicroFeatures(
        spread_window=20,
        volatility_window=20,
        intensity_window=20,
    )
    bars_with_micro = micro.build_all_features(x)
    fb = FeatureBuilder(
        windows=[5, 10, 20],
        lags=[1, 2, 3],
        include_micro=False,
    )
    featured = fb.build_all(bars_with_micro, dropna=False)
    if not isinstance(featured, pd.DataFrame):
        raise RuntimeError("FeatureBuilder.build_all did not return a DataFrame")

    numeric = featured.select_dtypes(include=[np.number]).copy()
    if not isinstance(numeric.index, pd.DatetimeIndex):
        raise RuntimeError("Reference base builder lost the DatetimeIndex")
    numeric.index.name = "ts"
    numeric = numeric.reset_index()
    numeric["bar_ts"] = _utc(numeric["ts"])
    numeric = numeric.drop(columns=["ts"])
    cols = [c for c in numeric.columns if c != "bar_ts" and pd.api.types.is_numeric_dtype(numeric[c])]
    if len(cols) != BASE_FEATURE_COUNT:
        raise RuntimeError(
            f"Base feature inventory mismatch: expected {BASE_FEATURE_COUNT}, got {len(cols)}. "
            f"Reference sequence is now used (MicroFeatures -> FeatureBuilder windows=[5,10,20], "
            f"lags=[1,2,3]); inspect the actual Dollar Bar numeric columns and the repository's "
            f"build_features.py/micro_features.py against jerichon_mark_63(3)."
        )
    out = numeric[["bar_ts"] + cols].rename(columns={c: f"base__{c}" for c in cols})
    return out.dropna(subset=["bar_ts"]).sort_values("bar_ts").drop_duplicates("bar_ts", keep="last").reset_index(drop=True)

def build_feature_dataset(ticker,start_date,end_date,*,provider:Optional[AlgoPackProvider]=None,history_path="./data/for_ml/storage/history",archive_raw=True,include_provenance=True,strict=True,dollar_bar_threshold=10_000_000):
    provider=provider or AlgoPackProvider(); frames=[]
    for d in pd.date_range(start_date,end_date,freq="B"):
        date=d.strftime("%Y-%m-%d"); bundle=provider.fetch_day_bundle(ticker,date)
        if archive_raw: _archive(bundle,ticker,date,Path(history_path))
        trades = bundle.get("trades", pd.DataFrame())
        bars=build_dollar_bars_from_trades(trades,dollar_bar_threshold)
        bars_for_features = bars.copy()
        bars_for_features["ts"] = _utc(bars_for_features["bar_ts"])
        bars_for_features = bars_for_features.rename(columns={"open":"OPEN", "high":"HIGH", "low":"LOW", "close":"CLOSE", "volume":"VOLUME"})
        base_features = _base(bars)
        # Engineer the exact 94 + 171 + 136 SC feature groups from the reference notebook.
        raw_sources = {s: bundle.get(s, pd.DataFrame()) for s in SOURCES}
        sc_features = _build_sc_features(raw_sources, bars_for_features)
        out = base_features.sort_values("bar_ts")
        for source in SOURCES:
            out = _attach_source(out, sc_features[source], raw_sources[source], source)
        frames.append(out.sort_values("bar_ts"))
    if not frames: raise RuntimeError("No feature data produced")
    out=pd.concat(frames,ignore_index=True).sort_values("bar_ts").drop_duplicates("bar_ts").reset_index(drop=True)
    for s in SOURCES:
        av=f"{s}__available_at"; ev=f"{s}__event_ts"
        if av not in out or ev not in out: raise RuntimeError(f"Missing provenance columns for {s}")
        if (out[av].notna() & (out[av]>out.bar_ts)).any(): raise RuntimeError(f"Availability look-ahead detected for {s}")
        if (out[ev].notna() & (out[ev]>out.bar_ts)).any(): raise RuntimeError(f"Event-time look-ahead detected for {s}")
    provenance=[c for c in out.columns if c.endswith("__event_ts") or c.endswith("__available_at")]
    ml=[c for c in out.columns if c not in provenance and c not in ("bar_ts", "bar_ts_utc")]
    if len(ml)!=FULL_FEATURE_COUNT:
        groups={s:len([c for c in ml if c.startswith(s+"__")]) for s in SOURCES}
        base_count=len([c for c in ml if c.startswith("base__")])
        raise RuntimeError(f"Feature inventory mismatch: {len(ml)} ML features (base={base_count}, SC={groups}), expected 474 (73 + 94 + 171 + 136). Do not train until reconciled with jerichon_mark_63(3).")
    if not include_provenance: out=out.drop(columns=provenance,errors="ignore")
    return out

build_dataset=build_feature_dataset
