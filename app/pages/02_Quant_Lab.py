# app/pages/02_Quant_Lab.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import logging
import json
import pickle
import joblib

from src.data.builders.dollar_bars import InformationDrivenBars
from src.data.labeling.cusum import cusum_filter
from src.data.labeling.triple_barrier import triple_barrier_labeling
from src.data.builders.micro_features import MicroFeatures
from src.data.builders.build_features import FeatureBuilder

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("quant_lab")

st.set_page_config(
    page_title="🧠 Quant Lab — Оптимизация и обучение",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# СТИЛЬ
# ============================================================
st.markdown("""
<style>
    .main, .stApp { background-color: #0a0a0a; color: #d0d0d0; }
    .silver-glow {
        font-weight: 700;
        background: linear-gradient(90deg, #e8e8e8, #ffffff, #c0c0c0, #e8e8e8);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: silverShine 4s ease-in-out infinite;
        display: inline-block;
    }
    @keyframes silverShine {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    h1, h2, h3, h4 {
        color: #ff4444;
        text-shadow: 0 0 20px #8b0000, 0 0 40px #ff0000;
        font-family: 'Courier New', monospace;
    }
    .stButton>button {
        background: linear-gradient(135deg, #8b0000, #ff0000);
        color: white;
        border: 2px solid #ff4444;
        border-radius: 12px;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 0.6rem 2rem;
        transition: 0.3s;
        box-shadow: 0 0 20px #ff0000;
        width: 100%;
        height: auto !important;
        min-height: 55px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #a00000, #ff3333);
        box-shadow: 0 0 35px #ff4444;
    }
    [data-testid="stMetric"] {
        background: rgba(20,0,0,0.9);
        border: 2px solid #8b0000;
        border-radius: 14px;
        padding: 15px;
        box-shadow: 0 0 20px rgba(139,0,0,0.5);
    }
    [data-testid="stMetric"] label {
        color: #ff8888 !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetric"] div {
        color: #ffffff !important;
        font-size: 2.0rem !important;
        font-weight: 700;
        text-shadow: 0 0 10px #ff0000;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #8b0000, #ff0000, #ff4444) !important;
        box-shadow: 0 0 20px #ff0000 !important;
        height: 12px !important;
        border-radius: 8px !important;
    }
    .status-text {
        color: #ff8888;
        font-family: 'Courier New', monospace;
        font-size: 1.2rem;
        text-shadow: 0 0 10px #8b0000;
        padding: 10px;
        border: 1px solid #8b0000;
        border-radius: 10px;
        background: rgba(20,0,0,0.5);
        margin: 10px 0;
    }
    .status-text.success { color: #66ff88; border-color: #00ff44; box-shadow: 0 0 20px rgba(0,255,68,0.2); }
    .status-text.error { color: #ff4444; border-color: #ff0000; box-shadow: 0 0 20px rgba(255,0,0,0.3); }
    .save-path {
        color: #66ff88;
        background: #0a0a0a;
        padding: 10px;
        border: 1px solid #00ff44;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="silver-glow">🧠 QUANT LAB — ОПТИМИЗАЦИЯ ПАРАМЕТРОВ</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#8888ff; font-size:1.2rem;">⬡ ПОИСК ОПТИМАЛЬНЫХ ПАРАМЕТРОВ ДЛЯ ФОРМИРОВАНИЯ СОБЫТИЙ И ТОРГОВЛИ</p>', unsafe_allow_html=True)

# ============================================================
# ПАРАМЕТРЫ ОПТИМИЗАЦИИ
# ============================================================
ALL_TICKERS = ['AFLT','ALRS','GAZP','GMKN','LKOH','MAGN','MGNT','MTLR','NVTK','PLZL','ROSN','RUAL','SBER','SMLT','SPBE','T','TATN','VTBR','YDEX']

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.selectbox("🎯 Тикер", ALL_TICKERS, index=ALL_TICKERS.index("ROSN"))
with col2:
    model_type = st.selectbox("🧠 Модель", ["RandomForest", "XGBoost", "LightGBM", "CatBoost"], index=0)
with col3:
    n_trials = st.number_input("🔁 Итераций Optuna", min_value=10, max_value=200, value=50, step=5)

col1, col2, col3, col4 = st.columns(4)
with col1:
    train_ratio = st.slider("📊 Train/Test", 0.5, 0.9, 0.8, 0.05)
with col2:
    target_metric = st.selectbox("🎯 Целевая метрика", ["sharpe", "return", "return_per_dd"], index=0)
with col3:
    commission_pct = st.number_input("💸 Комиссия (%)", min_value=0.0, max_value=0.5, value=0.06, step=0.01) / 100
with col4:
    initial_capital = st.number_input("💰 Капитал (₽)", value=100000, step=10000)

# ============================================================
# ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ ДЛЯ СОХРАНЕНИЯ
# ============================================================
if 'optimization_done' not in st.session_state:
    st.session_state.optimization_done = False
if 'best_params' not in st.session_state:
    st.session_state.best_params = None
if 'best_trial' not in st.session_state:
    st.session_state.best_trial = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'trades_prep' not in st.session_state:
    st.session_state.trades_prep = None
if 'ticker' not in st.session_state:
    st.session_state.ticker = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'train_ratio' not in st.session_state:
    st.session_state.train_ratio = None
if 'commission_pct' not in st.session_state:
    st.session_state.commission_pct = None
if 'study_best_value' not in st.session_state:
    st.session_state.study_best_value = None
if 'lot_size' not in st.session_state:
    st.session_state.lot_size = 1

# ============================================================
# ЗАГРУЗКА ДАННЫХ И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def get_available_dates(ticker: str) -> list:
    history_root = Path("data/for_ml/storage/history")
    dates = set()
    for subdir in ["realtime", "supercandles"]:
        base = history_root / subdir / ticker
        if base.exists():
            for d in base.iterdir():
                if d.is_dir():
                    dates.add(d.name)
    return sorted(dates)

def load_history(ticker: str) -> dict:
    history_root = Path("data/for_ml/storage/history")
    result = {"candles": [], "trades": [], "orderbook": [], "tradestats": [], "obstats": [], "orderstats": []}
    dates = get_available_dates(ticker)
    for date in dates:
        rt_dir = history_root / "realtime" / ticker / date
        for key in ["candles", "trades", "orderbook"]:
            filepath = rt_dir / f"{key}.parquet"
            if filepath.exists():
                try:
                    df = pd.read_parquet(filepath)
                    df["date"] = date
                    result[key].append(df)
                except Exception:
                    pass
        sc_dir = history_root / "supercandles" / ticker / date
        for key in ["tradestats", "obstats", "orderstats"]:
            filepath = sc_dir / f"{key}.parquet"
            if filepath.exists():
                try:
                    df = pd.read_parquet(filepath)
                    df["date"] = date
                    result[key].append(df)
                except Exception:
                    pass
    for key in result:
        if result[key]:
            result[key] = pd.concat(result[key], ignore_index=True)
        else:
            result[key] = pd.DataFrame()
    return result

def prepare_trades_for_bars(trades_df: pd.DataFrame) -> pd.DataFrame:
    df = trades_df.copy()
    col_map = {}
    for col in df.columns:
        col_upper = col.upper()
        if col_upper in ["PRICE", "TRADEPRICE"]:
            col_map[col] = "price"
        elif col_upper in ["QUANTITY", "VOLUME"]:
            col_map[col] = "quantity"
        elif col_upper == "VALUE":
            col_map[col] = "value"
    if col_map:
        df = df.rename(columns=col_map)
    required = ["price", "value", "ts"]
    if not all(c in df.columns for c in required):
        raise ValueError("Отсутствуют колонки price, value, ts")
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Europe/Moscow")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["price", "value", "ts"])
    df = df[df["price"] > 0]
    df["quantity"] = df["value"] / df["price"]
    return df[["ts", "price", "quantity"]]

def run_backtest(model, X_train, y_train, X_test, y_test, labeled_events, bar_close,
                 pt_sl, commission=0.0006, initial_capital=100000, lot_size=1):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds_orig = np.array([-1 if p == 0 else 1 for p in preds])
    
    capital = initial_capital
    equity = [capital]
    trades = []
    position = None
    
    for i, (evt_time, row) in enumerate(labeled_events.iterrows()):
        signal = preds_orig[i]
        if signal == 0:
            continue
        
        # Принудительное закрытие после 23:00
        if evt_time.hour >= 23 and position is not None:
            exit_price = row['exit_price']
            if position['side'] == 'long':
                pnl = (exit_price - position['entry']) * position['shares'] * lot_size
            else:
                pnl = (position['entry'] - exit_price) * position['shares'] * lot_size
            pnl -= commission * position['entry'] * position['shares'] * lot_size
            pnl -= commission * exit_price * position['shares'] * lot_size
            capital += pnl
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': evt_time,
                'side': position['side'],
                'entry': position['entry'],
                'exit': exit_price,
                'shares': position['shares'],
                'pnl': pnl,
                'exit_reason': 'forced_close'
            })
            equity.append(capital)
            position = None
            continue
        
        entry_price = bar_close.loc[evt_time]
        exit_price = row['exit_price']
        exit_type = row.get('exit_type', 'vertical')
        exit_time = row['exit_time']
        
        if position is None:
            shares = int(capital // (entry_price * lot_size))
            if shares == 0:
                continue
            capital -= commission * entry_price * shares * lot_size
            position = {
                'side': 'long' if signal == 1 else 'short',
                'entry': entry_price,
                'shares': shares,
                'entry_time': evt_time
            }
        else:
            # Закрываем позицию
            if position['side'] == 'long':
                pnl = (exit_price - position['entry']) * position['shares'] * lot_size
            else:
                pnl = (position['entry'] - exit_price) * position['shares'] * lot_size
            pnl -= commission * exit_price * position['shares'] * lot_size
            capital += pnl
            
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': exit_time,
                'side': position['side'],
                'entry': position['entry'],
                'exit': exit_price,
                'shares': position['shares'],
                'pnl': pnl,
                'exit_reason': exit_type
            })
            equity.append(capital)
            
            # Reverse
            if (position['side'] == 'long' and signal == -1) or (position['side'] == 'short' and signal == 1):
                shares = int(capital // (entry_price * lot_size))
                if shares == 0:
                    position = None
                    continue
                capital -= commission * entry_price * shares * lot_size
                position = {
                    'side': 'long' if signal == 1 else 'short',
                    'entry': entry_price,
                    'shares': shares,
                    'entry_time': evt_time
                }
            else:
                position = None
    
    # Закрываем последнюю позицию
    if position is not None:
        exit_price = labeled_events.iloc[-1]['exit_price']
        if position['side'] == 'long':
            pnl = (exit_price - position['entry']) * position['shares'] * lot_size
        else:
            pnl = (position['entry'] - exit_price) * position['shares'] * lot_size
        pnl -= commission * exit_price * position['shares'] * lot_size
        capital += pnl
        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': labeled_events.iloc[-1]['exit_time'],
            'side': position['side'],
            'entry': position['entry'],
            'exit': exit_price,
            'shares': position['shares'],
            'pnl': pnl,
            'exit_reason': 'last'
        })
        equity.append(capital)
    
    # Метрики
    total_return = (capital / initial_capital - 1) * 100
    returns = [t['pnl'] for t in trades]
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) != 0 else 0
    peak = np.maximum.accumulate(equity)
    dd = (np.array(equity) - peak) / peak
    max_dd = dd.min() * 100
    win_rate = (sum(1 for t in trades if t['pnl'] > 0) / len(trades)) * 100 if trades else 0
    profit_factor = (sum(t['pnl'] for t in trades if t['pnl'] > 0) / 
                     abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))) if any(t['pnl'] < 0 for t in trades) else np.inf
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'trades': trades,
        'equity_curve': equity
    }

# ============================================================
# ЗАПУСК ОПТИМИЗАЦИИ
# ============================================================
if st.button("🚀 ЗАПУСТИТЬ ОПТИМИЗАЦИЮ", use_container_width=True):
    status = st.empty()
    progress_bar = st.progress(0)
    lot_size = 1
    
    try:
        # 1. Загрузка данных
        status.markdown('<div class="status-text">📥 Загрузка исторических данных...</div>', unsafe_allow_html=True)
        data = load_history(ticker)
        if data["trades"].empty:
            st.error("Нет данных по сделкам. Сначала загрузите данные через Data Center.")
            st.stop()
        
        trades_prep = prepare_trades_for_bars(data["trades"])
        
        # Определяем размер лота из данных (берём медиану)
        trades_raw = data["trades"].copy()
        if "QUANTITY" in trades_raw.columns and "VALUE" in trades_raw.columns:
            trades_raw["price_x_qty"] = trades_raw["PRICE"] * trades_raw["QUANTITY"]
            lot_size_vals = trades_raw["VALUE"] / trades_raw["price_x_qty"]
            lot_size = lot_size_vals.replace([np.inf, -np.inf], np.nan).dropna().median()
            if np.isnan(lot_size) or lot_size <= 0:
                lot_size = 1
        st.session_state.lot_size = lot_size
        
        # 2. Функция objective для Optuna
        def objective(trial):
            dollar_threshold = trial.suggest_int("dollar_threshold", 3, 10) * 1_000_000
            pt = trial.suggest_float("pt", 0.3, 2.0, step=0.1)
            sl = trial.suggest_float("sl", 0.3, 2.0, step=0.1)
            cusum = trial.suggest_float("cusum", 0.001, 0.005, step=0.0005)
            vertical_barrier = trial.suggest_int("vertical_barrier", 10, 50, step=5)
            target_window = trial.suggest_int("target_window", 25, 75, step=5)
            
            try:
                bar_builder = InformationDrivenBars()
                dollar_bars = bar_builder.build_dollar(trades_prep, threshold=dollar_threshold)
                if len(dollar_bars) < 20:
                    return -np.inf
                dollar_bars["ts"] = dollar_bars["close_time"]
                dollar_bars["ts"] = pd.to_datetime(dollar_bars["ts"])
                if dollar_bars["ts"].dt.tz is None:
                    dollar_bars["ts"] = dollar_bars["ts"].dt.tz_localize("Europe/Moscow")
                dollar_bars = dollar_bars.set_index("ts").sort_index()
                dollar_bars = dollar_bars.rename(columns={"open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE", "volume": "VOLUME"})
                dollar_bars = dollar_bars[~dollar_bars.index.duplicated(keep="first")]
                dollar_bars = dollar_bars[~dollar_bars.index.isna()]
                
                events_idx = cusum_filter(dollar_bars["CLOSE"], threshold=cusum)
                if len(events_idx) < 5:
                    return -np.inf
                if dollar_bars.index.tz is not None:
                    if events_idx.tz is None:
                        events_idx = events_idx.tz_localize(dollar_bars.index.tz)
                    elif events_idx.tz != dollar_bars.index.tz:
                        events_idx = events_idx.tz_convert(dollar_bars.index.tz)
                else:
                    if events_idx.tz is not None:
                        events_idx = events_idx.tz_localize(None)
                events_idx = events_idx[~events_idx.duplicated()]
                pos = dollar_bars.index.get_indexer(events_idx, method="nearest")
                pos = pos[pos >= 0]
                events_idx = dollar_bars.index[pos]
                events_idx = events_idx[~events_idx.duplicated()].sort_values()
                if len(events_idx) < 5:
                    return -np.inf
                
                labeled = triple_barrier_labeling(
                    bars=dollar_bars,
                    events=events_idx,
                    pt_sl=(pt, sl),
                    vertical_barrier=vertical_barrier,
                    target_window=target_window,
                    target_method="ewm"
                )
                if len(labeled) < 30:
                    return -np.inf
                
                bars_lower = dollar_bars.rename(columns={"OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close", "VOLUME": "volume"})
                mf = MicroFeatures(spread_window=20, volatility_window=20, intensity_window=20)
                bars_with_micro = mf.build_all_features(bars_lower)
                fb = FeatureBuilder(windows=[5, 10, 20], lags=[1, 2, 3], include_micro=False)
                bars_featured = fb.build_all(bars_with_micro)
                numeric_cols = bars_featured.select_dtypes(include=[np.number]).columns
                bars_featured = bars_featured[numeric_cols]
                
                events_df = labeled.copy()
                bars_at_events = bars_featured.reindex(events_df.index)
                bars_at_events.columns = [f"bar_{col}" for col in bars_at_events.columns]
                events_df = pd.concat([events_df, bars_at_events], axis=1)
                events_df = events_df.dropna()
                if len(events_df) < 30:
                    return -np.inf
                
                mask_bin = events_df["label"] != 0
                feature_cols = [c for c in events_df.columns if c.startswith("bar_")]
                trial.set_user_attr("feature_cols", feature_cols)  # <--- сохраняем в trial
                
                X = events_df[feature_cols]
                y = events_df["label"]
                X_bin = X.loc[mask_bin]
                y_bin = y.loc[mask_bin].map({1: 1, -1: 0})
                if len(X_bin) < 30:
                    return -np.inf
                
                split_idx = int(len(X_bin) * train_ratio)
                X_train = X_bin.iloc[:split_idx]
                X_test = X_bin.iloc[split_idx:]
                y_train = y_bin.iloc[:split_idx]
                y_test = y_bin.iloc[split_idx:]
                if len(np.unique(y_train)) < 2:
                    return -np.inf
                
                if model_type == "RandomForest":
                    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=13, class_weight="balanced", n_jobs=-1)
                elif model_type == "XGBoost":
                    model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=13, eval_metric="logloss", verbosity=0)
                elif model_type == "LightGBM":
                    model = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=13, verbose=-1)
                elif model_type == "CatBoost":
                    model = CatBoostClassifier(iterations=200, depth=5, learning_rate=0.05, subsample=0.8, random_seed=13, verbose=0, allow_writing_files=False)
                else:
                    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=13, class_weight="balanced", n_jobs=-1)
                
                result = run_backtest(
                    model, X_train, y_train, X_test, y_test,
                    labeled_events=labeled.loc[X_test.index],
                    bar_close=dollar_bars["CLOSE"],
                    pt_sl=(pt, sl),
                    lot_size=lot_size,
                    commission=commission_pct,
                    initial_capital=initial_capital
                )
                
                sharpe = result["sharpe"]
                if np.isnan(sharpe) or not np.isfinite(sharpe):
                    sharpe = -np.inf
                trial.set_user_attr("total_return", result["total_return"])
                trial.set_user_attr("max_drawdown", result["max_drawdown"])
                trial.set_user_attr("win_rate", result["win_rate"])
                trial.set_user_attr("n_trades", len(result["trades"]))
                trial.set_user_attr("equity_curve", result["equity_curve"])
                if target_metric == "sharpe":
                    return sharpe
                elif target_metric == "return":
                    return result["total_return"]
                else:  # return_per_dd
                    dd = abs(result["max_drawdown"]) if result["max_drawdown"] != 0 else 0.001
                    return result["total_return"] / dd if dd > 0 else 0
            except Exception:
                return -np.inf
        
        # 3. Запуск Optuna
        status.markdown('<div class="status-text">⚙️ Запуск Optuna...</div>', unsafe_allow_html=True)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=13))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        progress_bar.progress(0)
        def callback(study, trial):
            progress = (trial.number + 1) / n_trials
            progress_bar.progress(min(progress, 1.0))
            if trial.value is not None and trial.value > -np.inf:
                status.markdown(f'<div class="status-text">⚙️ Итерация {trial.number + 1}/{n_trials} | Лучший Sharpe: {study.best_value:.4f}</div>', unsafe_allow_html=True)
        
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
        progress_bar.progress(1.0)
        
        if study.best_value > -np.inf:
            best_params = study.best_params
            best_trial = study.best_trial
            
            # Извлекаем список признаков из best_trial
            feature_cols = best_trial.user_attrs.get("feature_cols")
            if feature_cols is None:
                st.error("Не удалось получить список признаков.")
                st.stop()
            
            # Сохраняем в session_state
            st.session_state.optimization_done = True
            st.session_state.best_params = best_params
            st.session_state.best_trial = best_trial
            st.session_state.feature_cols = feature_cols
            st.session_state.trades_prep = trades_prep
            st.session_state.ticker = ticker
            st.session_state.model_type = model_type
            st.session_state.train_ratio = train_ratio
            st.session_state.commission_pct = commission_pct
            st.session_state.study_best_value = study.best_value
            st.session_state.lot_size = lot_size
            
            st.markdown("---")
            st.markdown('<h2 style="color:#ff4444;">✅ РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("💰 Dollar Threshold", f"{best_params['dollar_threshold']:,.0f}")
            col2.metric("📈 TP/SL", f"{best_params['pt']:.1f} / {best_params['sl']:.1f}")
            col3.metric("📊 CUSUM", f"{best_params['cusum']:.4f}")
            col4.metric("📐 Vertical Barrier", best_params['vertical_barrier'])
            col5.metric("🎯 Target Window", best_params['target_window'])
            
            st.metric("🏆 Лучшее значение метрики", f"{study.best_value:.4f}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("📈 Доходность (период)", f"{best_trial.user_attrs.get('total_return', 0):.2f}%")
            col2.metric("📊 Sharpe (годовой)", f"{study.best_value:.2f}")
            col3.metric("⬇️ Max DD", f"{best_trial.user_attrs.get('max_drawdown', 0):.2f}%")
            col4.metric("🎯 Win Rate", f"{best_trial.user_attrs.get('win_rate', 0):.1f}%")
            col5.metric("📊 Сделок", best_trial.user_attrs.get('n_trades', 0))
            
            # График equity curve
            equity_curve = best_trial.user_attrs.get("equity_curve")
            if equity_curve is not None and len(equity_curve) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(equity_curve))),
                    y=equity_curve,
                    mode='lines+markers',
                    name='Equity',
                    line=dict(color='#ff4444', width=2),
                    marker=dict(size=4)
                ))
                y_min = min(equity_curve) * 0.995
                y_max = max(equity_curve) * 1.005
                fig.update_layout(
                    template='plotly_dark',
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#c8c8c8'),
                    xaxis=dict(title='Номер сделки', gridcolor='#333'),
                    yaxis=dict(title='Капитал (₽)', range=[y_min, y_max], gridcolor='#333')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("График equity недоступен — недостаточно данных.")
        else:
            st.error("Оптимизация не нашла валидных параметров.")
    
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        import traceback
        st.code(traceback.format_exc())

# ============================================================
# СОХРАНЕНИЕ МОДЕЛИ (использует данные из session_state)
# ============================================================
if st.session_state.optimization_done and st.session_state.best_params is not None:
    st.markdown("---")
    if st.button("💾 СОХРАНИТЬ МОДЕЛЬ И ПАРАМЕТРЫ", key="save_model"):
        with st.spinner("Переобучение и сохранение..."):
            status_save = st.empty()
            status_save.markdown('<div class="status-text">💾 Переобучение модели на всех данных...</div>', unsafe_allow_html=True)
            
            ticker = st.session_state.ticker
            model_type = st.session_state.model_type
            train_ratio = st.session_state.train_ratio
            commission_pct = st.session_state.commission_pct
            trades_prep = st.session_state.trades_prep
            best_params = st.session_state.best_params
            best_trial = st.session_state.best_trial
            feature_cols = st.session_state.feature_cols
            lot_size = st.session_state.lot_size
            
            if feature_cols is None:
                st.error("Нет списка признаков. Повторите оптимизацию.")
                st.stop()
            
            dollar_threshold = best_params['dollar_threshold']
            pt = best_params['pt']
            sl = best_params['sl']
            cusum = best_params['cusum']
            vertical_barrier = best_params['vertical_barrier']
            target_window = best_params['target_window']
            
            bar_builder = InformationDrivenBars()
            dollar_bars = bar_builder.build_dollar(trades_prep, threshold=dollar_threshold)
            dollar_bars["ts"] = dollar_bars["close_time"]
            dollar_bars["ts"] = pd.to_datetime(dollar_bars["ts"])
            if dollar_bars["ts"].dt.tz is None:
                dollar_bars["ts"] = dollar_bars["ts"].dt.tz_localize("Europe/Moscow")
            dollar_bars = dollar_bars.set_index("ts").sort_index()
            dollar_bars = dollar_bars.rename(columns={"open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE", "volume": "VOLUME"})
            dollar_bars = dollar_bars[~dollar_bars.index.duplicated(keep="first")]
            dollar_bars = dollar_bars[~dollar_bars.index.isna()]
            
            events_idx = cusum_filter(dollar_bars["CLOSE"], threshold=cusum)
            if len(events_idx) > 0:
                if dollar_bars.index.tz is not None:
                    if events_idx.tz is None:
                        events_idx = events_idx.tz_localize(dollar_bars.index.tz)
                    elif events_idx.tz != dollar_bars.index.tz:
                        events_idx = events_idx.tz_convert(dollar_bars.index.tz)
                else:
                    if events_idx.tz is not None:
                        events_idx = events_idx.tz_localize(None)
                events_idx = events_idx[~events_idx.duplicated()]
                pos = dollar_bars.index.get_indexer(events_idx, method="nearest")
                pos = pos[pos >= 0]
                events_idx = dollar_bars.index[pos]
                events_idx = events_idx[~events_idx.duplicated()].sort_values()
            
            labeled = triple_barrier_labeling(
                bars=dollar_bars,
                events=events_idx,
                pt_sl=(pt, sl),
                vertical_barrier=vertical_barrier,
                target_window=target_window,
                target_method="ewm"
            )
            
            bars_lower = dollar_bars.rename(columns={"OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close", "VOLUME": "volume"})
            mf = MicroFeatures(spread_window=20, volatility_window=20, intensity_window=20)
            bars_with_micro = mf.build_all_features(bars_lower)
            fb = FeatureBuilder(windows=[5, 10, 20], lags=[1, 2, 3], include_micro=False)
            bars_featured = fb.build_all(bars_with_micro)
            numeric_cols = bars_featured.select_dtypes(include=[np.number]).columns
            bars_featured = bars_featured[numeric_cols]
            
            events_df = labeled.copy()
            bars_at_events = bars_featured.reindex(events_df.index)
            bars_at_events.columns = [f"bar_{col}" for col in bars_at_events.columns]
            events_df = pd.concat([events_df, bars_at_events], axis=1)
            events_df = events_df.dropna()
            
            mask_bin = events_df["label"] != 0
            X = events_df[feature_cols]
            y = events_df["label"]
            X_bin = X.loc[mask_bin]
            y_bin = y.loc[mask_bin].map({1: 1, -1: 0})
            
            if model_type == "RandomForest":
                final_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=13, class_weight="balanced", n_jobs=-1)
            elif model_type == "XGBoost":
                final_model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=13, eval_metric="logloss", verbosity=0)
            elif model_type == "LightGBM":
                final_model = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=13, verbose=-1)
            elif model_type == "CatBoost":
                final_model = CatBoostClassifier(iterations=200, depth=5, learning_rate=0.05, subsample=0.8, random_seed=13, verbose=0, allow_writing_files=False)
            else:
                final_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=13, class_weight="balanced", n_jobs=-1)
            
            final_model.fit(X_bin, y_bin)
            
            models_dir = Path("models/params")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            model_file = models_dir / f"{ticker}_{model_type.lower()}.pkl"
            joblib.dump(final_model, model_file)
            
            params = {
                "ticker": ticker,
                "model_type": model_type,
                "dollar_threshold": dollar_threshold,
                "pt_sl": [pt, sl],
                "cusum_threshold": cusum,
                "vertical_barrier": vertical_barrier,
                "target_window": target_window,
                "train_ratio": train_ratio,
                "commission": commission_pct,
                "lot_size": lot_size,
                "sharpe": st.session_state.study_best_value,
                "total_return": best_trial.user_attrs.get("total_return"),
                "max_drawdown": best_trial.user_attrs.get("max_drawdown"),
                "win_rate": best_trial.user_attrs.get("win_rate"),
                "n_trades": best_trial.user_attrs.get("n_trades"),
                "created_at": datetime.now().isoformat()
            }
            with open(models_dir / f"{ticker}_params.json", "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            
            with open(models_dir / f"{ticker}_features.pkl", "wb") as f:
                pickle.dump(feature_cols, f)
            
            status_save.markdown(f'<div class="status-text success">✅ Модель сохранена в <br><span class="save-path">{model_file}</span></div>', unsafe_allow_html=True)
            st.success(f"Модель и параметры сохранены в папку `models/params/` для тикера {ticker}")