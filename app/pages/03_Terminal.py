"""JERICHON Live Terminal — Stage 4D
Preserves the existing cyberpunk terminal UX while fixing:
- warmup/live temporal leakage
- Details auto-refresh
- stable Streamlit fragment rendering
- Moscow timezone handling at the UI boundary
- overlapping polling snapshots
- no timestamp de-duplication
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
for p in (ROOT, DATA_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.data.providers.algopack_provider import AlgoPackProvider
from live.live_pipeline import LiveStrategy
from live.live_engine import LiveEngine
from live import trading_controller

MSK = ZoneInfo("Europe/Moscow")
UI_REFRESH_SECONDS = 20
ENGINE_POLL_SECONDS = 2.0
RECENT_TAIL = 2_500

st.set_page_config(page_title="TERMINAL — Live Trading", page_icon="⚡", layout="wide")

# =========================================================================
# CYBERPUNK STYLES — kept intentionally close to the existing terminal
# =========================================================================
st.markdown(
    """
    <style>
    .main, .stApp { background-color: #0a0a0a; color: #d0d0d0; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0000 0%, #0d0000 100%) !important;
        border-right: 2px solid #8b0000 !important;
        animation: neonPulse 2s infinite !important;
    }
    [data-testid="stSidebar"] * { color: #ffcccc !important; }
    [data-testid="stSidebar"] > div:first-child {
        border-right: 3px solid #8b0000 !important;
    }
    @keyframes neonPulse {
        0% { border-right-color: #8b0000; box-shadow: 0 0 5px #ff0000; }
        50% { border-right-color: #ff0000; box-shadow: 0 0 20px #ff0000; }
        100% { border-right-color: #8b0000; box-shadow: 0 0 5px #ff0000; }
    }
    h1, h2, h3, h4 {
        color: #ff4444;
        text-shadow: 0 0 14px #8b0000;
        font-family: 'Courier New', monospace;
    }
    .stButton>button {
        background: linear-gradient(135deg, #8b0000, #ff0000);
        color: white;
        border: 1px solid #ff4444;
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #a00000, #ff3333);
        box-shadow: 0 0 20px #ff6666;
    }
    .stTextInput>div>div, .stNumberInput>div>div, .stSelectbox>div>div {
        background-color: #1a0000;
        border: 2px solid #8b0000;
        border-radius: 8px;
        color: #ffcccc;
    }
    .card {
        background: #100505;
        border: 1px solid #444;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .bull { border-color: #00ff88; box-shadow: 0 0 12px #00ff8844; }
    .bear { border-color: #ff0040; box-shadow: 0 0 12px #ff004044; }
    .ticker { font-size: 1.5rem; font-weight: 800; color: #ff4444; }
    .price { font-size: 1.15rem; color: #fff; }
    .buy { color: #00ff88; font-weight: 800; }
    .sell { color: #ff0040; font-weight: 800; }
    .flat { color: #888; font-weight: 800; }
    .pnl-pos { color: #00ff88; font-weight: bold; }
    .pnl-neg { color: #ff0040; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================================
# STATE
# =========================================================================
def _init_state() -> None:
    defaults = {
        "terminal_running": False,
        "strategies": {},
        "provider": None,
        "candles_data": {},
        "selected_tickers": [],
        "capital_inputs": {},
        "paused_tickers": set(),
        "all_trades_history": [],
        "current_view": "terminal",
        "detail_ticker": None,
        "last_update": None,
        "poll_count": 0,
        "errors": [],
        # Do NOT store st.empty()/DeltaGenerator objects in Session State.
        "live_update_meta": {},
        "automated_trading": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


def get_available_models() -> list[str]:
    models_dir = ROOT / "models" / "params"
    if not models_dir.exists():
        return []
    result = []
    for params in sorted(models_dir.glob("*_params.json")):
        ticker = params.stem.removesuffix("_params")
        features = models_dir / f"{ticker}_features.pkl"
        models = [
            p for p in models_dir.glob(f"{ticker}_*.pkl")
            if "_features" not in p.stem and "_params" not in p.stem
        ]
        if features.exists() and models:
            result.append(ticker)
    return result


def normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize trade columns without timestamp de-duplication."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    cmap = {str(c).lower(): c for c in out.columns}

    def copy(dst: str, names: tuple[str, ...]) -> None:
        if dst in out.columns:
            return
        for name in names:
            src = cmap.get(name.lower())
            if src is not None:
                out[dst] = out[src]
                return

    copy("trade_id", ("TRADENO", "tradeno", "trade_id"))
    copy("price", ("TRADEPRICE", "PRICE", "price"))
    copy("quantity", ("QUANTITY", "quantity", "size", "volume"))
    copy("value", ("VALUE", "value", "amount"))
    copy("ts", ("ts", "TRADETIME", "tradetime", "time"))
    copy("side", ("BUYSELL", "buysell", "side"))

    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    if getattr(out["ts"].dt, "tz", None) is None:
        out["ts"] = out["ts"].dt.tz_localize(MSK)
    else:
        out["ts"] = out["ts"].dt.tz_convert(MSK)

    for col in ("price", "quantity", "value"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    required = [c for c in ("ts", "price", "quantity", "value") if c in out.columns]
    return (
        out.dropna(subset=required)
        .sort_values("ts", kind="mergesort")
        .reset_index(drop=True)
    )


def _local_trade_dates(ticker: str) -> list[str]:
    base = DATA_ROOT / "for_ml" / "storage" / "history" / "realtime" / ticker
    if not base.exists():
        return []
    return sorted((p.name for p in base.iterdir() if p.is_dir()), reverse=True)


def load_local_warmup(
    provider: AlgoPackProvider,
    ticker: str,
    strategy: LiveStrategy,
) -> pd.DataFrame:
    frames = []

    for date in _local_trade_dates(ticker):
        path = (
            DATA_ROOT / "for_ml" / "storage" / "history" /
            "realtime" / ticker / date / "trades.parquet"
        )
        if not path.exists():
            continue

        try:
            df = normalize_trades(pd.read_parquet(path))
            if not df.empty:
                frames.append(df)
                candidate = pd.concat(frames, ignore_index=True).sort_values(
                    "ts", kind="mergesort"
                )
                bars_est = int(candidate["value"].sum() // strategy.dollar_threshold)
                if bars_est >= strategy.min_bars_for_features:
                    break
        except Exception as exc:
            st.session_state.errors.append(
                f"{ticker} local warmup {date}: {exc}"
            )

    if frames:
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values("ts", kind="mergesort")
            .reset_index(drop=True)
        )

    # API fallback is historical initialization only.
    today = datetime.now(MSK).date().isoformat()
    try:
        return normalize_trades(provider.fetch_trades(ticker, date=today))
    except Exception as exc:
        st.session_state.errors.append(f"{ticker} API warmup {today}: {exc}")
        return pd.DataFrame()


def _normalize_candles_for_ui(candles: pd.DataFrame) -> pd.DataFrame:
    if candles is None or candles.empty:
        return pd.DataFrame()

    c = candles.copy()
    if "ts" not in c.columns:
        return pd.DataFrame()

    c["ts"] = pd.to_datetime(c["ts"], errors="coerce")
    if getattr(c["ts"].dt, "tz", None) is None:
        c["ts"] = c["ts"].dt.tz_localize(MSK)
    else:
        c["ts"] = c["ts"].dt.tz_convert(MSK)

    rename = {
        k: k.lower()
        for k in ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME")
        if k in c.columns
    }
    c = c.rename(columns=rename)

    if "close" not in c.columns:
        return pd.DataFrame()

    return c.dropna(subset=["ts", "close"]).set_index("ts").sort_index(
        kind="mergesort"
    )


def _record_closed_trades(ticker: str, strategy: LiveStrategy, before: int) -> None:
    if len(strategy.trades) <= before:
        return
    for trade in strategy.trades[before:]:
        item = dict(trade)
        item["ticker"] = ticker
        st.session_state.all_trades_history.append(item)


def _engine_snapshot():
    return trading_controller.snapshot()


def _refresh_candles_for_ui(ticker: str) -> None:
    provider = st.session_state.provider
    if provider is None:
        return
    today = datetime.now(MSK).date().isoformat()
    yesterday = (datetime.now(MSK) - timedelta(days=1)).date().isoformat()
    try:
        candles = provider.fetch_candles(ticker, from_date=yesterday, till_date=today, interval=1)
        c = _normalize_candles_for_ui(candles)
        if not c.empty:
            st.session_state.candles_data[ticker] = c
    except Exception as exc:
        st.session_state.errors.append(f"{ticker} candles: {exc}")


def _price_for(ticker: str, strategy: LiveStrategy) -> float | None:
    # Execution state must use the latest trade price, not a potentially
    # delayed candle snapshot. This keeps Widget/Details PnL consistent.
    if getattr(strategy, "last_trade_price", None) is not None:
        return float(strategy.last_trade_price)
    candles = st.session_state.candles_data.get(ticker, pd.DataFrame())
    if not candles.empty and "close" in candles.columns:
        try:
            return float(candles["close"].iloc[-1])
        except Exception:
            pass
    return None


def _signal_text(sig: int) -> tuple[str, str]:
    if sig == 1:
        return "BUY", "buy"
    if sig == -1:
        return "SELL", "sell"
    return "FLAT", "flat"


def render_one_card(ticker: str) -> None:
    strategy = st.session_state.strategies[ticker]
    price = _price_for(ticker, strategy)
    status = strategy.get_status(price)

    sig = int(status["last_signal"])
    signal_text, signal_cls = _signal_text(sig)
    position = status["position"]
    paused = ticker in st.session_state.paused_tickers

    if position:
        pos_text = "LONG" if position["side"] == "long" else "SHORT"
    else:
        pos_text = "NONE"

    pnl = float(status["total_equity"] - strategy.initial_capital)
    pnl_pct = (
        pnl / strategy.initial_capital * 100
        if strategy.initial_capital else 0.0
    )
    pnl_cls = "pnl-pos" if pnl >= 0 else "pnl-neg"

    card_cls = "card"
    if position and position["side"] == "long":
        card_cls += " bull"
    elif position and position["side"] == "short":
        card_cls += " bear"
    elif sig == 1:
        card_cls += " bull"
    elif sig == -1:
        card_cls += " bear"

    price_text = f"{price:,.2f} ₽" if price is not None else "—"
    state_text = "⏸ ПАУЗА" if paused else "▶ АКТИВЕН"
    live_text = "LIVE" if strategy.live_started else "ARMING"

    st.markdown(
        f"""
        <div class="{card_cls}">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div class="ticker">{ticker}</div>
                <div style="font-size:0.8rem;color:#888;">
                    {state_text} · {live_text}
                </div>
            </div>
            <div class="price">{price_text}</div>
            <div class="{signal_cls}" style="font-size:1.2rem;margin:8px 0;">
                {signal_text}
            </div>
            <div style="font-size:0.9rem;color:#ccc;">
                Позиция: <b>{pos_text}</b><br>
                Лоты: <b>{int(position["lots"]) if position else 0}</b>
                | Акции: <b>{int(position.get("shares", 0)) if position else 0}</b><br>
                PnL:
                <span class="{pnl_cls}">{pnl_pct:+.2f}%</span>
                ({status["total_equity"]:,.0f} ₽)<br>
                Bars: {len(strategy.dollar_bars)}
                | CUSUM: {len(strategy.cusum_events)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2 = st.columns(2)
    with b1:
        label = "▶ Возобновить" if paused else "⏸ Остановить"
        if st.button(label, key=f"pause_{ticker}", width="stretch"):
            if paused:
                st.session_state.paused_tickers.discard(ticker)
            else:
                st.session_state.paused_tickers.add(ticker)
            st.rerun()

    with b2:
        if st.button("🔍 Детали", key=f"detail_{ticker}", width="stretch"):
            st.session_state.current_view = "details"
            st.session_state.detail_ticker = ticker
            st.rerun()


def render_details() -> None:
    strategies = st.session_state.strategies
    if not strategies:
        st.warning("Нет активных стратегий.")
        return

    if st.button("⬅ Назад к терминалу", key="back_terminal", width="stretch"):
        st.session_state.current_view = "terminal"
        st.rerun()

    tickers = list(strategies.keys())
    current = st.session_state.detail_ticker
    if current not in tickers:
        current = tickers[0]
        st.session_state.detail_ticker = current

    ticker = st.selectbox(
        "Инструмент",
        tickers,
        index=tickers.index(current),
        key="detail_ticker_selector",
    )
    st.session_state.detail_ticker = ticker

    strategy = strategies[ticker]
    _refresh_candles_for_ui(ticker)
    candles = st.session_state.candles_data.get(ticker, pd.DataFrame())
    price = _price_for(ticker, strategy)
    status = strategy.get_status(price)

    st.markdown(f"## 📊 Детали торговли: {ticker}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Капитал", f"{status['capital']:,.2f} ₽")
    c2.metric("📈 PnL", f"{status['unrealized_pnl']:+,.2f} ₽")
    c3.metric(
        "🎯 Позиция",
        (
            "LONG" if status["position"] and status["position"]["side"] == "long"
            else "SHORT" if status["position"] else "NONE"
        ),
    )
    c4.metric("⚡ Signal", "BUY" if status["last_signal"] == 1 else "SELL" if status["last_signal"] == -1 else "—")

    if candles.empty or "close" not in candles.columns:
        st.warning("Нет данных свечей для отображения графика.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=candles.index,
                y=candles["close"],
                mode="lines",
                name="Close",
                line=dict(color="#00ff88", width=2),
            )
        )

        closed = [
            t for t in st.session_state.all_trades_history
            if t.get("ticker") == ticker
        ]
        buys = [t for t in closed if t.get("side") == "long"]
        sells = [t for t in closed if t.get("side") == "short"]

        if buys:
            fig.add_trace(
                go.Scatter(
                    x=[t["entry_time"] for t in buys],
                    y=[t["entry_price"] for t in buys],
                    mode="markers",
                    name="Long Entry",
                    marker=dict(symbol="triangle-up", size=10, color="#00ff88"),
                )
            )
        if sells:
            fig.add_trace(
                go.Scatter(
                    x=[t["entry_time"] for t in sells],
                    y=[t["entry_price"] for t in sells],
                    mode="markers",
                    name="Short Entry",
                    marker=dict(symbol="triangle-down", size=10, color="#ff0040"),
                )
            )

        # Current open position is rendered separately from CLOSED history.
        # This removes the ambiguity where the last closed trade is SHORT while
        # the current live position is LONG (or vice versa).
        if strategy.position:
            pos = strategy.position
            marker_symbol = "triangle-up" if pos["side"] == "long" else "triangle-down"
            marker_name = f"OPEN {pos['side'].upper()}"
            fig.add_trace(
                go.Scatter(
                    x=[pos["entry_time"]],
                    y=[pos["entry_price"]],
                    mode="markers",
                    name=marker_name,
                    marker=dict(symbol=marker_symbol, size=14),
                )
            )
            pt_sl = strategy.params.get("pt_sl", 1.5)
            if isinstance(pt_sl, (list, tuple)) and len(pt_sl) == 2:
                tp_pct, sl_pct = float(pt_sl[0]), float(pt_sl[1])
            else:
                tp_pct = sl_pct = float(pt_sl)
            entry = float(pos["entry_price"])
            if pos["side"] == "long":
                sl = entry * (1 - sl_pct / 100)
                tp = entry * (1 + tp_pct / 100)
            else:
                sl = entry * (1 + sl_pct / 100)
                tp = entry * (1 - tp_pct / 100)

            x0 = pos["entry_time"]
            x1 = candles.index[-1]

            fig.add_shape(
                type="line", x0=x0, x1=x1, y0=sl, y1=sl,
                line=dict(color="red", dash="dash", width=2),
            )
            fig.add_shape(
                type="line", x0=x0, x1=x1, y0=tp, y1=tp,
                line=dict(color="#00ff88", dash="dash", width=2),
            )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            margin=dict(l=10, r=10, t=40, b=20),
            xaxis_title="Время (Europe/Moscow)",
            yaxis_title="Цена",
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)

        last_candle = candles.index[-1]
        st.caption(
            f"Свечей: {len(candles):,} · "
            f"Последняя свеча: {last_candle.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )

    st.markdown("### 📜 История сделок по тикеру")
    ticker_trades = [
        t for t in st.session_state.all_trades_history
        if t.get("ticker") == ticker
    ]
    if ticker_trades:
        st.dataframe(
            pd.DataFrame(ticker_trades).drop(columns=["ticker"], errors="ignore"),
            width="stretch",
            height=300,
        )
    else:
        st.info("Сделок пока не было.")


def render_diagnostics() -> None:
    """Compact execution-flow audit: proves where CUSUM events disappear."""
    strategies = st.session_state.strategies
    if not strategies:
        return
    st.markdown("### 🔬 Диагностика CUSUM → Prediction → Position")
    rows = []
    for ticker, strategy in strategies.items():
        d = strategy.get_debug_info()
        q = d.get("stats", {})
        preds = int(q.get("predictions", 0))
        events = int(q.get("cusum_events", 0))
        rows.append({
            "Ticker": ticker,
            "Trades accepted": int(q.get("trades_accepted", 0)),
            "Dup skipped": int(q.get("trades_duplicate", 0)),
            "Bars": int(q.get("bars_closed", 0)),
            "CUSUM": events,
            "Predictions": preds,
            "BUY": int(q.get("pred_long", 0)),
            "SELL": int(q.get("pred_short", 0)),
            "Same signal": int(q.get("same_signal", 0)),
            "Opens": int(q.get("opens", 0)),
            "Reversals": int(q.get("reversals", 0)),
            "TP/SL closes": int(q.get("barrier_closes", 0)),
            "Vertical closes": int(q.get("vertical_closes", 0)),
            "23:00 closes": int(q.get("forced_closes_23", 0)),
            "Blocked >=22:00": int(q.get("opens_blocked_after_22", 0)),
            "Closed trades": int(q.get("closed_trades", 0)),
            "Feature errors": int(q.get("feature_errors", 0)),
            "Prediction errors": int(q.get("prediction_errors", 0)),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    ticker = st.selectbox("Лог стратегии", list(strategies.keys()), key="diagnostic_ticker")
    strategy = strategies[ticker]
    d = strategy.get_debug_info()
    q = d.get("stats", {})
    events = int(q.get("cusum_events", 0))
    preds = int(q.get("predictions", 0))
    opens = int(q.get("opens", 0))
    closes = int(q.get("closed_trades", 0))
    st.caption(
        f"{ticker}: CUSUM={events} → predictions={preds} "
        f"({preds/events*100:.1f}% если events>0 else 0) → opens={opens} → closed={closes}"
        if events else f"{ticker}: CUSUM пока не генерировал событий"
    )
    st.json({
        "live_started": d.get("live_started"),
        "cutover_time": str(d.get("live_cutover_time")),
        "cutover_registered_snapshot": d.get("live_cutover_trade_count"),
        "last_trade_price": d.get("last_trade_price"),
        "last_processed_time": str(d.get("last_processed_time")),
        "last_event": str(d.get("last_event")),
        "last_prediction": d.get("last_prediction"),
        "position": d.get("position"),
        "stats": q,
        "last_compute_error": d.get("last_compute_error"),
        "last_compute_missing": d.get("last_compute_missing"),
    })
    st.code("\n".join(strategy.logs[-30:]) if strategy.logs else "Логов пока нет", language="text")


def render_terminal():
    strategies = st.session_state.strategies
    if not strategies:
        return
    snap = _engine_snapshot()
    st.markdown("### 🎛️ Панель управления")
    if st.button("⏹ ОСТАНОВИТЬ ТЕРМИНАЛ", key="stop_terminal", width="stretch"):
        trading_controller.stop_engine()
        try:
            if st.session_state.provider is not None:
                st.session_state.provider.close()
        except Exception:
            pass
        st.session_state.terminal_running = False
        st.session_state.strategies = {}
        st.session_state.provider = None
        st.session_state.paused_tickers = set()
        st.rerun()
    col1, col2, col3, col4 = st.columns([1, 1, 1.2, 2])
    with col1:
        st.metric("ENGINE", "RUNNING" if snap["running"] else "STOPPED")
    with col2:
        st.metric("BCS MODE", snap["execution_mode"].upper())
    with col3:
        enabled = bool(snap["automated_trading"])
        if st.button("🟢 AUTO ON" if enabled else "🔴 AUTO OFF", key="auto_trade_toggle", width="stretch"):
            if not enabled and snap["execution_mode"].lower() == "live" and any(s.position for s in st.session_state.strategies.values()):
                st.error("Сначала дождитесь/закройте текущие локальные позиции. AUTO LIVE нельзя включать при уже открытой позиции: broker и local state должны стартовать синхронно.")
            else:
                trading_controller.set_automated_trading(not enabled)
                st.rerun()
    with col4:
        last = snap.get("last_update")
        st.caption(f"Engine: {last.strftime('%H:%M:%S') if last else '—'} | poll #{snap.get('poll_count',0)} | UI {UI_REFRESH_SECONDS}s | engine {ENGINE_POLL_SECONDS}s")
    if snap["execution_mode"].lower() == "live" and snap["automated_trading"]:
        st.error("⚠️ LIVE BCS EXECUTION ENABLED — реальные заявки разрешены")
    elif snap["automated_trading"]:
        st.warning(f"AUTO ON — BCS mode: {snap['execution_mode'].upper()}")
    else:
        st.info(f"AUTO OFF — стратегия работает, но BCS заявки не отправляются. Mode: {snap['execution_mode'].upper()}")
    st.markdown("---")
    tickers = list(strategies.keys())
    cols = st.columns(min(3, len(tickers)))
    for i, ticker in enumerate(tickers):
        with cols[i % len(cols)]:
            render_one_card(ticker)
    with st.expander("🔬 ДИАГНОСТИКА", expanded=False):
        render_diagnostics()


# =========================================================================
# STARTUP / ARMING
# =========================================================================
st.markdown("# ⚡ TERMINAL — LIVE TRADING")
st.caption("ML / Dollar Bars / CUSUM / real-time demo execution")

if not st.session_state.terminal_running:
    st.markdown("### ⚙️ Настройка параметров торговли")

    models = get_available_models()
    if not models:
        st.error("Нет моделей в models/params/. Сначала обучите модели.")
        st.stop()

    selected = st.multiselect(
        "Выберите инструменты для торговли",
        models,
        default=st.session_state.selected_tickers or models[:1],
    )
    st.session_state.selected_tickers = selected

    capital_inputs = {}
    if selected:
        st.markdown("#### 💰 Распределение капитала")
        cols = st.columns(min(3, len(selected)))
        for i, ticker in enumerate(selected):
            with cols[i % len(cols)]:
                current = float(
                    st.session_state.capital_inputs.get(ticker, 50_000.0)
                )
                capital_inputs[ticker] = st.number_input(
                    f"Капитал для {ticker} (₽)",
                    min_value=1_000.0,
                    value=current,
                    step=5_000.0,
                    key=f"capital_{ticker}",
                )
    st.session_state.capital_inputs = capital_inputs

    if st.button(
        "🚀 ЗАПУСТИТЬ ТЕРМИНАЛ",
        width="stretch",
        type="primary",
    ):
        if not selected:
            st.error("Выберите хотя бы один тикер")
            st.stop()

        provider = AlgoPackProvider()
        strategies = {}
        progress = st.empty()
        models_dir = ROOT / "models" / "params"

        for i, ticker in enumerate(selected, start=1):
            progress.markdown(
                f"🔄 Инициализация **{ticker}** ({i}/{len(selected)}) — загрузка warmup..."
            )

            model_candidates = [
                p for p in models_dir.glob(f"{ticker}_*.pkl")
                if "_features" not in p.stem and "_params" not in p.stem
            ]
            if not model_candidates:
                raise FileNotFoundError(f"Не найдена модель для {ticker}")

            strategy = LiveStrategy(
                ticker=ticker,
                model_path=sorted(model_candidates)[0],
                params_path=models_dir / f"{ticker}_params.json",
                features_path=models_dir / f"{ticker}_features.pkl",
                initial_capital=st.session_state.capital_inputs.get(
                    ticker, 50_000.0
                ),
            )

            warmup = load_local_warmup(provider, ticker, strategy)
            strategy.warmup(warmup)

            # CRITICAL:
            # Take a fresh recent snapshot AFTER warmup. Register it as
            # already delivered, then arm live. It is NOT replayed.
            try:
                today = datetime.now(MSK).date().isoformat()
                cutover_snapshot = normalize_trades(
                    provider.fetch_recent_trades(
                        ticker, date=today, tail_size=RECENT_TAIL
                    )
                )
            except Exception as exc:
                provider.close()
                raise RuntimeError(
                    f"{ticker}: не удалось получить cutover snapshot: {exc}"
                ) from exc

            strategy.start_live(
                snapshot=(
                    cutover_snapshot.to_dict("records")
                    if not cutover_snapshot.empty else []
                ),
                cutover_time=datetime.now(MSK),
            )

            # Hard accounting contract after warmup.
            assert strategy.position is None
            assert strategy.capital == strategy.initial_capital
            assert strategy.trades == []

            strategies[ticker] = strategy

        progress.empty()
        engine = LiveEngine(provider, strategies, normalize_trades, tail_size=RECENT_TAIL, poll_seconds=ENGINE_POLL_SECONDS)
        engine.set_automated_trading(False)
        trading_controller.install_engine(engine)
        trading_controller.start_engine()

        st.session_state.provider = provider
        st.session_state.strategies = strategies
        st.session_state.terminal_running = True
        st.session_state.all_trades_history = []
        st.session_state.candles_data = {}
        st.session_state.live_update_meta = {}
        st.session_state.current_view = "terminal"
        st.session_state.detail_ticker = selected[0]
        st.rerun()

# =========================================================================
# SINGLE LIVE FRAGMENT — BOTH TERMINAL AND DETAILS
# =========================================================================
if st.session_state.terminal_running:

    @st.fragment(run_every=f"{UI_REFRESH_SECONDS}s")
    def live_fragment():
        # UI refresh only. LiveEngine continues independently between redraws.
        if st.session_state.current_view == "details":
            render_details()
        else:
            render_terminal()

    live_fragment()
