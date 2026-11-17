# app/pages/data_center.py
import sys
from pathlib import Path
# Добавляем корень проекта и папку data в sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

from src.data.providers.algopack_provider import AlgoPackProvider
from src.data.builders.dataset_builder import DatasetBuilder

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="📡 DATA CENTER",
    page_icon="📡",
    layout="wide"
)

# ============================================================
# КИБЕРПАНК-СТИЛЬ
# ============================================================
st.markdown("""
<style>
    /* Основной фон */
    .main, .stApp {
        background-color: #0a0a0a;
        color: #d0d0d0;
    }

    /* Серебряный металлик для заголовков */
    .silver-glow {
        font-weight: 700;
        background: linear-gradient(90deg, #e8e8e8, #ffffff, #c0c0c0, #e8e8e8);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: silverShine 4s ease-in-out infinite;
        display: inline-block;
        text-shadow: 0 0 30px rgba(200,200,200,0.3);
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

    /* Сайдбар */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0000 0%, #0d0000 100%);
        border-right: 3px solid #8b0000;
        animation: neonPulse 2s infinite;
    }
    [data-testid="stSidebar"] * {
        color: #ffcccc !important;
        font-size: 1.05rem !important;
    }
    @keyframes neonPulse {
        0% { border-right-color: #8b0000; box-shadow: 0 0 5px #ff0000; }
        50% { border-right-color: #ff0000; box-shadow: 0 0 25px #ff0000; }
        100% { border-right-color: #8b0000; box-shadow: 0 0 5px #ff0000; }
    }

    /* Кнопки */
    .stButton>button {
        background: linear-gradient(135deg, #8b0000, #ff0000);
        color: #ffffff !important;
        border: 2px solid #ff4444;
        border-radius: 12px;
        font-weight: bold;
        font-size: 1.2rem !important;
        padding: 0.6rem 2rem !important;
        transition: 0.3s;
        box-shadow: 0 0 20px #ff0000, 0 0 40px rgba(255,0,0,0.3);
        width: 100%;
        height: auto !important;
        min-height: 55px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #a00000, #ff3333);
        box-shadow: 0 0 35px #ff4444, 0 0 70px rgba(255,0,0,0.5);
        transform: scale(1.01);
    }

    /* Поля ввода */
    .stTextInput>div>div, .stDateInput>div>div, .stSelectbox>div>div {
        background-color: #1a0a0a !important;
        border: 2px solid #8b0000 !important;
        border-radius: 10px !important;
        color: #ffcccc !important;
        font-size: 1.1rem !important;
    }
    .stTextInput input, .stDateInput input, .stSelectbox select {
        color: #ffcccc !important;
        font-size: 1.1rem !important;
        background-color: #1a0a0a !important;
    }
    .stSelectbox label, .stDateInput label, .stTextInput label {
        color: #ff8888 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    /* Метрики */
    [data-testid="stMetric"] {
        background: rgba(20,0,0,0.9);
        border: 2px solid #8b0000;
        border-radius: 14px;
        padding: 15px;
        box-shadow: 0 0 20px rgba(139,0,0,0.5), inset 0 0 30px rgba(139,0,0,0.1);
    }
    [data-testid="stMetric"] label {
        color: #ff8888 !important;
        font-size: 1rem !important;
        font-weight: 600;
    }
    [data-testid="stMetric"] div {
        color: #ffffff !important;
        font-size: 1.6rem !important;
        font-weight: 700;
        text-shadow: 0 0 10px #ff0000;
    }

    /* Прогресс-бар */
    .stProgress > div > div {
        background: linear-gradient(90deg, #8b0000, #ff0000, #ff4444) !important;
        box-shadow: 0 0 20px #ff0000 !important;
        height: 12px !important;
        border-radius: 8px !important;
    }
    .stProgress > div {
        background: #1a0a0a !important;
        border: 2px solid #8b0000 !important;
        border-radius: 10px !important;
        height: 16px !important;
    }

    /* Таблицы */
    .stDataFrame {
        background: #0a0a0a !important;
        border: 2px solid #8b0000 !important;
        border-radius: 12px !important;
    }
    .stDataFrame thead th {
        color: #ff8888 !important;
        font-size: 1rem !important;
        background: #1a0a0a !important;
        border-bottom: 2px solid #8b0000 !important;
    }
    .stDataFrame tbody td {
        color: #d0d0d0 !important;
        font-size: 1rem !important;
        border-bottom: 1px solid #2a0a0a !important;
    }

    /* Статус-текст */
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
    .status-text.success {
        color: #66ff88;
        border-color: #00ff44;
        box-shadow: 0 0 20px rgba(0,255,68,0.2);
    }
    .status-text.error {
        color: #ff4444;
        border-color: #ff0000;
        box-shadow: 0 0 20px rgba(255,0,0,0.3);
    }

    /* Checkbox */
    .stCheckbox label {
        color: #ffcccc !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    .stCheckbox label span {
        background-color: #1a0a0a !important;
        border: 2px solid #8b0000 !important;
        border-radius: 6px !important;
    }
    .stCheckbox label span svg {
        fill: #ff0000 !important;
    }

    /* Серебряный текст для обычных параграфов */
    p, li, .stMarkdown, .stText, .stAlert {
        color: #c8c8c8 !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
    .stAlert {
        background-color: #1a0a0a !important;
        border: 2px solid #8b0000 !important;
        color: #ffcccc !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
    }
    .stAlert > div {
        color: #ffcccc !important;
        font-size: 1.1rem !important;
    }

    /* Инфо-боксы */
    .info-box {
        background: rgba(20,0,0,0.7);
        border: 2px solid #8b0000;
        border-radius: 14px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 0 30px rgba(139,0,0,0.2);
    }
    .info-box .label {
        color: #8888ff;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .info-box .value {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(255,255,255,0.1);
    }
    .info-box .value.green { color: #66ff88; }
    .info-box .value.red { color: #ff4444; }
    .info-box .value.gold { color: #ffaa44; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ЗАГОЛОВОК
# ============================================================
st.markdown('<h1 class="silver-glow">📡 DATA CENTER</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#8888ff; font-size:1.2rem; margin-bottom:2rem;">⬡ ЗАГРУЗКА И КОНТРОЛЬ РЫНОЧНЫХ ДАННЫХ</p>', unsafe_allow_html=True)

# ============================================================
# ДОСТУПНЫЕ ТИКЕРЫ
# ============================================================
ALL_TICKERS = [
    'AFLT', 'ALRS', 'GAZP', 'LKOH', 'MAGN', 'MGNT',
    'MTLR', 'NVTK', 'ROSN', 'RUAL', 'SBER', 'SMLT',
    'SPBE', 'T', 'VTBR', 'YDEX'
]

# ============================================================
# ПАРАМЕТРЫ ЗАГРУЗКИ
# ============================================================
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    ticker = st.selectbox(
        "🎯 ТИКЕР",
        ALL_TICKERS,
        index=ALL_TICKERS.index("ROSN") if "ROSN" in ALL_TICKERS else 0,
        help="Выберите инструмент для загрузки"
    )

with col2:
    st.markdown("##### 📅 ДАТА")
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        date_start = st.date_input(
            "С",
            datetime.now().date() - timedelta(days=1),
            help="Начальная дата (включительно)"
        )
    with col_date2:
        date_end = st.date_input(
            "По",
            datetime.now().date() - timedelta(days=1),
            help="Конечная дата (включительно)"
        )

with col3:
    st.markdown("##### ⚡ РЕЖИМ")
    load_mode = st.radio(
        "Данные",
        ["ВСЕ", "СВЕЖИЕ"],
        index=0,
        help="ВСЕ — перезаписать всё за период, СВЕЖИЕ — только отсутствующие дни"
    )
    st.caption("⚠️ Данные всегда загружаются полностью")

# ============================================================
# ИНФОРМАЦИЯ О ЗАГРУЗКЕ
# ============================================================
dates = pd.date_range(date_start, date_end, freq='B')
total_days = len(dates)

col_info1, col_info2, col_info3, col_info4 = st.columns(4)

with col_info1:
    st.markdown(f"""
    <div class="info-box">
        <div class="label">📆 ДНЕЙ</div>
        <div class="value gold">{total_days}</div>
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    st.markdown(f"""
    <div class="info-box">
        <div class="label">🏷️ ТИКЕР</div>
        <div class="value gold">{ticker}</div>
    </div>
    """, unsafe_allow_html=True)

with col_info3:
    st.markdown(f"""
    <div class="info-box">
        <div class="label">📦 ДАННЫЕ</div>
        <div class="value gold">ВСЕ</div>
    </div>
    """, unsafe_allow_html=True)

with col_info4:
    st.markdown(f"""
    <div class="info-box">
        <div class="label">🔄 РЕЖИМ</div>
        <div class="value gold">{load_mode}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# КНОПКА ЗАГРУЗКИ
# ============================================================
st.markdown("---")

if st.button("⚡ ЗАГРУЗИТЬ ДАННЫЕ", use_container_width=True):
    if date_start > date_end:
        st.error("❌ Начальная дата не может быть позже конечной.")
        st.stop()

    # Инициализация
    provider = AlgoPackProvider()
    builder = DatasetBuilder(provider=provider, default_ticker=ticker)

    st.markdown(f"""
    <div class="status-text">
        ⚡ ЗАПУСК ЗАГРУЗКИ: {ticker} | {date_start.strftime('%Y-%m-%d')} — {date_end.strftime('%Y-%m-%d')} | {total_days} ДНЕЙ
    </div>
    """, unsafe_allow_html=True)

    # Прогресс
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.container()

    results = []
    errors = []

    for i, date_obj in enumerate(dates):
        date_str = date_obj.strftime("%Y-%m-%d")

        # Проверяем, нужно ли загружать (для режима СВЕЖИЕ)
        if load_mode == "СВЕЖИЕ":
            snap_path = builder._snapshot_path(ticker, date_str)
            if snap_path.exists():
                status_text.markdown(f"""
                <div class="status-text" style="border-color:#444;">
                    ⏭️ {date_str} — ПРОПУСК (уже есть)
                </div>
                """, unsafe_allow_html=True)
                results.append({"date": date_str, "status": "skipped"})
                progress_bar.progress((i + 1) / total_days)
                continue

        status_text.markdown(f"""
        <div class="status-text">
            🔄 {date_str} — ЗАГРУЗКА... ({i+1}/{total_days})
        </div>
        """, unsafe_allow_html=True)

        try:
            # Загружаем ВСЕ данные (свечи, сделки, стакан, суперсвечи)
            snap = builder.build_minute_snapshot(
                ticker,
                date_str,
                include_supercandles=True
            )

            # Сохраняем снапшот
            builder.save_snapshot(snap, ticker, date_str)

            # Проверяем, что сохранилось
            realtime_path = builder.history_path / "realtime" / ticker / date_str
            super_path = builder.history_path / "supercandles" / ticker / date_str

            status = {
                "date": date_str,
                "candles": (realtime_path / "candles.parquet").exists(),
                "trades": (realtime_path / "trades.parquet").exists(),
                "orderbook": (realtime_path / "orderbook.parquet").exists(),
                "tradestats": (super_path / "tradestats.parquet").exists(),
                "obstats": (super_path / "obstats.parquet").exists(),
                "orderstats": (super_path / "orderstats.parquet").exists(),
                "snapshot": builder._snapshot_path(ticker, date_str).exists(),
                "rows": len(snap),
                "status": "ok"
            }
            results.append(status)

            status_text.markdown(f"""
            <div class="status-text success">
                ✅ {date_str} — ГОТОВО ({len(snap)} строк)
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            error_msg = str(e)
            errors.append(f"{date_str}: {error_msg}")
            results.append({"date": date_str, "status": "error", "error": error_msg})

            status_text.markdown(f"""
            <div class="status-text error">
                ❌ {date_str} — ОШИБКА: {error_msg[:60]}...
            </div>
            """, unsafe_allow_html=True)

        progress_bar.progress((i + 1) / total_days)

    # ============================================================
    # ИТОГИ
    # ============================================================
    st.markdown("---")
    st.markdown('<h2 style="color:#ff4444;">📊 РЕЗУЛЬТАТЫ ЗАГРУЗКИ</h2>', unsafe_allow_html=True)

    # Статистика
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    error_count = sum(1 for r in results if r.get("status") == "error")
    skip_count = sum(1 for r in results if r.get("status") == "skipped")

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("✅ УСПЕШНО", f"{ok_count}/{total_days}", delta=f"{ok_count/total_days*100:.0f}%")
    col_r2.metric("❌ ОШИБОК", error_count, delta=f"{error_count/total_days*100:.0f}%" if total_days > 0 else "0")
    col_r3.metric("⏭️ ПРОПУЩЕНО", skip_count, delta=f"{skip_count/total_days*100:.0f}%" if total_days > 0 else "0")
    total_rows = sum(r.get("rows", 0) for r in results if isinstance(r.get("rows"), (int, float)))
    col_r4.metric("📊 ВСЕГО СТРОК", f"{total_rows:,}", delta="—")

    # Таблица результатов
    if results:
        df_results = pd.DataFrame(results)
        # Показываем только дату, статус и количество строк
        cols_to_show = ["date", "status", "rows"] + [c for c in ["candles", "trades", "orderbook", "tradestats", "obstats", "orderstats", "snapshot"] if c in df_results.columns]
        st.dataframe(
            df_results[cols_to_show],
            use_container_width=True,
            column_config={
                "date": "📅 Дата",
                "status": "🔄 Статус",
                "rows": "📊 Строк",
                "candles": "🕯️ Свечи",
                "trades": "📈 Сделки",
                "orderbook": "📚 Стакан",
                "tradestats": "📊 TradeStats",
                "obstats": "📊 OBStats",
                "orderstats": "📊 OrderStats",
                "snapshot": "💾 Снапшот"
            }
        )

    # Ошибки
    if errors:
        st.markdown("---")
        st.markdown('<h3 style="color:#ff4444;">⚠️ ОШИБКИ ЗАГРУЗКИ</h3>', unsafe_allow_html=True)
        for err in errors:
            st.error(f"❌ {err}")

    # ============================================================
    # УСПЕШНОЕ ЗАВЕРШЕНИЕ
    # ============================================================
    if ok_count == total_days and error_count == 0:
        st.markdown("""
        <div class="status-text success" style="text-align:center; font-size:1.4rem; padding:20px;">
            ✅ ВСЕ ДАННЫЕ ЗАГРУЖЕНЫ УСПЕШНО
        </div>
        """, unsafe_allow_html=True)
    elif ok_count > 0:
        st.markdown(f"""
        <div class="status-text" style="text-align:center; font-size:1.2rem; padding:15px; border-color:#ffaa44;">
            ⚡ ЗАГРУЖЕНО {ok_count} ИЗ {total_days} ДНЕЙ
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# САЙДБАР — БЫСТРЫЙ ДОСТУП
# ============================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("### ⚡ БЫСТРЫЙ СТАРТ")

    quick_tickers = ["ROSN", "GAZP", "RUAL", "MAGN", "SBER"]
    quick_ticker = st.selectbox("🎯 Быстрый тикер", quick_tickers, index=0)

    quick_days = st.slider("📅 Дней назад", 1, 30, 5)

    if st.button("🚀 ЗАГРУЗИТЬ БЫСТРО", use_container_width=True):
        st.info(f"Загрузка {quick_ticker} за последние {quick_days} дней...")

        # Переключаем селектор на быстрый тикер
        # (в реальности здесь нужно обновить параметры, но проще через session_state)
        st.session_state['quick_ticker'] = quick_ticker
        st.session_state['quick_days'] = quick_days
        st.rerun()

    st.markdown("---")
    st.markdown("### 📂 ЛОКАЛЬНЫЕ ДАННЫЕ")

    # Показываем последние снапшоты для выбранного тикера
    if st.button("🔍 ПРОВЕРИТЬ"):
        provider = AlgoPackProvider()
        builder = DatasetBuilder(provider=provider, default_ticker=ticker)

        today = datetime.now().date()
        found = []
        for i in range(1, 10):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            if builder._snapshot_path(ticker, d).exists():
                found.append(d)

        if found:
            st.success(f"✅ Найдено {len(found)} дней для {ticker}")
            for d in found[-5:]:
                st.caption(f"📁 {d}")
        else:
            st.info(f"📭 Нет данных для {ticker} за последние 10 дней")

    st.markdown("---")
    st.caption("⚡ JERICHON · DATA CENTER v1.0")