# app/dashboard.py
import streamlit as st

st.set_page_config(
    page_title="ИЕРЕХОН",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Киберпанк-стиль + Анимированный реактор ----------
st.markdown("""
<style>
    /* Основной фон */
    .main, .stApp {
        background-color: #000000;
        color: #e0e0e0;
    }

    /* Сайдбар */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0000 0%, #0d0000 100%);
        border-right: 2px solid #8b0000;
        animation: neonPulse 2s infinite;
    }
    [data-testid="stSidebar"] * {
        color: #ffcccc !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        border-right: 3px solid #8b0000;
        animation: neonPulse 2s infinite;
    }
    [data-testid="stSidebar"] .st-emotion-cache-1gwvyvv {
        gap: 0.2rem;
    }
    [data-testid="stSidebar"] .st-emotion-cache-1gwvyvv p {
        font-size: 1.1rem;
    }

    @keyframes neonPulse {
        0% { border-right-color: #8b0000; box-shadow: 0 0 5px #ff0000; }
        50% { border-right-color: #ff0000; box-shadow: 0 0 20px #ff0000; }
        100% { border-right-color: #8b0000; box-shadow: 0 0 5px #ff0000; }
    }

    /* Заголовки */
    h1, h2, h3, h4 {
        color: #ff4444;
        text-shadow: 0 0 8px #8b0000;
        font-family: 'Courier New', monospace;
    }

    /* Кнопки */
    .stButton>button {
        background: linear-gradient(135deg, #8b0000, #ff0000);
        color: white;
        border: 1px solid #ff4444;
        border-radius: 10px;
        box-shadow: 0 0 12px #ff0000;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #a00000, #ff3333);
        box-shadow: 0 0 20px #ff6666;
    }

    /* Поля ввода */
    .stTextInput>div>div, .stNumberInput>div>div, .stSelectbox>div>div {
        background-color: #1a0000;
        border: 2px solid #8b0000;
        border-radius: 8px;
        color: #ffcccc;
    }

    /* Карточки метрик */
    [data-testid="stMetric"] {
        background: rgba(20,0,0,0.8);
        border: 1px solid #8b0000;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 0 8px #8b0000;
    }
    [data-testid="stMetric"] label { color: #ff8888 !important; }
    [data-testid="stMetric"] div { color: #ffffff !important; }

    /* ---------- ЗАГОЛОВОК ---------- */
    .cyber-title {
        font-size: 5rem;
        font-weight: bold;
        color: #ff4444;
        text-shadow: 0 0 20px #ff0000, 0 0 40px #8b0000;
        font-family: 'Courier New', monospace;
        letter-spacing: 1.8rem;
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 0;
    }

    /* ---------- РЕАКТОР (ЯДРО) ---------- */
    .reactor-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 3rem 0;
        position: relative;
    }

    /* Внешнее кольцо */
    .reactor-outer {
        width: 220px;
        height: 220px;
        border-radius: 50%;
        background: conic-gradient(from 0deg, #ff0000, #8b0000, #ff4444, #ff0000);
        animation: rotateOuter 4s linear infinite;
        display: flex;
        justify-content: center;
        align-items: center;
        box-shadow: 0 0 60px #ff0000, 0 0 120px #8b0000;
    }

    @keyframes rotateOuter {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Среднее кольцо */
    .reactor-middle {
        width: 160px;
        height: 160px;
        border-radius: 50%;
        background: #0d0000;
        border: 2px solid #ff4444;
        display: flex;
        justify-content: center;
        align-items: center;
        animation: rotateMiddle 3s linear infinite reverse;
        box-shadow: inset 0 0 40px #ff0000, 0 0 30px #ff0000;
    }

    @keyframes rotateMiddle {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Внутреннее ядро */
    .reactor-core {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, #ff6666, #ff0000, #8b0000);
        animation: pulseCore 1.5s ease-in-out infinite;
        box-shadow: 0 0 80px #ff0000, 0 0 160px #ff0000;
    }

    @keyframes pulseCore {
        0%, 100% { transform: scale(1); box-shadow: 0 0 80px #ff0000, 0 0 160px #ff0000; }
        50% { transform: scale(1.15); box-shadow: 0 0 120px #ff4444, 0 0 200px #ff0000; }
    }

    /* Частицы вокруг реактора */
    .reactor-particles {
        position: absolute;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        border: 1px dashed #ff4444;
        animation: rotateOuter 8s linear infinite;
        opacity: 0.4;
    }

    /* ---------- МИНИ-ГРАФИК ПОД РЕАКТОРОМ ---------- */
    .mini-chart-container {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
    }
    .mini-chart {
        width: 350px;
        height: 80px;
        background: linear-gradient(90deg, #8b0000, #ff0000, #8b0000);
        border-radius: 8px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 20px #ff0000;
    }
    .mini-chart::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 0;
        width: 200%;
        height: 3px;
        background: repeating-linear-gradient(
            90deg,
            #ffffff 0px,
            #ffffff 4px,
            transparent 4px,
            transparent 8px
        );
        animation: dashFlow 2s linear infinite;
    }
    @keyframes dashFlow {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }

    /* Текст статуса */
    .status-text {
        text-align: center;
        color: #ff8888;
        font-family: 'Courier New', monospace;
        font-size: 1.2rem;
        margin-top: 1.5rem;
        text-shadow: 0 0 8px #8b0000;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Главный экран ----------
st.markdown("""
<div class="cyber-title">И Е Р Е Х О Н</div>
""", unsafe_allow_html=True)

# ---------- Реактор ----------
st.markdown("""
<div class="reactor-container">
    <div class="reactor-particles"></div>
    <div class="reactor-outer">
        <div class="reactor-middle">
            <div class="reactor-core"></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Мини-график ----------
st.markdown("""
<div class="mini-chart-container">
    <div class="mini-chart"></div>
</div>
""", unsafe_allow_html=True)

# ---------- Статус ----------
st.markdown("""
<div class="status-text">⚡ СИСТЕМА АКТИВНА — ВСЕ МОДУЛИ В РАБОТЕ ⚡</div>
""", unsafe_allow_html=True)

# Опциональное описание модулей
with st.expander("📡 Модули системы", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🛰️ Quant Lab** — обучение ML-стратегий")
        st.markdown("**📊 Фундаментальный анализ** — кредитоспособность компаний")
    with col2:
        st.markdown("**🧠 AI Hedge Fund** — управление стратегиями и портфелем")
        st.markdown("**⚙️ Настройки** — параметры подключения, конфигурация")