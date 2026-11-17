import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

# Импортируем наш анализатор
from src.agents.statement_analysis import CreditAnalyzer

# ---------- Настройка страницы ----------
st.set_page_config(page_title="Фундаментальный анализ", page_icon="📊", layout="wide")

# ---------- Киберпанк CSS ----------
st.markdown("""
<style>
    .main, .stApp { background-color: #000000; color: #e0e0e0; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a0000 0%, #0d0000 100%); border-right: 2px solid #8b0000; }
    [data-testid="stSidebar"] * { color: #ffcccc !important; }
    [data-testid="stSidebar"] > div:first-child { border-right: 3px solid #8b0000; animation: neonPulse 2s infinite; }
    @keyframes neonPulse { 
        0% { border-right-color: #8b0000; box-shadow: 0 0 5px #ff0000; } 
        50% { border-right-color: #ff0000; box-shadow: 0 0 20px #ff0000; } 
        100% { border-right-color: #8b0000; box-shadow: 0 0 5px #ff0000; } 
    }
    h1, h2, h3, h4 { color: #ff4444; text-shadow: 0 0 8px #8b0000; }
    .stButton>button { 
        background: linear-gradient(135deg, #8b0000, #ff0000); 
        color: white; border: 1px solid #ff4444; 
        border-radius: 10px; box-shadow: 0 0 12px #ff0000; 
        font-weight: bold; transition: 0.3s; 
    }
    .stButton>button:hover { 
        background: linear-gradient(135deg, #a00000, #ff3333); 
        box-shadow: 0 0 20px #ff6666; 
    }
    .stSelectbox>div>div, .stNumberInput>div>div { 
        background-color: #1a0000; border: 2px solid #8b0000; border-radius: 8px; color: #ffcccc; 
    }
    [data-testid="stMetric"] { 
        background: rgba(20,0,0,0.8); border: 1px solid #8b0000; 
        border-radius: 12px; padding: 10px; box-shadow: 0 0 8px #8b0000; 
    }
    [data-testid="stMetric"] label { color: #ff8888 !important; }
    [data-testid="stMetric"] div { color: #ffffff !important; }
    .signal-box { 
        background: rgba(20,0,0,0.9); border: 1px solid #ff4444; 
        border-radius: 10px; padding: 15px; margin: 10px 0; color: #ffcccc; 
    }
    .stExpander { 
        border: 1px solid #8b0000; border-radius: 8px; 
        background: rgba(20,0,0,0.5); 
    }
    .stExpander > div:first-child { background: #1a0000; color: #ff4444; }
    .stDataFrame { 
        border: 1px solid #8b0000; border-radius: 8px; 
        background: #0d0000; 
    }
    .stDataFrame th { background: #1a0000; color: #ff4444; }
    .stDataFrame td { color: #ffcccc; }
    .rating-A { color: #00ff88; text-shadow: 0 0 10px #00ff88; }
    .rating-B { color: #88ff00; text-shadow: 0 0 10px #88ff00; }
    .rating-C { color: #ffaa00; text-shadow: 0 0 10px #ffaa00; }
    .rating-D { color: #ff0000; text-shadow: 0 0 10px #ff0000; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Фундаментальный анализ")
st.markdown("Кредитоспособность и финансовые коэффициенты (Грэм/Баффет)")

# ---------- Загрузка базы данных ----------
DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "statement_database.json"

@st.cache_data
def load_database():
    if not DB_PATH.exists():
        st.error(f"База данных не найдена по пути: {DB_PATH}")
        return {"companies": {}}
    with open(DB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

data = load_database()
companies = data.get("companies", {})

if not companies:
    st.warning("Нет данных о компаниях. Загрузите отчётность через модуль сбора данных.")
    st.stop()

# ---------- Выбор компании и года ----------
tickers = list(companies.keys())
ticker = st.selectbox("Выберите компанию", tickers, key="ticker_select")

company = companies[ticker]
reports = company.get("reports", {})
years = sorted(reports.keys(), reverse=True)
if not years:
    st.warning(f"Нет отчётности для {ticker}")
    st.stop()

year = st.selectbox("Отчётный год", years, key="year_select")
report = reports[year]

# ---------- Преобразование данных для анализатора ----------
def map_to_analyzer_input(report: dict) -> dict:
    """
    Преобразует данные из формата JSON в словарь с русскими ключами,
    ожидаемый классом CreditAnalyzer.
    Все значения переводятся в ТЫСЯЧИ рублей (анализатор работает в тыс.)
    """
    income = report.get("income_statement", {})
    balance = report.get("balance_sheet", {})
    cashflow = report.get("cash_flow", {})

    def to_thousands(value):
        return value * 1000 if value is not None else 0.0

    # Вычисляем краткосрочные обязательства как разность между всеми обязательствами и долгосрочным долгом
    total_liabilities = balance.get("total_liabilities", 0)
    long_term_debt = balance.get("long_term_debt", 0)
    short_term_liabilities = max(0, total_liabilities - long_term_debt)

    return {
        "Выручка": to_thousands(income.get("revenue", 0)),
        "EBITDA": to_thousands(income.get("ebitda", 0)),
        "EBIT": to_thousands(income.get("operating_income", 0)),
        "Чистая прибыль": to_thousands(income.get("net_income", 0)),
        "Проценты к уплате": to_thousands(income.get("interest_expense", 0)),
        "Денежный поток от операционной деятельности": to_thousands(cashflow.get("operating_cash_flow", 0)),
        "Оборотные активы": to_thousands(balance.get("current_assets", 0)),
        "Краткосрочные обязательства": to_thousands(short_term_liabilities),
        "Денежные средства и эквиваленты": to_thousands(balance.get("cash", 0)),
        "Дебиторская задолженность": to_thousands(balance.get("receivables", 0)),
        "Собственный капитал": to_thousands(balance.get("total_equity", 0)),
        "Всего активов": to_thousands(balance.get("total_assets", 0)),
        "Долгосрочный долг": to_thousands(balance.get("long_term_debt", 0)),
        "Краткосрочный долг": to_thousands(balance.get("short_term_debt", 0)),
        "Выплаченные дивиденды": to_thousands(cashflow.get("dividends_paid", 0)),
    }

financials = map_to_analyzer_input(report)

# ---------- Определение сектора ----------
sector = company.get('sector', 'industrial')

# ---------- Запуск анализатора ----------
try:
    analyzer = CreditAnalyzer(config_dir="config")
    credit_report = analyzer.analyze(financials, sector=sector)
except FileNotFoundError as e:
    st.error(f"Ошибка: {e}")
    st.stop()
except Exception as e:
    st.error(f"Ошибка при анализе: {e}")
    st.stop()

# ---------- Отображение результатов ----------
st.markdown("### 📋 Результаты кредитного анализа")

col1, col2, col3 = st.columns(3)

# Рейтинг с цветом
rating_class = f"rating-{credit_report.rating}"
col1.metric("Рейтинг", f"{credit_report.rating}")

col2.metric("Итоговый балл", f"{credit_report.total_score:.1f} / 100")
col3.metric("Кол-во рисков", len(credit_report.risks))

# Вердикт
st.markdown(f"""
<div class="signal-box">
    <strong>💡 Вердикт:</strong> {credit_report.verdict}
</div>
""", unsafe_allow_html=True)

# ---------- Детали по метрикам ----------
with st.expander("📊 Детальные метрики и прохождение порогов", expanded=True):
    metrics_df = pd.DataFrame({
        "Метрика": list(credit_report.metrics.keys()),
        "Значение": list(credit_report.metrics.values()),
        "Пройдена": [credit_report.passed.get(k, False) for k in credit_report.metrics.keys()]
    })
    metrics_df["Значение"] = metrics_df["Значение"].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    metrics_df["Пройдена"] = metrics_df["Пройдена"].apply(lambda x: "✅" if x else "❌")
    st.dataframe(metrics_df, use_container_width=True)

# ---------- Детальные финансовые таблицы ----------
with st.expander("📋 Баланс (в млн руб)", expanded=False):
    bal_df = pd.DataFrame({
        "Показатель": ["Всего активов", "Оборотные активы", "Денежные средства", "Запасы", "Дебиторская задолженность",
                       "Собственный капитал", "Всего обязательств", "Краткосрочные займы", "Долгосрочные займы", "Общий долг"],
        "Значение": [
            report.get("balance_sheet", {}).get("total_assets", "—"),
            report.get("balance_sheet", {}).get("current_assets", "—"),
            report.get("balance_sheet", {}).get("cash", "—"),
            report.get("balance_sheet", {}).get("inventory", "—"),
            report.get("balance_sheet", {}).get("receivables", "—"),
            report.get("balance_sheet", {}).get("total_equity", "—"),
            report.get("balance_sheet", {}).get("total_liabilities", "—"),
            report.get("balance_sheet", {}).get("short_term_debt", "—"),
            report.get("balance_sheet", {}).get("long_term_debt", "—"),
            report.get("balance_sheet", {}).get("total_debt", "—")
        ]
    })
    bal_df["Значение"] = bal_df["Значение"].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
    st.dataframe(bal_df, use_container_width=True)

with st.expander("📈 Отчёт о прибылях и убытках (в млн руб)", expanded=False):
    inc_df = pd.DataFrame({
        "Показатель": ["Выручка", "Операционные расходы", "Операционная прибыль", "Финансовые доходы", 
                       "Проценты к уплате", "Налоги", "Чистая прибыль", "EPS (руб)"],
        "Значение": [
            report.get("income_statement", {}).get("revenue", "—"),
            report.get("income_statement", {}).get("operating_expenses", "—"),
            report.get("income_statement", {}).get("operating_income", "—"),
            report.get("income_statement", {}).get("financial_income", "—"),
            report.get("income_statement", {}).get("interest_expense", "—"),
            report.get("income_statement", {}).get("taxes", "—"),
            report.get("income_statement", {}).get("net_income", "—"),
            report.get("income_statement", {}).get("eps", "—")
        ]
    })
    inc_df["Значение"] = inc_df["Значение"].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) and x != "—" else x)
    st.dataframe(inc_df, use_container_width=True)

with st.expander("💸 Отчёт о движении денежных средств (в млн руб)", expanded=False):
    cf_df = pd.DataFrame({
        "Показатель": ["Операционный денежный поток", "Капитальные вложения", "Свободный денежный поток", "Дивиденды выплаченные"],
        "Значение": [
            report.get("cash_flow", {}).get("operating_cash_flow", "—"),
            report.get("cash_flow", {}).get("capex", "—"),
            report.get("cash_flow", {}).get("free_cash_flow", "—"),
            report.get("cash_flow", {}).get("dividends_paid", "—")
        ]
    })
    cf_df["Значение"] = cf_df["Значение"].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
    st.dataframe(cf_df, use_container_width=True)

# ---------- Графики динамики (если есть несколько лет) ----------
if len(reports) > 1:
    st.markdown("### 📈 Динамика ключевых показателей")
    
    years_ordered = sorted(reports.keys())
    years_labels = [str(y) for y in years_ordered]
    
    # Собираем данные для графиков
    revenues = []
    net_incomes = []
    roes = []
    debts = []
    for y in years_ordered:
        r = reports[y]
        revenues.append(r.get("income_statement", {}).get("revenue", 0))
        net_incomes.append(r.get("income_statement", {}).get("net_income", 0))
        roes.append(r.get("calculated_metrics", {}).get("roe", 0))
        debts.append(r.get("balance_sheet", {}).get("total_debt", 0))
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Выручка", "Чистая прибыль", "ROE (%)", "Общий долг"))
    fig.add_trace(go.Bar(x=years_labels, y=revenues, name="Выручка", marker_color="#ff4444"), row=1, col=1)
    fig.add_trace(go.Bar(x=years_labels, y=net_incomes, name="Чистая прибыль", marker_color="#ff8800"), row=1, col=2)
    fig.add_trace(go.Scatter(x=years_labels, y=roes, name="ROE", mode='lines+markers', line=dict(color='#00ff88')), row=2, col=1)
    fig.add_trace(go.Bar(x=years_labels, y=debts, name="Общий долг", marker_color="#ff00ff"), row=2, col=2)
    fig.update_layout(height=600, template='plotly_dark', paper_bgcolor='#000000', plot_bgcolor='#000000')
    fig.update_xaxes(title_text="Год", tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Сравнение с другими компаниями ----------
if len(companies) > 1:
    with st.expander("🔍 Сравнение с другими компаниями (текущий год)", expanded=False):
        current_year = year
        compare_data = []
        for t, comp in companies.items():
            rep = comp.get("reports", {}).get(current_year)
            if rep:
                m = rep.get("calculated_metrics", {})
                compare_data.append({
                    "Компания": comp.get("name", t),
                    "ROE (%)": m.get("roe"),
                    "ROA (%)": m.get("roa"),
                    "Долг/Капитал": m.get("debt_to_equity"),
                    "Чистая прибыль (млн ₽)": rep.get("income_statement", {}).get("net_income"),
                    "Активы (млн ₽)": rep.get("balance_sheet", {}).get("total_assets"),
                })
        if compare_data:
            df_comp = pd.DataFrame(compare_data)
            # Форматирование с разделителями тысяч и десятичными знаками
            styled = df_comp.style.format({
                'ROE (%)': '{:.1f}',
                'ROA (%)': '{:.1f}',
                'Долг/Капитал': '{:.2f}',
                'Чистая прибыль (млн ₽)': '{:,.0f}',
                'Активы (млн ₽)': '{:,.0f}'
            })
            st.dataframe(styled, use_container_width=True)
        else:
            st.info("Нет данных для сравнения за выбранный год.")

# ---------- Подвал ----------
st.caption(f"Данные обновлены: {report.get('updated_at', '—')} | Источник: МСФО/РСБУ отчётность")