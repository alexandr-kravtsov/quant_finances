"""
JERICHON — End-to-End Pipeline Test (REAL DATA ONLY)
=====================================================
Полный тест пайплайна на реальных данных MOEX.
Токен читается из API.env (ALGOPACK_API_KEY).
Синтетики нет. Только живой рынок.
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Корень проекта для импортов
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.providers.algopack_provider import AlgoPackProvider
from src.data.builders.dataset_builder import DatasetBuilder
from src.data.builders.dollar_bars import InformationDrivenBars
from src.data.labeling.cusum import cusum_filter, adaptive_cusum_filter
from src.data.labeling.triple_barrier import triple_barrier_labeling

# ---------------------------------------------------------------------------
# НАСТРОЙКИ
# ---------------------------------------------------------------------------
TICKERS = ["AFLT", "ALRS", "GAZP", "GMKN", "LKOH", "MAGN", "MGNT", "MTLR", "NVTK", "PLZL",
           "ROSN", "RUAL", "SBER", "SMLT", "SPBE", "T", "TATN", "VTBR", "YDEX"]

DATE = datetime.now().strftime("%Y-%m-%d")

BAR_TYPE = "dollar"
BAR_THRESHOLD = 10_000_000

CUSUM_MODE = "fixed"
CUSUM_THRESHOLD = 0.002
CUSUM_MULTIPLIER = 3.0

PT_SL = (1.0, 1.0)
VERTICAL_BARRIER = 20
TARGET_WINDOW = 50

# ---------------------------------------------------------------------------
# ЗАПУСК
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("test_pipeline")

print("=" * 70)
print("JERICHON — End-to-End Pipeline Test (REAL DATA)")
print("=" * 70)
print(f"Тикеры: {', '.join(TICKERS)}")
print(f"Дата:   {DATE}")
print(f"Бары:   {BAR_TYPE}, порог {BAR_THRESHOLD:,}")
print(f"CUSUM:  {CUSUM_MODE}", end="")
if CUSUM_MODE == "fixed":
    print(f", порог {CUSUM_THRESHOLD}")
else:
    print(f", множитель {CUSUM_MULTIPLIER}")
print()

# ==================================================================
# 1. ПРОВАЙДЕР
# ==================================================================
print("▶ 1. Подключение к MOEX ISS...")
provider = AlgoPackProvider()   # если нужно отключить SSL, передай verify_ssl=False

if not provider.token:
    logger.error("ALGOPACK_API_KEY не найден в API.env")
    sys.exit(1)

print(f"   URL: {provider.base_url}")
print(f"   Токен: {'✔' if provider.token else '✘'}")

# ==================================================================
# 2. ОБРАБОТКА ВСЕХ ТИКЕРОВ
# ==================================================================
total_success = 0
total_errors = 0

for ticker in TICKERS:
    print(f"\n{'='*70}")
    print(f"ОБРАБОТКА {ticker}")
    print(f"{'='*70}")

    # --- Загрузка снапшота ---
    print(f"\n▶ 2. Загрузка снапшота {ticker} за {DATE}...")
    builder = DatasetBuilder(provider=provider, default_ticker=ticker)

    try:
        snap = builder.build_minute_snapshot(ticker, DATE, include_supercandles=False)
    except Exception as e:
        logger.error(f"Ошибка загрузки {ticker}: {e}")
        total_errors += 1
        continue

    if snap.empty:
        logger.warning(f"Пустой снапшот для {ticker} — пропускаем")
        total_errors += 1
        continue

    print(f"   Минут: {len(snap)}")
    print(f"   Диапазон: {snap.index[0]} — {snap.index[-1]}")

    # ========== НОВЫЙ БЛОК – диагностика и гибкий поиск колонок ==========
    print("   Колонки в снапшоте:", list(snap.columns))
    print("   Первые 3 строки:\n", snap.head(3))

    # Ищем колонку цены (приоритет: CLOSE > close > price > vwap)
    price_candidates = ["CLOSE", "close", "price", "vwap"]
    close_col = None
    for col in price_candidates:
        if col in snap.columns:
            close_col = col
            break

    # Ищем колонку объёма (приоритет: VOLUME > volume > trades_volume > value)
    volume_candidates = ["VOLUME", "volume", "trades_volume", "value"]
    vol_col = None
    for col in volume_candidates:
        if col in snap.columns:
            vol_col = col
            break

    if close_col is None or vol_col is None:
        logger.error(
            f"Не найдены колонки цены и/или объёма для {ticker}.\n"
            f"Доступные колонки: {list(snap.columns)}\n"
            f"Попробуйте добавить нужное имя в список candidates."
        )
        total_errors += 1
        continue

    print(f"   Использую цену: '{close_col}', объём: '{vol_col}'")
    # =====================================================================

    # --- Построение баров ---
    print(f"\n▶ 3. Построение {BAR_TYPE.upper()} баров...")

    trades_df = snap[[close_col, vol_col]].copy()
    trades_df.columns = ["price", "quantity"]
    trades_df = trades_df.dropna()
    trades_df = trades_df.reset_index()
    trades_df.rename(columns={trades_df.columns[0]: "ts"}, inplace=True)

    bar_builder = InformationDrivenBars()
    if BAR_TYPE == "dollar":
        bars = bar_builder.build_dollar(trades_df, threshold=BAR_THRESHOLD)
    elif BAR_TYPE == "volume":
        bars = bar_builder.build_volume(trades_df, threshold=BAR_THRESHOLD)
    elif BAR_TYPE == "tick":
        bars = bar_builder.build_tick(trades_df, threshold=BAR_THRESHOLD)
    else:
        logger.error(f"Неизвестный тип баров: {BAR_TYPE}")
        sys.exit(1)

    print(f"   Получено баров: {len(bars)}")

    if len(bars) < 5:
        logger.warning(f"Слишком мало баров для {ticker} — пропускаем")
        total_errors += 1
        continue

    bars["ts"] = bars["close_time"]
    bars = bars.set_index("ts").sort_index()
    bars_for_labeling = bars.rename(columns={
        "open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE"
    })

    # --- CUSUM ---
    print(f"\n▶ 4. CUSUM фильтр ({CUSUM_MODE})...")

    if CUSUM_MODE == "fixed":
        events = cusum_filter(bars["close"], threshold=CUSUM_THRESHOLD)
    elif CUSUM_MODE == "adaptive":
        events = adaptive_cusum_filter(
            bars["close"],
            multiplier=CUSUM_MULTIPLIER,
            volatility_window=TARGET_WINDOW,
        )
    else:
        sys.exit(1)

    print(f"   Найдено событий: {len(events)}")

    if len(events) == 0:
        logger.warning(f"Нет событий CUSUM для {ticker} — пропускаем")
        total_errors += 1
        continue

    # --- Triple Barrier ---
    print(f"\n▶ 5. Triple Barrier разметка...")

    labeled = triple_barrier_labeling(
        bars=bars_for_labeling,
        events=events,
        pt_sl=PT_SL,
        vertical_barrier=VERTICAL_BARRIER,
        target_window=TARGET_WINDOW,
        target_method="ewm",
    )

    print(f"   Размечено событий: {len(labeled)}")

    if labeled.empty:
        logger.warning(f"Разметка пуста для {ticker}")
        total_errors += 1
        continue

    label_dist = labeled["label"].value_counts()
    total = len(labeled)

    print("   Распределение меток:")
    for lbl, name, emoji in [(1, "UP (+1)", "🟢"), (-1, "DOWN (-1)", "🔴"), (0, "TIME (0)", "🟡")]:
        cnt = label_dist.get(lbl, 0)
        pct = 100 * cnt / total
        print(f"   {emoji} {name}: {cnt:4d} ({pct:5.1f}%)")

    # --- Валидация ---
    print("   Валидация:")

    checks = []

    c1 = (labeled["exit_time"] >= labeled.index).all()
    checks.append(("exit_time >= event_time", c1))

    c2 = (labeled["exit_time"] <= labeled["t1"]).all()
    checks.append(("exit_time <= t1", c2))

    up_ret = labeled[labeled["label"] == 1]["ret"]
    c3 = len(up_ret) == 0 or up_ret.mean() > 0
    checks.append(("средний ret UP > 0", c3))

    down_ret = labeled[labeled["label"] == -1]["ret"]
    c4 = len(down_ret) == 0 or down_ret.mean() < 0
    checks.append(("средний ret DOWN < 0", c4))

    c5 = labeled["label"].notna().all()
    checks.append(("нет NaN в метках", c5))

    for desc, result in checks:
        print(f"   {'✔' if result else '✘'} {desc}")

    passed = sum(1 for _, r in checks if r)
    total_checks = len(checks)

    if passed == total_checks:
        print(f"   ✓ {ticker}: ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
        total_success += 1
    else:
        print(f"   ✗ {ticker}: Пройдено {passed}/{total_checks} проверок")
        total_errors += 1

# ==================================================================
# ИТОГ
# ==================================================================
print("\n" + "=" * 70)
print("ИТОГОВЫЙ ОТЧЁТ")
print("=" * 70)
print(f"Всего тикеров: {len(TICKERS)}")
print(f"Успешно обработано: {total_success}")
print(f"Ошибок/пропусков: {total_errors}")
print("=" * 70)