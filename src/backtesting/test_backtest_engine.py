"""Synthetic invariants for the canonical multi-position backtest."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from src.backtesting.backtest_engine import backtest, BacktestConfig


def make_bars():
    ts = pd.date_range("2026-01-01 10:00", periods=8, freq="min", tz="UTC")
    return pd.DataFrame({
        "bar_ts":ts,
        "open":[100,100,100,100,100,100,100,100],
        "high":[100.2,100.5,100.6,101.5,100.4,100.4,100.4,100.4],
        "low":[99.8,99.7,99.6,99.5,99.5,99.5,99.5,99.5],
        "close":[100,100,100,100,100,100,100,100],
    })


def test_same_direction_additions_and_leverage_cap():
    bars=make_bars()
    idx=pd.DatetimeIndex(bars.bar_ts)
    signals=pd.Series([1,1,1,0,0,0,0,0],index=idx)
    # wide barriers to keep positions open; labels are signal-time levels rebased to next open
    labels=pd.DataFrame({"upper":[110.0]*8,"lower":[90.0]*8,
                         "t1":[idx[-1]]*8},index=idx)
    cfg=BacktestConfig(barrier_mode="event_labels",sizing_mode="fixed_fraction",
                       position_fraction=0.5,max_gross_leverage=1.0,commission=0,slippage=0,
                       close_opposite_on_signal=False)
    result=backtest(bars,signals,config=cfg,event_labels=labels,initial_capital=1000)
    assert result.diagnostics["same_direction_additions"] >= 1
    assert result.diagnostics["max_gross_exposure"] <= 1000.0 + 1e-6
    assert result.diagnostics["same_bar_execution_violations"] == 0


def test_signal_executes_next_bar_open():
    bars=make_bars()
    idx=pd.DatetimeIndex(bars.bar_ts)
    signals=pd.Series([1,0,0,0,0,0,0,0],index=idx)
    labels=pd.DataFrame({"upper":[110.0]*8,"lower":[90.0]*8,
                         "t1":[idx[-1]]*8},index=idx)
    cfg=BacktestConfig(barrier_mode="event_labels",sizing_mode="fixed_fraction",
                       position_fraction=0.1,max_gross_leverage=3,commission=0,slippage=0,
                       close_opposite_on_signal=False)
    result=backtest(bars,signals,config=cfg,event_labels=labels,initial_capital=1000)
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["signal_time"] == idx[0]
    assert result.trades.iloc[0]["entry_time"] == idx[1]


if __name__ == "__main__":
    test_same_direction_additions_and_leverage_cap()
    test_signal_executes_next_bar_open()
    print("PASS: synthetic backtest invariants")
