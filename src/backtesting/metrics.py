"""Small dependency-light performance metric set for JERICHON."""
from __future__ import annotations
import numpy as np
import pandas as pd


def calculate_metrics(equity: pd.Series, trades: pd.DataFrame, periods_per_year: float = 252.0) -> dict:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    pnl = pd.to_numeric(trades.get("net_pnl", pd.Series(dtype=float)), errors="coerce").dropna() if isinstance(trades, pd.DataFrame) else pd.Series(dtype=float)
    if eq.empty:
        return {"total_return": np.nan, "max_drawdown": np.nan, "sharpe": np.nan,
                "sortino": np.nan, "win_rate": np.nan, "profit_factor": np.nan,
                "expectancy": np.nan, "trades": int(len(pnl))}
    returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    mean = returns.mean() if len(returns) else np.nan
    std = returns.std(ddof=1) if len(returns) > 1 else np.nan
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    return {
        "max_drawdown": float(dd.min()) if len(dd) else np.nan,
        "sharpe": float(mean / std * np.sqrt(periods_per_year)) if np.isfinite(std) and std > 0 else np.nan,
        "sortino": float(mean / downside_std * np.sqrt(periods_per_year)) if np.isfinite(downside_std) and downside_std > 0 else np.nan,
        "win_rate": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "expectancy": float(pnl.mean()) if len(pnl) else np.nan,
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "trades": int(len(pnl)),
    }
