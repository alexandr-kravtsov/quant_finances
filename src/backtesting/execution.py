"""Execution and position-sizing primitives for JERICHON."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ExecutionConfig:
    commission: float = 0.0006
    slippage: float = 0.0001


def execution_price(raw_price: float, side: int, action: str, slippage: float) -> float:
    p = float(raw_price)
    if not np.isfinite(p) or p <= 0:
        raise ValueError("Execution price must be finite and positive")
    if action not in {"buy", "sell"}:
        raise ValueError("action must be 'buy' or 'sell'")
    if slippage < 0:
        raise ValueError("slippage cannot be negative")
    return p * (1.0 + slippage) if action == "buy" else p * (1.0 - slippage)


def gross_pnl(side: int, qty: float, entry: float, exit_price: float) -> float:
    return float(qty) * (float(exit_price) - float(entry)) if int(side) == 1 else float(qty) * (float(entry) - float(exit_price))


def round_trip_fees(qty: float, entry: float, exit_price: float, commission: float) -> float:
    return (abs(float(qty) * float(entry)) + abs(float(qty) * float(exit_price))) * float(commission)


class FixedFractionSizer:
    """Position notional = equity * fraction; fraction is per entry, not total portfolio."""
    def __init__(self, fraction: float = 0.10):
        self.fraction = float(fraction)

    def size(self, equity: float, entry_price: float, side: int, stop_price: float | None = None) -> float:
        if not np.isfinite(equity) or equity <= 0 or not np.isfinite(entry_price) or entry_price <= 0:
            return 0.0
        if not np.isfinite(self.fraction) or self.fraction <= 0:
            return 0.0
        return float(equity) * self.fraction / float(entry_price)


class RiskFractionSizer:
    """Risk budget = equity * risk_fraction if stop is hit, before costs/gaps."""
    def __init__(self, risk_fraction: float = 0.0025):
        self.risk_fraction = float(risk_fraction)

    def size(self, equity: float, entry_price: float, side: int, stop_price: float | None = None) -> float:
        if not np.isfinite(equity) or equity <= 0 or not np.isfinite(entry_price) or entry_price <= 0:
            return 0.0
        if stop_price is None or not np.isfinite(stop_price):
            return 0.0
        stop_distance = abs(float(entry_price) - float(stop_price))
        if stop_distance <= 0 or not np.isfinite(self.risk_fraction) or self.risk_fraction <= 0:
            return 0.0
        return float(equity) * self.risk_fraction / stop_distance
