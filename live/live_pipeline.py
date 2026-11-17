"""JERICHON live inference pipeline.

Causal path:
    raw trades -> dollar bars -> CUSUM -> features -> model -> position

Design rules:
- TRADENO is the trade identity when available.
- Timestamp is NEVER a deduplication key.
- Multiple trades and multiple dollar bars may share one timestamp.
- Warmup is state initialization, not model training. There is no fixed
  "2 trading days" requirement; the caller supplies enough local history to
  make the feature state valid.
- Warmup never opens/closes positions.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

try:
    from src.data.builders.micro_features import MicroFeatures
    from src.data.builders.build_features import FeatureBuilder
except ModuleNotFoundError:
    from src.data.builders.micro_features import MicroFeatures
    from src.data.builders.build_features import FeatureBuilder

logger = logging.getLogger("jerichon.live_pipeline")
logger.addHandler(logging.NullHandler())


class CUSUMState:
    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.last_event_time: Optional[pd.Timestamp] = None

    def update(self, ret: float, timestamp: pd.Timestamp) -> bool:
        self.s_pos = max(0.0, self.s_pos + float(ret))
        self.s_neg = min(0.0, self.s_neg + float(ret))
        if self.s_pos > self.threshold or self.s_neg < -self.threshold:
            self.s_pos = 0.0
            self.s_neg = 0.0
            self.last_event_time = timestamp
            return True
        return False


class LiveStrategy:
    def __init__(self, ticker: str, model_path: Path, params_path: Path,
                 features_path: Path, initial_capital: float = 100_000.0,
                 commission: Optional[float] = None) -> None:
        self.ticker = ticker
        self.initial_capital = float(initial_capital)
        self.capital = float(initial_capital)

        self.model = joblib.load(model_path)
        self.params = self._load_params(params_path)
        self.feature_cols = self._load_features(features_path)

        self.dollar_threshold = float(self.params.get("dollar_threshold", 10_000_000))
        self.cusum_threshold = float(self.params.get("cusum_threshold", 0.002))
        self.lot_size = float(self.params.get("lot_size", 1.0))
        self.commission = float(self.params.get("commission", 0.0006) if commission is None else commission)
        self.min_bars_for_features = int(self.params.get("min_bars_for_features", 50))

        micro_cfg = self.params.get("micro_features", {})
        feature_cfg = self.params.get("features", {})
        self.micro_builder = MicroFeatures(
            spread_window=int(micro_cfg.get("spread_window", 20)),
            volatility_window=int(micro_cfg.get("volatility_window", 20)),
            intensity_window=int(micro_cfg.get("intensity_window", 20)),
        )
        self.feature_builder = FeatureBuilder(
            windows=feature_cfg.get("windows", [5, 10, 20]),
            lags=feature_cfg.get("lags", [1, 2, 3]),
            include_micro=bool(feature_cfg.get("include_micro", False)),
        )

        # Current incomplete bar.
        self._bar_buffer: List[Dict[str, Any]] = []
        self._bar_value = 0.0
        self.dollar_bars: List[Dict[str, Any]] = []

        self.cusum_state = CUSUMState(self.cusum_threshold)
        self.cusum_events: List[pd.Timestamp] = []
        # Event identity is the bar sequence number, NOT timestamp.
        self._event_bar_numbers: set[int] = set()

        # Delivery de-duplication only. TRADENO identifies a trade; timestamp does not.
        self._seen_trade_ids: set[str] = set()
        self._seen_fallback_keys: set[str] = set()

        self.last_processed_time: Optional[pd.Timestamp] = None
        self.last_input_ts: Optional[pd.Timestamp] = None
        self.last_signal = 0
        self.last_signal_time: Optional[pd.Timestamp] = None
        self.last_compute_error: Optional[str] = None
        self.last_compute_missing: List[str] = []
        self.logs: List[str] = []
        self.warmup_done = False
        self.live_started = False
        self.live_cutover_time: Optional[pd.Timestamp] = None
        self.live_cutover_trade_count = 0

        # Full execution-flow audit counters. These are intentionally kept
        # separate from the trading state so we can prove where events are
        # being lost between CUSUM -> prediction -> order/position.
        self.stats: Dict[str, Any] = {
            "trades_seen": 0,
            "trades_accepted": 0,
            "trades_duplicate": 0,
            "invalid_trades": 0,
            "bars_closed": 0,
            "cusum_events": 0,
            "predictions": 0,
            "pred_long": 0,
            "pred_short": 0,
            "same_signal": 0,
            "opens": 0,
            "reversals": 0,
            "barrier_closes": 0,
            "vertical_closes": 0,
            "forced_closes_23": 0,
            "opens_blocked_after_22": 0,
            "reverse_closes": 0,
            "closed_trades": 0,
            "feature_errors": 0,
            "prediction_errors": 0,
            "open_skipped_insufficient_capital": 0,
        }
        self.last_prediction: Optional[Dict[str, Any]] = None
        self.last_trade_price: Optional[float] = None

        self.position: Optional[Dict[str, Any]] = None
        self.trades: List[Dict[str, Any]] = []

        # Explicit warmup -> live boundary. Warmup builds state only.
        self.live_started = False
        self.live_cutover_time: Optional[pd.Timestamp] = None
        self.live_cutover_trade_count = 0

    @staticmethod
    def _load_params(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_features(path: Path) -> List[str]:
        with open(path, "rb") as f:
            return list(pickle.load(f))

    @staticmethod
    def _ts(value: Any) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("Europe/Moscow")
        return ts.tz_convert("Europe/Moscow")

    @classmethod
    def _normalize_trade(cls, row: Any) -> Dict[str, Any]:
        r = row if isinstance(row, dict) else row.to_dict()
        ts = cls._ts(r["ts"])
        price = float(r.get("price", r.get("TRADEPRICE")))
        quantity = float(r.get("quantity", r.get("QUANTITY", r.get("size", 0))))
        raw_value = r.get("value", r.get("VALUE"))
        value = float(raw_value) if raw_value is not None and pd.notna(raw_value) else price * quantity
        trade_id = r.get("trade_id", r.get("TRADENO", r.get("tradeno")))
        side = r.get("side", r.get("BUYSELL", ""))
        return {"ts": ts, "price": price, "quantity": quantity, "value": value,
                "trade_id": trade_id, "side": side}

    @staticmethod
    def _fallback_trade_key(t: Dict[str, Any]) -> str:
        return "|".join([
            pd.Timestamp(t["ts"]).isoformat(),
            f"{float(t['price']):.12g}", f"{float(t['quantity']):.12g}",
            f"{float(t['value']):.12g}", str(t.get("side", "")),
        ])

    def _is_seen_delivery(self, t: Dict[str, Any]) -> bool:
        tid = t.get("trade_id")
        if tid is not None and str(tid).strip() not in {"", "nan", "None"}:
            return str(tid) in self._seen_trade_ids
        return self._fallback_trade_key(t) in self._seen_fallback_keys

    def _remember_delivery(self, t: Dict[str, Any]) -> None:
        tid = t.get("trade_id")
        if tid is not None and str(tid).strip() not in {"", "nan", "None"}:
            self._seen_trade_ids.add(str(tid))
        else:
            self._seen_fallback_keys.add(self._fallback_trade_key(t))

    def warmup(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Initialize causal state from history. Never trades."""
        if trades_df is None or trades_df.empty:
            self._log("warmup: пустой поток сделок")
            self.warmup_done = False
            return {"trades": 0, "bars": 0, "ready": False}

        df = trades_df.copy()
        if "ts" not in df.columns:
            raise ValueError("warmup: отсутствует колонка ts")
        df["ts"] = pd.to_datetime(df["ts"])
        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize("Europe/Moscow")
        else:
            df["ts"] = df["ts"].dt.tz_convert("Europe/Moscow")
        # Stable sort: equal timestamps retain exchange/API order.
        df = df.sort_values("ts", kind="mergesort")

        accepted = 0
        for _, row in df.iterrows():
            t = self._normalize_trade(row)
            if self._is_seen_delivery(t):
                continue
            self._remember_delivery(t)
            self._process_trade_internal(t, warmup=True)
            accepted += 1

        self.last_processed_time = df["ts"].max()
        self.last_input_ts = self.last_processed_time
        self.warmup_done = len(self.dollar_bars) >= self.min_bars_for_features
        self._log(f"warmup: input={len(df):,}, accepted={accepted:,}, bars={len(self.dollar_bars)}, ready={self.warmup_done}")
        return {"trades": accepted, "bars": len(self.dollar_bars), "ready": self.warmup_done}

    def start_live(self, snapshot: Optional[Iterable[Any]] = None,
                   cutover_time: Optional[Any] = None) -> Dict[str, Any]:
        """Arm live mode and register the current snapshot without replaying it."""
        self.live_cutover_time = (
            self._ts(cutover_time)
            if cutover_time is not None
            else pd.Timestamp.now(tz="Europe/Moscow")
        )
        registered = 0
        if snapshot is not None:
            for row in snapshot:
                t = self._normalize_trade(row)
                if not self._is_seen_delivery(t):
                    self._remember_delivery(t)
                    registered += 1

        # Warmup state (bars/CUSUM/features) is preserved; trading accounting is reset.
        self.position = None
        self.capital = self.initial_capital
        self.trades = []
        self.last_signal = 0
        self.last_signal_time = None
        self.live_started = True
        self.live_cutover_trade_count = registered
        self._log(f"LIVE START cutover={self.live_cutover_time} registered_snapshot={registered}")
        return {
            "live_started": True,
            "cutover_time": self.live_cutover_time,
            "registered_snapshot": registered,
        }

    def process_trade(self, trade: Dict[str, Any]) -> bool:
        if not self.live_started:
            raise RuntimeError(f"{self.ticker}: process_trade() called before start_live()")
        t = self._normalize_trade(trade)
        if self._is_seen_delivery(t):
            self.stats["trades_duplicate"] += 1
            return False
        if self.live_cutover_time is not None and t["ts"] <= self.live_cutover_time:
            # The polling API returns overlapping snapshots. Anything at or
            # before cutover is historical, even if it was not present in the
            # registration snapshot. It must never mutate live trading state.
            self._remember_delivery(t)
            self.stats["trades_duplicate"] += 1
            return False
        self._remember_delivery(t)
        self.stats["trades_seen"] += 1
        self.stats["trades_accepted"] += 1
        self.last_processed_time = t["ts"] if self.last_processed_time is None else max(self.last_processed_time, t["ts"])
        self.last_input_ts = t["ts"]
        self.last_trade_price = float(t["price"])

        # Battle-mode protection: if liquidity is so low that no new dollar
        # bar closes after 23:00, still force-close on the first actual trade
        # received at/after 23:00. With no market trade at all, no execution
        # price exists, so the position necessarily remains pending.
        if self.position is not None and t["ts"].time() >= pd.Timestamp("23:00:00").time():
            self.stats["forced_closes_23"] += 1
            self._close_position(float(t["price"]), t["ts"], reason="forced_close_23")

        self._process_trade_internal(t, warmup=False)
        return True

    def process_trades(self, trades: Iterable[Dict[str, Any]]) -> int:
        accepted = 0
        for trade in trades:
            accepted += int(self.process_trade(trade))
        return accepted

    def _process_trade_internal(self, trade: Dict[str, Any], warmup: bool) -> None:
        price = float(trade["price"])
        value = float(trade["value"])
        if not np.isfinite(price) or not np.isfinite(value) or price <= 0 or value <= 0:
            self.stats["invalid_trades"] += 1
            return
        self._bar_buffer.append(trade)
        self._bar_value += value
        if self._bar_value >= self.dollar_threshold:
            bar = self._close_current_bar()
            self._on_dollar_bar_closed(bar, warmup)

    def _close_current_bar(self) -> Dict[str, Any]:
        # Do NOT sort here. For equal timestamps the stable input order is meaningful.
        df = pd.DataFrame(self._bar_buffer)
        bar = {
            "open": float(df.iloc[0]["price"]),
            "high": float(df["price"].max()),
            "low": float(df["price"].min()),
            "close": float(df.iloc[-1]["price"]),
            "volume": float((df["value"] / df["price"]).sum()),
            "value": float(df["value"].sum()),
            "count": int(len(df)),
            "open_time": df.iloc[0]["ts"],
            "close_time": df.iloc[-1]["ts"],
        }
        self._bar_buffer = []
        self._bar_value = 0.0
        return bar

    def _on_dollar_bar_closed(self, bar: Dict[str, Any], warmup: bool) -> None:
        self.dollar_bars.append(bar)
        self.stats["bars_closed"] += 1
        n = len(self.dollar_bars)
        if n < 2:
            return
        prev = float(self.dollar_bars[-2]["close"])
        curr = float(bar["close"])
        if prev <= 0 or curr <= 0:
            return
        ret = float(np.log(curr / prev))
        event_ts = self._ts(bar["close_time"])

        # 1. Всегда обновляем CUSUM (независимо от позиции)
        cusum_fired = self.cusum_state.update(ret, event_ts)

        if cusum_fired:
            bar_number = n - 1
            if bar_number not in self._event_bar_numbers:
                self._event_bar_numbers.add(bar_number)
                self.cusum_events.append(event_ts)
                self.stats["cusum_events"] += 1
                self.last_signal_time = event_ts
                self._log(f"CUSUM event bar={bar_number} ts={event_ts}")

        # 2. Live-only exits. Protection order:
        #    23:00 forced close -> TP/SL -> vertical barrier -> prediction.
        if not warmup and self.warmup_done and self.position is not None:
            if event_ts.time() >= pd.Timestamp("23:00:00").time():
                self.stats["forced_closes_23"] += 1
                self._close_position(curr, event_ts, reason="forced_close_23")
                return

            self._check_barriers(bar)
            if self.position is None:
                return

            if self._check_vertical_barrier(n - 1, event_ts, curr):
                return

        # 3. If CUSUM fired, predict a signal. Opening is separately blocked
        #    from 22:00 onward, while existing positions may still be closed.
        if cusum_fired and not warmup and self.warmup_done:
            features = self._compute_features()
            if features is None:
                return
            try:
                pred = int(self.model.predict(features.to_numpy(dtype=float).reshape(1, -1))[0])
                signal = 1 if pred == 1 else -1
                self.stats["predictions"] += 1
                if signal == 1:
                    self.stats["pred_long"] += 1
                else:
                    self.stats["pred_short"] += 1
                self.last_signal = signal
                self.last_prediction = {
                    "time": event_ts,
                    "bar_number": n - 1,
                    "signal": signal,
                    "price": curr,
                    "position_before": self.position["side"] if self.position else None,
                }
                self._log(f"PREDICT signal={signal:+d} bar={n - 1} pos_before={self.position['side'] if self.position else 'NONE'}")
                self._manage_position(signal, curr, event_ts)
            except Exception as exc:
                self.stats["prediction_errors"] += 1
                self.last_compute_error = f"prediction: {exc}"
                self._log(self.last_compute_error)

    def _bars_frame(self) -> pd.DataFrame:
        if not self.dollar_bars:
            return pd.DataFrame()
        df = pd.DataFrame(self.dollar_bars)
        df["ts"] = pd.to_datetime(df["close_time"])
        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize("Europe/Moscow")
        else:
            df["ts"] = df["ts"].dt.tz_convert("Europe/Moscow")
        # Duplicate timestamps are valid; never drop them.
        return df.set_index("ts", drop=True).sort_index(kind="mergesort")

    def _compute_features(self) -> Optional[pd.Series]:
        self.last_compute_error = None
        self.last_compute_missing = []
        bars = self._bars_frame()
        if len(bars) < self.min_bars_for_features:
            self.last_compute_error = f"not enough bars: {len(bars)} < {self.min_bars_for_features}"
            return None
        try:
            enriched = self.micro_builder.build_all_features(bars.copy())
            featured = self.feature_builder.build_all(enriched)
        except Exception as exc:
            self.stats["feature_errors"] += 1
            self.last_compute_error = f"features: {exc}"
            self._log(self.last_compute_error)
            return None

        numeric = featured.select_dtypes(include=[np.number]).copy()
        # Research creates event features as bar_<feature>.
        prefixed = numeric.copy()
        prefixed.columns = [f"bar_{c}" for c in numeric.columns]
        combined = pd.concat([numeric, prefixed], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]

        missing = [c for c in self.feature_cols if c not in combined.columns]
        if missing:
            self.last_compute_missing = missing
            self.stats["feature_errors"] += 1
            self.last_compute_error = f"missing features: {missing}"
            self._log(self.last_compute_error)
            return None
        row = combined[self.feature_cols].iloc[-1]
        if row.isna().any() or not np.isfinite(row.to_numpy(dtype=float)).all():
            self.stats["feature_errors"] += 1
            self.last_compute_error = "features: NaN/inf in final row"
            self._log(self.last_compute_error)
            return None
        return row

    def _check_vertical_barrier(self, current_bar_number: int, timestamp: pd.Timestamp, price: float) -> bool:
        """Close an open position after the configured vertical-barrier horizon.

        Research passes vertical_barrier to triple-barrier labeling as a bar
        horizon. Live therefore counts completed dollar bars from the entry bar;
        target_window is deliberately NOT treated as a second execution timer.
        """
        if self.position is None:
            return False
        p = self.position
        entry_bar = p.get("entry_bar_number")
        vertical = self.params.get("vertical_barrier")
        if vertical is None or entry_bar is None:
            return False
        try:
            vertical = int(vertical)
        except (TypeError, ValueError):
            return False
        if vertical <= 0:
            return False
        if current_bar_number >= int(entry_bar) + vertical:
            self.stats["vertical_closes"] += 1
            self._close_position(price, timestamp, reason="vertical")
            return True
        return False

    def _get_tp_sl_pct(self) -> tuple[float, float]:
        """Return TP%, SL%; scalar pt_sl remains backward compatible."""
        value = self.params.get("pt_sl", 1.5)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return float(value[0]), float(value[1])
        pct = float(value)
        return pct, pct

    def _check_barriers(self, bar: Dict[str, Any]) -> None:
        """Проверяет TP/SL по high/low бара. Закрывает позицию при пробитии."""
        if self.position is None:
            return
        p = self.position
        entry_price = p["entry_price"]
        side = p["side"]
        high = float(bar["high"])
        low = float(bar["low"])
        bar_ts = self._ts(bar["close_time"])

        tp_pct, sl_pct = self._get_tp_sl_pct()
        if side == "long":
            tp_level = entry_price * (1 + tp_pct / 100)
            sl_level = entry_price * (1 - sl_pct / 100)
            if high >= tp_level:
                self.stats["barrier_closes"] += 1
                self._close_position(tp_level, bar_ts, reason="take_profit")
                return
            if low <= sl_level:
                self.stats["barrier_closes"] += 1
                self._close_position(sl_level, bar_ts, reason="stop_loss")
                return
        else:  # short
            tp_level = entry_price * (1 - tp_pct / 100)
            sl_level = entry_price * (1 + sl_pct / 100)
            if low <= tp_level:
                self.stats["barrier_closes"] += 1
                self._close_position(tp_level, bar_ts, reason="take_profit")
                return
            if high >= sl_level:
                self.stats["barrier_closes"] += 1
                self._close_position(sl_level, bar_ts, reason="stop_loss")
                return

    def _manage_position(self, signal: int, price: float, timestamp: pd.Timestamp) -> None:
        # Battle-mode liquidity protection: no new positions from 22:00.
        # Existing positions can still be closed by reverse/vertical/barriers.
        entry_cutoff = pd.Timestamp("22:00:00").time()
        if self.position is None:
            if signal in (-1, 1):
                if timestamp.time() >= entry_cutoff:
                    self.stats["opens_blocked_after_22"] += 1
                    self._log(f"OPEN blocked after 22:00 signal={signal:+d}")
                    return
                self._open_position(signal, price, timestamp)
            return
        
        p = self.position
        entry_price = p["entry_price"]
        side = p["side"]
        
        # Процентный порог TP/SL из параметров (например, 1.5 означает 1.5%)
        tp_pct, sl_pct = self._get_tp_sl_pct()
        tp_level = entry_price * (1 + tp_pct / 100) if side == "long" else entry_price * (1 - tp_pct / 100)
        sl_level = entry_price * (1 - sl_pct / 100) if side == "long" else entry_price * (1 + sl_pct / 100)
        
        # 1. Проверка барьеров (TP/SL)
        if side == "long":
            if price >= tp_level:
                self._close_position(price, timestamp, reason="take_profit")
                return
            elif price <= sl_level:
                self._close_position(price, timestamp, reason="stop_loss")
                return
        else:  # short
            if price <= tp_level:
                self._close_position(price, timestamp, reason="take_profit")
                return
            elif price >= sl_level:
                self._close_position(price, timestamp, reason="stop_loss")
                return
        
        # 2. Проверка смены сигнала (REVERSE)
        same = (side == "long" and signal == 1) or (side == "short" and signal == -1)
        if same:
            self.stats["same_signal"] += 1
            self._log(f"HOLD same_signal={signal:+d} side={side.upper()}")
            return

        self.stats["reversals"] += 1
        self._close_position(price, timestamp, reason="reverse_signal")
        if timestamp.time() < entry_cutoff and signal in (-1, 1):
            self._open_position(signal, price, timestamp)
        else:
            self.stats["opens_blocked_after_22"] += 1
            self._log(f"REVERSE reopen blocked after 22:00 signal={signal:+d}")

    def _open_position(self, signal: int, price: float, timestamp: pd.Timestamp) -> None:
        if price <= 0 or self.lot_size <= 0:
            return
        lots = int(self.capital / (price * self.lot_size))
        if lots <= 0:
            self.stats["open_skipped_insufficient_capital"] += 1
            self._log("OPEN skipped: insufficient capital")
            return
        side = "long" if signal == 1 else "short"
        entry_bar_number = len(self.dollar_bars) - 1
        self.position = {"side": side, "entry_price": price,
                         "entry_time": timestamp, "entry_bar_number": entry_bar_number,
                         "lots": int(lots),
                         "shares": int(lots * self.lot_size),
                         "notional": float(lots * self.lot_size * price)}
        self.stats["opens"] += 1
        self._log(f"OPEN {side.upper()} lots={lots} shares={int(lots*self.lot_size)} price={price:.6f}")

    def _close_position(self, price: float, timestamp: pd.Timestamp, reason: str = "signal_change") -> None:
        if self.position is None:
            return
        p = self.position
        multiplier = p["lots"] * self.lot_size
        gross = (price - p["entry_price"]) * multiplier if p["side"] == "long" else (p["entry_price"] - price) * multiplier
        fees = (p["entry_price"] * multiplier + price * multiplier) * self.commission
        pnl = gross - fees
        self.capital += pnl
        base = p["entry_price"] * multiplier
        pnl_pct = (pnl / base * 100) if base else 0.0
        
        if reason == "reverse_signal":
            self.stats["reverse_closes"] += 1
        self.stats["closed_trades"] += 1
        self.trades.append({
            "entry_time": p["entry_time"],
            "exit_time": timestamp,
            "side": p["side"],
            "entry_price": p["entry_price"],
            "exit_price": price,
            "lots": p["lots"],
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": reason,
        })
        
        self._log(f"CLOSE {p['side'].upper()} {reason} lots={p['lots']} pnl={pnl:+.2f} ({pnl_pct:+.2f}%)")
        self.position = None

    def _log(self, message: str) -> None:
        line = f"{pd.Timestamp.now(tz='Europe/Moscow').strftime('%H:%M:%S')} | {message}"
        self.logs.append(line)
        if len(self.logs) > 100:
            del self.logs[:-100]
        logger.info("%s: %s", self.ticker, message)

    def get_status(self, current_price: Optional[float] = None) -> Dict[str, Any]:
        unrealized = 0.0
        if self.position is not None and current_price is not None:
            p = self.position
            mult = p["lots"] * self.lot_size
            unrealized = ((current_price - p["entry_price"]) if p["side"] == "long" else (p["entry_price"] - current_price)) * mult
        return {"capital": self.capital, "position": self.position, "trades_count": len(self.trades),
                "last_signal": self.last_signal, "warmup_done": self.warmup_done,
                "bars_count": len(self.dollar_bars), "event_count": len(self.cusum_events),
                "unrealized_pnl": unrealized, "total_equity": self.capital + unrealized,
                "current_price": current_price if current_price is not None else self.last_trade_price,
                "live_started": self.live_started}

    def get_debug_info(self) -> Dict[str, Any]:
        return {"bar_count": len(self.dollar_bars), "event_count": len(self.cusum_events),
                "warmup_done": self.warmup_done, "bars_available": len(self.dollar_bars),
                "last_signal": self.last_signal, "capital": self.capital, "position": self.position,
                "trades_count": len(self.trades), "last_compute_error": self.last_compute_error,
                "last_compute_missing": self.last_compute_missing, "logs": list(self.logs),
                "last_event": self.cusum_events[-1] if self.cusum_events else None,
                "last_processed_time": self.last_processed_time, "buffer_value": self._bar_value,
                "dollar_threshold": self.dollar_threshold, "cusum_threshold": self.cusum_threshold,
                "seen_trade_ids": len(self._seen_trade_ids), "seen_fallback_keys": len(self._seen_fallback_keys),
                "live_started": self.live_started, "live_cutover_time": self.live_cutover_time,
                "live_cutover_trade_count": self.live_cutover_trade_count,
                "last_prediction": self.last_prediction, "last_trade_price": self.last_trade_price,
                "stats": dict(self.stats)}
