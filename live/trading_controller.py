"""Process-wide controller used by Streamlit to start/stop the live engine."""
from __future__ import annotations

import threading
from typing import Optional

_ENGINE = None
_LOCK = threading.RLock()


def get_engine():
    with _LOCK:
        return _ENGINE


def install_engine(engine):
    global _ENGINE
    with _LOCK:
        if _ENGINE is not None and _ENGINE is not engine:
            _ENGINE.stop()
        _ENGINE = engine
        return _ENGINE


def start_engine():
    with _LOCK:
        if _ENGINE is not None:
            _ENGINE.start()


def stop_engine():
    with _LOCK:
        if _ENGINE is not None:
            _ENGINE.stop()


def set_paused(ticker: str, paused: bool):
    with _LOCK:
        if _ENGINE is not None:
            _ENGINE.set_paused(ticker, paused)


def set_automated_trading(enabled: bool):
    with _LOCK:
        if _ENGINE is not None:
            _ENGINE.set_automated_trading(enabled)


def snapshot() -> dict:
    with _LOCK:
        return _ENGINE.snapshot() if _ENGINE is not None else {
            "running": False, "automated_trading": False, "execution_mode": "paper",
            "last_update": None, "poll_count": 0, "errors": [], "meta": {}
        }
