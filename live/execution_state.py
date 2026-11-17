"""Small helpers for displaying execution state safely in the UI."""

def status(engine_snapshot: dict) -> dict:
    return {
        "engine_running": bool(engine_snapshot.get("running")),
        "automated_trading": bool(engine_snapshot.get("automated_trading")),
        "execution_mode": str(engine_snapshot.get("execution_mode", "paper")).upper(),
        "last_update": engine_snapshot.get("last_update"),
        "poll_count": int(engine_snapshot.get("poll_count", 0)),
    }
