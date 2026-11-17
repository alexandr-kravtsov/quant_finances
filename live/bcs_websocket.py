import json
import logging
import os
import threading
import websocket

logger = logging.getLogger("jerichon.bcs_websocket")

class BCSOrderWebSocket:
    URL = "wss://ws.broker.ru/trade-api-bff-operations/api/v1/orders/events/ws"

    def __init__(self, token=None, on_event=None):
        self.token = token or os.getenv("BCS_TRADE_TOKEN", "")
        self.on_event = on_event
        self._ws = None
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="bcs-order-ws")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def _run(self):
        if not self.token:
            logger.warning("BCS_TRADE_TOKEN is empty; websocket not started")
            return
        while not self._stop.is_set():
            try:
                self._ws = websocket.WebSocketApp(
                    self.URL,
                    header=[f"Authorization: Bearer {self.token}"],
                    on_message=self._on_message,
                    on_error=lambda ws, err: logger.error("BCS websocket error: %s", err),
                    on_close=lambda ws, code, msg: logger.info("BCS websocket closed: %s %s", code, msg),
                )
                self._ws.run_forever()
            except Exception:
                logger.exception("BCS websocket failure")
            if not self._stop.is_set():
                self._stop.wait(5)

    def _on_message(self, ws, message):
        try:
            payload = json.loads(message)
        except Exception:
            logger.warning("Invalid BCS websocket message")
            return
        if self.on_event:
            self.on_event(payload)
