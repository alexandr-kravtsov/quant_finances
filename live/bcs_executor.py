from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from .bcs_client import BCSClient
from .bcs_models import BCSOrder, BCSOrderResult

logger = logging.getLogger("jerichon.bcs_executor")


class BCSExecutor:
    """Translate JERICHON execution intents into BCS orders.

    PAPER never calls BCS. LIVE sends a real HTTP order to the BCS operations
    service.  Market orders are the default execution type.
    """

    def __init__(self, mode: Optional[str] = None, client: Optional[BCSClient] = None):
        if load_dotenv is not None:
            load_dotenv(Path(__file__).resolve().parents[1] / "API.env", override=False)
            load_dotenv(Path(__file__).resolve().parents[2] / "API.env", override=False)

        self.mode = (mode or os.getenv("BCS_EXECUTION_MODE", "paper")).strip().lower()
        if self.mode not in {"paper", "live"}:
            raise ValueError("BCS_EXECUTION_MODE must be 'paper' or 'live'")

        self.client = client or BCSClient()
        self.default_class_code = os.getenv("BCS_DEFAULT_CLASS_CODE", "TQBR").strip() or "TQBR"

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    def submit(self, order: BCSOrder) -> BCSOrderResult:
        # BCS requires a UUID clientOrderId.  Keep the same UUID when an
        # intent is retried so a transient network error cannot create a new
        # independent order on every retry.
        client_order_id = order.client_order_id or str(uuid.uuid4())
        payload = self._build_payload(order, client_order_id)

        logger.info(
            "BCS submit mode=%s ticker=%s side=%s type=%s shares=%s class=%s clientOrderId=%s reason=%s",
            self.mode,
            order.ticker,
            order.side.upper(),
            payload.get("orderType"),
            order.shares,
            payload.get("classCode"),
            client_order_id,
            order.reason,
        )

        if self.mode == "paper":
            logger.info("[BCS PAPER] payload=%s", payload)
            return BCSOrderResult(
                True,
                "paper",
                client_order_id,
                payload,
            )

        try:
            result = self.client.create_order(payload)
            order_id = self._extract_order_id(result, client_order_id)
            logger.info(
                "BCS API ACCEPTED ticker=%s side=%s shares=%s clientOrderId=%s orderId=%s response=%s",
                order.ticker,
                order.side.upper(),
                order.shares,
                client_order_id,
                order_id,
                result,
            )
            return BCSOrderResult(
                True,
                "submitted",
                order_id,
                result,
            )
        except Exception as exc:
            logger.exception(
                "BCS API order failed ticker=%s side=%s shares=%s clientOrderId=%s payload=%s",
                order.ticker,
                order.side.upper(),
                order.shares,
                client_order_id,
                payload,
            )
            return BCSOrderResult(
                False,
                "error",
                client_order_id,
                payload,
                str(exc),
            )

    @staticmethod
    def _extract_order_id(result: dict, client_order_id: str) -> str:
        # Create-order responses have changed shape across BCS versions.
        # Preserve the broker order id when present; otherwise the UUID we
        # generated remains the authoritative lookup id.
        if isinstance(result, dict):
            for key in ("orderId", "order_id", "id"):
                value = result.get(key)
                if value:
                    return str(value)
            data = result.get("data")
            if isinstance(data, dict):
                for key in ("orderId", "order_id", "id"):
                    value = data.get(key)
                    if value:
                        return str(value)
            if isinstance(data, list) and data:
                for item in data:
                    if isinstance(item, dict):
                        value = item.get("orderId") or item.get("order_id") or item.get("id")
                        if value:
                            return str(value)
            value = result.get("clientOrderId")
            if value:
                return str(value)
        return str(client_order_id)

    def get_order_status(self, order_id: str) -> BCSOrderResult:
        """Fetch the current broker-side status for a submitted order."""
        try:
            result = self.client.get_order(order_id)
            return BCSOrderResult(True, "status", order_id, result)
        except Exception as exc:
            logger.exception("BCS order status failed order_id=%s", order_id)
            return BCSOrderResult(False, "error", order_id, error=str(exc))

    def _build_payload(self, order: BCSOrder, client_order_id: str) -> dict:
        side = str(order.side).strip().lower()
        if side in {"buy", "b", "1"}:
            side_code = "1"
        elif side in {"sell", "s", "2"}:
            side_code = "2"
        else:
            raise ValueError(f"Unsupported BCS side: {order.side!r}")

        order_type = str(order.order_type or "market").strip().lower()
        if order_type in {"market", "m", "1"}:
            order_type_code = "1"
        elif order_type in {"limit", "l", "2"}:
            order_type_code = "2"
        else:
            raise ValueError(f"Unsupported BCS order type: {order.order_type!r}")

        class_code = (
            order.class_code
            or order.metadata.get("class_code")
            or self.default_class_code
        )

        payload = {
            "clientOrderId": client_order_id,
            "side": side_code,
            "orderType": order_type_code,
            "orderQuantity": int(order.shares),
            "ticker": order.ticker,
            "classCode": class_code,
        }

        # Market orders deliberately contain no price.  For limit orders,
        # price is mandatory and is passed through as a number.
        if order_type_code == "2":
            if order.price is None:
                raise ValueError("BCS limit order requires price")
            payload["price"] = float(order.price)

        return payload
