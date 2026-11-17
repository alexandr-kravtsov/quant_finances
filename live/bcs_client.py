from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import urllib3

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class BCSClient:
    """HTTP client for BCS Trade API with refresh-token authentication.

    BCS issues a refresh token from the web cabinet. Trading HTTP endpoints
    require an access token. The access token lives for about 24 hours, so
    this client exchanges the refresh token automatically and refreshes it
    again after expiry or a 401 response.
    """

    OPERATIONS_BASE = "https://be.broker.ru/trade-api-bff-operations"
    ORDER_DETAILS_BASE = "https://be.broker.ru/trade-api-bff-order-details"
    TRADE_DETAILS_BASE = "https://be.broker.ru/trade-api-bff-trade-details"
    TOKEN_URL = "https://be.broker.ru/trade-api-keycloak/realms/tradeapi/protocol/openid-connect/token"
    CLIENT_ID = "trade-api-write"

    def __init__(
        self,
        token: Optional[str] = None,
        timeout: float = 15.0,
        verify_ssl: bool = False,
    ):
        if load_dotenv is not None:
            load_dotenv(Path(__file__).resolve().parents[1] / "API.env", override=False)
            load_dotenv(Path(__file__).resolve().parents[2] / "API.env", override=False)

        # BCS web cabinet tokens are REFRESH tokens.  BCS_TRADE_TOKEN is kept
        # as a backward-compatible variable name from the previous version.
        self.refresh_token = (
            token
            or os.getenv("BCS_REFRESH_TOKEN", "")
            or os.getenv("BCS_TRADE_TOKEN", "")
        ).strip()
        self.access_token = os.getenv("BCS_ACCESS_TOKEN", "").strip()
        self.client_id = os.getenv("BCS_CLIENT_ID", self.CLIENT_ID).strip() or self.CLIENT_ID
        self.timeout = float(timeout)
        self.verify_ssl = bool(verify_ssl)
        self._access_expires_at = 0.0
        self._auth_lock = threading.Lock()

        self.operations_base = os.getenv("BCS_OPERATIONS_BASE", self.OPERATIONS_BASE).rstrip("/")
        self.order_details_base = os.getenv("BCS_ORDER_DETAILS_BASE", self.ORDER_DETAILS_BASE).rstrip("/")
        self.trade_details_base = os.getenv("BCS_TRADE_DETAILS_BASE", self.TRADE_DETAILS_BASE).rstrip("/")

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        if self.access_token:
            self.session.headers["Authorization"] = f"Bearer {self.access_token}"

    def _refresh_access_token(self) -> str:
        if not self.refresh_token:
            raise RuntimeError(
                "BCS refresh token is missing. Set BCS_REFRESH_TOKEN in API.env "
                "(the token issued in the BCS web cabinet)."
            )

        with self._auth_lock:
            # Another thread may already have refreshed the token.
            if self.access_token and time.monotonic() < self._access_expires_at:
                return self.access_token

            response = requests.post(
                self.TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "refresh_token": self.refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=self.timeout,
                verify=self.verify_ssl,
                headers={"Accept": "application/json"},
            )
            if not response.ok:
                body = response.text.strip()
                message = f"BCS token refresh failed: {response.status_code} {response.reason}"
                if body:
                    message += f" | body={body[:2000]}"
                raise requests.HTTPError(message, response=response)

            data = response.json()
            access_token = str(data.get("access_token", "")).strip()
            if not access_token:
                raise RuntimeError("BCS token endpoint returned no access_token")

            expires_in = float(data.get("expires_in", 86400))
            self.access_token = access_token
            self._access_expires_at = time.monotonic() + max(60.0, expires_in - 60.0)
            self.session.headers["Authorization"] = f"Bearer {access_token}"
            return access_token

    def _ensure_access_token(self) -> None:
        # If an explicit access token was supplied without a refresh token,
        # use it as-is. Otherwise obtain/refresh the access token automatically.
        if self.access_token and not self.refresh_token:
            return
        if not self.access_token or time.monotonic() >= self._access_expires_at:
            self._refresh_access_token()

    def _request(self, method: str, base_url: str, path: str, **kwargs) -> Dict[str, Any]:
        url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
        self._ensure_access_token()

        response = self.session.request(
            method,
            url,
            timeout=self.timeout,
            verify=self.verify_ssl,
            **kwargs,
        )

        # An expired/revoked access token must never become a permanent live
        # failure. Refresh once and retry the same request exactly once.
        if response.status_code == 401 and self.refresh_token:
            self._access_expires_at = 0.0
            self._refresh_access_token()
            response = self.session.request(
                method,
                url,
                timeout=self.timeout,
                verify=self.verify_ssl,
                **kwargs,
            )

        if not response.ok:
            body = response.text.strip()
            message = f"{response.status_code} {response.reason} for url: {url}"
            if body:
                message += f" | body={body[:2000]}"
            raise requests.HTTPError(message, response=response)

        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            return {"raw_text": response.text}

    def create_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", self.operations_base, "/api/v1/orders", json=payload)

    def get_order(self, order_id: str) -> Dict[str, Any]:
        return self._request(
            "GET",
            self.operations_base,
            "/api/v1/orders",
            params={"orderIdType": "1", "orderId": order_id},
        )

    def search_orders(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request(
            "POST",
            self.order_details_base,
            "/api/v1/orders/search",
            json=payload or {},
        )

    def search_trades(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request(
            "POST",
            self.trade_details_base,
            "/api/v1/trades/search",
            json=payload or {},
        )
