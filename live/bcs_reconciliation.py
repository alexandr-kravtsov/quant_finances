from .bcs_client import BCSClient

class BCSReconciler:
    def __init__(self, client: BCSClient):
        self.client = client

    def search_orders(self, payload=None):
        return self.client.search_orders(payload or {})

    def search_trades(self, payload=None):
        return self.client.search_trades(payload or {})

    def compare_local_positions(self, local_positions, broker_positions):
        broker = {}
        for item in broker_positions:
            ticker = item.get("ticker") or item.get("symbol") or item.get("instrument")
            if ticker:
                broker[str(ticker)] = item

        mismatches = []
        for ticker, local in local_positions.items():
            if ticker not in broker:
                mismatches.append({"ticker": ticker, "reason": "missing_on_broker", "local": local})
        for ticker in broker:
            if ticker not in local_positions:
                mismatches.append({"ticker": ticker, "reason": "unexpected_on_broker", "broker": broker[ticker]})

        return {"ok": not mismatches, "mismatches": mismatches}
