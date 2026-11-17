from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BCSOrder:
    ticker: str
    side: str
    lots: int
    shares: int
    price: Optional[float] = None
    order_type: str = "market"
    client_order_id: Optional[str] = None
    reason: str = ""
    class_code: str = "TQBR"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BCSOrderResult:
    ok: bool
    status: str
    order_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class BCSTrade:
    trade_id: str
    order_id: Optional[str]
    ticker: str
    side: str
    price: float
    shares: int
    lots: int
    trade_time: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BCSPosition:
    ticker: str
    side: str
    shares: int
    lots: int
    average_price: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)
