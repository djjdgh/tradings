"""
事件定义 — 系统内所有模块间通信的数据结构
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EventType(Enum):
    """事件类型"""
    BAR = "bar"               # K线闭合
    TICK = "tick"              # 实时价格
    SIGNAL = "signal"          # 策略信号
    ORDER = "order"            # 订单请求
    FILL = "fill"              # 成交回报
    POSITION = "position"      # 持仓更新
    ACCOUNT = "account"        # 账户更新
    RISK_ALERT = "risk_alert"  # 风控告警


class Side(Enum):
    """交易方向"""
    LONG = "long"
    SHORT = "short"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SignalAction(Enum):
    """信号动作"""
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


@dataclass
class Bar:
    """K线数据"""
    symbol: str
    exchange: str
    timeframe: str
    timestamp: float          # Unix 毫秒时间戳
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = True    # K线是否已闭合


@dataclass
class Tick:
    """实时价格"""
    symbol: str
    exchange: str
    timestamp: float
    bid: float
    ask: float
    last: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    exchange: str
    strategy: str
    action: SignalAction
    price: float              # 信号触发价格
    timestamp: float = field(default_factory=time.time)
    strength: float = 1.0     # 信号强度 (0-1)，元标签输出
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Order:
    """订单"""
    id: str
    symbol: str
    exchange: str
    side: Side
    order_type: OrderType
    amount: float             # 合约数量
    price: Optional[float] = None  # 限价单价格
    status: OrderStatus = OrderStatus.PENDING
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    strategy: str = ""
    filled_amount: float = 0.0
    filled_price: float = 0.0
    fee: float = 0.0
    exchange_order_id: Optional[str] = None


@dataclass
class Position:
    """持仓"""
    symbol: str
    exchange: str
    side: Side
    amount: float             # 持仓数量
    entry_price: float        # 开仓均价
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: float = 1.0
    timestamp: float = field(default_factory=time.time)
    strategy: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    bars_held: int = 0        # 已持仓K线数
    highest_price: float = 0.0   # 持仓期间最高价（用于追踪止盈）
    lowest_price: float = float('inf')  # 持仓期间最低价


@dataclass
class AccountSnapshot:
    """账户快照"""
    exchange: str
    timestamp: float
    total_equity: float       # 总权益
    available_balance: float  # 可用余额
    unrealized_pnl: float     # 未实现盈亏
    margin_used: float        # 已用保证金


@dataclass
class RiskAlert:
    """风控告警"""
    level: str                # "warning" | "critical" | "emergency"
    message: str
    action: str               # 建议动作: "reduce" | "close_all" | "halt"
    timestamp: float = field(default_factory=time.time)
