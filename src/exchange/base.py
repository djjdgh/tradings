"""
交易所抽象基类 — 定义统一接口
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

from src.core.event import AccountSnapshot, Order, Position, Side, OrderType, OrderStatus


class BaseExchange(ABC):
    """交易所抽象基类"""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @abstractmethod
    async def connect(self) -> None:
        """连接交易所"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """断开连接"""
        ...

    @abstractmethod
    async def fetch_account(self) -> AccountSnapshot:
        """获取账户信息"""
        ...

    @abstractmethod
    async def fetch_positions(self) -> list[Position]:
        """获取当前持仓"""
        ...

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[dict] = None,
    ) -> Order:
        """创建订单"""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        ...

    @abstractmethod
    async def fetch_funding_rate(self, symbol: str) -> dict:
        """获取当前资金费率"""
        ...

    @abstractmethod
    async def fetch_open_interest(self, symbol: str) -> dict:
        """获取未平仓量"""
        ...

    # ─── WebSocket 方法（ccxt.pro 代理） ───

    @abstractmethod
    async def watch_ohlcv(self, symbol: str, timeframe: str) -> list:
        """WebSocket 监听K线"""
        ...

    @abstractmethod
    async def watch_ticker(self, symbol: str) -> dict:
        """WebSocket 监听Ticker"""
        ...

    @abstractmethod
    async def watch_order_book(self, symbol: str, limit: int = 10) -> dict:
        """WebSocket 监听订单簿"""
        ...

    @abstractmethod
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str,
        since: Optional[int] = None, limit: int = 200,
    ) -> list:
        """REST 获取历史K线"""
        ...

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """设置杠杆倍数（子类可覆写）"""
        pass
