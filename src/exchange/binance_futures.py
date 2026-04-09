"""
Binance USDT-M 永续合约交易所实现
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

import ccxt.pro as ccxtpro
from loguru import logger

from src.core.event import (
    AccountSnapshot, Order, OrderStatus, OrderType, Position, Side,
)
from src.exchange.base import BaseExchange


class BinanceFutures(BaseExchange):
    """Binance USDT-M 永续合约"""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self._exchange: Optional[ccxtpro.binanceusdm] = None

    async def connect(self) -> None:
        """连接 Binance 合约"""
        options = self.config.get("options", {})
        sandbox = self.config.get("testnet", False)

        self._exchange = ccxtpro.binanceusdm({
            "apiKey": self.config.get("api_key", ""),
            "secret": self.config.get("api_secret", ""),
            "sandbox": sandbox,
            "options": options,
            "enableRateLimit": True,
        })

        # 加载市场信息
        await self._exchange.load_markets()
        self._connected = True
        logger.info(f"Binance Futures connected (testnet={sandbox})")

    async def close(self) -> None:
        """断开连接"""
        if self._exchange:
            await self._exchange.close()
            self._connected = False
            logger.info("Binance Futures disconnected")

    async def fetch_account(self) -> AccountSnapshot:
        """获取账户信息"""
        balance = await self._exchange.fetch_balance()
        total = balance.get("total", {})
        free = balance.get("free", {})
        usdt_total = float(total.get("USDT", 0))
        usdt_free = float(free.get("USDT", 0))

        # 从 info 中获取更详细的数据
        info = balance.get("info", {})
        unrealized_pnl = 0.0
        margin_used = 0.0

        if isinstance(info, dict):
            for asset in info.get("assets", []):
                if asset.get("asset") == "USDT":
                    unrealized_pnl = float(asset.get("unrealizedProfit", 0))
                    margin_used = float(asset.get("initialMargin", 0))
                    break

        return AccountSnapshot(
            exchange=self.name,
            timestamp=time.time(),
            total_equity=usdt_total,
            available_balance=usdt_free,
            unrealized_pnl=unrealized_pnl,
            margin_used=margin_used,
        )

    async def fetch_positions(self) -> list[Position]:
        """获取当前持仓"""
        positions = await self._exchange.fetch_positions()
        result = []

        for pos in positions:
            contracts = abs(float(pos.get("contracts", 0) or 0))
            if contracts == 0:
                continue

            side_str = pos.get("side", "long")
            side = Side.LONG if side_str == "long" else Side.SHORT

            result.append(Position(
                symbol=pos.get("symbol", ""),
                exchange=self.name,
                side=side,
                amount=contracts,
                entry_price=float(pos.get("entryPrice", 0) or 0),
                unrealized_pnl=float(pos.get("unrealizedPnl", 0) or 0),
                leverage=float(pos.get("leverage", 1) or 1),
                timestamp=time.time(),
            ))

        return result

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
        params = params or {}

        try:
            result = await self._exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params,
            )

            order_side = Side.LONG if side in ("buy", "long") else Side.SHORT
            o_type = OrderType.MARKET if order_type == "market" else OrderType.LIMIT

            return Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                exchange=self.name,
                side=order_side,
                order_type=o_type,
                amount=amount,
                price=price,
                status=OrderStatus.FILLED if result.get("status") == "closed" else OrderStatus.SUBMITTED,
                timestamp=time.time(),
                filled_amount=float(result.get("filled", 0) or 0),
                filled_price=float(result.get("average", 0) or price or 0),
                fee=float(result.get("fee", {}).get("cost", 0) or 0),
                exchange_order_id=str(result.get("id", "")),
            )
        except Exception as e:
            logger.error(f"Create order error: {e}")
            return Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                exchange=self.name,
                side=Side.LONG if side in ("buy", "long") else Side.SHORT,
                order_type=OrderType.MARKET if order_type == "market" else OrderType.LIMIT,
                amount=amount,
                price=price,
                status=OrderStatus.REJECTED,
                timestamp=time.time(),
            )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        try:
            await self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Cancel order error: {e}")
            return False

    async def fetch_funding_rate(self, symbol: str) -> dict:
        """获取当前资金费率"""
        try:
            result = await self._exchange.fetch_funding_rate(symbol)
            return {
                "symbol": symbol,
                "funding_rate": float(result.get("fundingRate", 0) or 0),
                "next_funding_time": float(result.get("fundingTimestamp", 0) or 0),
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"Fetch funding rate error [{symbol}]: {e}")
            return {"symbol": symbol, "funding_rate": 0, "timestamp": time.time()}

    async def fetch_open_interest(self, symbol: str) -> dict:
        """获取未平仓量"""
        try:
            result = await self._exchange.fetch_open_interest(symbol)
            return {
                "symbol": symbol,
                "open_interest": float(result.get("openInterestAmount", 0) or 0),
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"Fetch open interest error [{symbol}]: {e}")
            return {"symbol": symbol, "open_interest": 0, "timestamp": time.time()}

    # ─── WebSocket 方法 ───

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> list:
        """WebSocket 监听K线"""
        return await self._exchange.watch_ohlcv(symbol, timeframe)

    async def watch_ticker(self, symbol: str) -> dict:
        """WebSocket 监听Ticker"""
        return await self._exchange.watch_ticker(symbol)

    async def watch_order_book(self, symbol: str, limit: int = 10) -> dict:
        """WebSocket 监听订单簿"""
        return await self._exchange.watch_order_book(symbol, limit)

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str,
        since: Optional[int] = None, limit: int = 200,
    ) -> list:
        """REST 获取历史K线"""
        return await self._exchange.fetch_ohlcv(
            symbol, timeframe, since=since, limit=limit
        )

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """设置杠杆倍数"""
        try:
            await self._exchange.set_leverage(leverage, symbol)
            logger.info(f"Set leverage {leverage}x for {symbol}")
        except Exception as e:
            logger.warning(f"Set leverage error [{symbol}]: {e}")
