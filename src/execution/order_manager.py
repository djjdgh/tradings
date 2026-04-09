"""
订单管理器 — 订单生命周期管理

匹配 engine.py 中的调用接口：
- open_position(exchange, symbol, side, amount, ...) -> Optional[Order]
- close_position(exchange, symbol, strategy) -> Optional[Order]
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional

from loguru import logger

from src.core.event import Order, OrderStatus, OrderType, Side
from src.exchange.base import BaseExchange


class OrderManager:
    """
    订单管理器

    负责：
    1. 市价下单（开仓/平仓）
    2. 止损止盈条件单
    3. 重试机制
    4. 滑点记录
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._active_orders: dict[str, Order] = {}

    async def open_position(
        self,
        exchange: BaseExchange,
        symbol: str,
        side: Side,
        amount: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy: str = "",
    ) -> Optional[Order]:
        """
        开仓 — 市价单

        Args:
            exchange: 交易所实例
            symbol: 交易品种
            side: 方向 (LONG/SHORT)
            amount: 合约数量
            stop_loss: 止损价
            take_profit: 止盈价
            strategy: 策略名称

        Returns:
            Order 或 None
        """
        if amount <= 0:
            logger.warning(f"Invalid amount: {amount}")
            return None

        # 方向映射
        order_side = "buy" if side == Side.LONG else "sell"

        # 带重试的下单
        order = await self._execute_with_retry(
            exchange=exchange,
            symbol=symbol,
            side=order_side,
            order_type="market",
            amount=amount,
        )

        if order is None:
            return None

        # 设置策略和止损止盈
        order.strategy = strategy
        order.stop_loss = stop_loss
        order.take_profit = take_profit

        logger.info(
            f"Position opened: {symbol} {side.value} "
            f"amount={order.filled_amount} price={order.filled_price} "
            f"sl={stop_loss} tp={take_profit}"
        )

        self._active_orders[order.id] = order
        return order

    async def close_position(
        self,
        exchange: BaseExchange,
        symbol: str,
        strategy: str = "",
    ) -> Optional[Order]:
        """
        平仓 — 市价平掉当前仓位

        自动检测持仓方向并反向下单。
        """
        try:
            positions = await exchange.fetch_positions()
        except Exception as e:
            logger.error(f"Fetch positions error: {e}")
            return None

        # 找到匹配的持仓
        target_pos = None
        for pos in positions:
            if pos.symbol == symbol and pos.amount > 0:
                target_pos = pos
                break

        if target_pos is None:
            logger.warning(f"No position to close for {symbol}")
            return None

        # 反向下单平仓
        close_side = "sell" if target_pos.side == Side.LONG else "buy"
        params = {"reduceOnly": True}

        order = await self._execute_with_retry(
            exchange=exchange,
            symbol=symbol,
            side=close_side,
            order_type="market",
            amount=target_pos.amount,
            params=params,
        )

        if order:
            order.strategy = strategy
            logger.info(
                f"Position closed: {symbol} amount={order.filled_amount} "
                f"price={order.filled_price}"
            )

        return order

    async def _execute_with_retry(
        self,
        exchange: BaseExchange,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[dict] = None,
    ) -> Optional[Order]:
        """带重试的订单执行"""
        last_error = None

        for attempt in range(self._max_retries):
            try:
                order = await exchange.create_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    amount=amount,
                    price=price,
                    params=params,
                )

                if order.status == OrderStatus.REJECTED:
                    logger.error(f"Order rejected: {symbol} {side}")
                    return None

                return order

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Order attempt {attempt + 1}/{self._max_retries} "
                    f"failed: {e}"
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))

        logger.error(f"All order attempts failed for {symbol}: {last_error}")
        return None
