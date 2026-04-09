"""
数据源 — WebSocket 实时行情 + REST fallback
"""

from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger

from src.core.event import Bar, Tick


class DataFeed:
    """
    实时行情数据源

    通过 ccxt.pro 的 WebSocket 接口获取K线和Ticker数据。
    """

    def __init__(self):
        self._last_bar_ts: dict[str, float] = {}  # key -> last timestamp

    async def watch_ohlcv(
        self, exchange, symbol: str, timeframe: str
    ) -> Optional[Bar]:
        """
        监听K线闭合事件

        使用 ccxt.pro 的 watchOHLCV 方法，
        当检测到新K线时返回已闭合的前一根K线。
        """
        try:
            ohlcvs = await exchange.watch_ohlcv(symbol, timeframe)
            if not ohlcvs:
                return None

            # ccxt 返回 [[timestamp, open, high, low, close, volume], ...]
            # 取最新的K线
            latest = ohlcvs[-1]
            ts, o, h, l, c, v = latest

            key = f"{symbol}:{timeframe}"

            # 检测K线切换（新K线到来意味着前一根已闭合）
            if key in self._last_bar_ts and ts != self._last_bar_ts[key]:
                # 前一根K线已闭合，但我们返回当前数据作为最新闭合K线
                # 实际上 ccxt watchOHLCV 会更新当前未闭合K线
                pass

            self._last_bar_ts[key] = ts

            # 判断K线是否闭合
            # 取倒数第二根（如果有的话）作为已闭合K线
            if len(ohlcvs) >= 2:
                prev = ohlcvs[-2]
                prev_ts, prev_o, prev_h, prev_l, prev_c, prev_v = prev
                prev_key = f"{symbol}:{timeframe}:prev"

                if prev_key not in self._last_bar_ts or \
                   self._last_bar_ts[prev_key] != prev_ts:
                    self._last_bar_ts[prev_key] = prev_ts
                    return Bar(
                        symbol=symbol,
                        exchange=exchange.name,
                        timeframe=timeframe,
                        timestamp=prev_ts,
                        open=prev_o,
                        high=prev_h,
                        low=prev_l,
                        close=prev_c,
                        volume=prev_v,
                        is_closed=True,
                    )

            return None

        except Exception as e:
            logger.error(f"DataFeed watch_ohlcv error [{symbol}]: {e}")
            await asyncio.sleep(1)
            return None

    async def watch_ticker(
        self, exchange, symbol: str
    ) -> Optional[Tick]:
        """
        监听实时价格

        使用 ccxt.pro 的 watchTicker 方法。
        """
        try:
            ticker = await exchange.watch_ticker(symbol)
            if not ticker:
                return None

            return Tick(
                symbol=symbol,
                exchange=exchange.name,
                timestamp=ticker.get("timestamp", 0) or 0,
                bid=ticker.get("bid", 0) or 0,
                ask=ticker.get("ask", 0) or 0,
                last=ticker.get("last", 0) or 0,
            )

        except Exception as e:
            logger.error(f"DataFeed watch_ticker error [{symbol}]: {e}")
            await asyncio.sleep(1)
            return None

    async def watch_order_book(
        self, exchange, symbol: str, limit: int = 10
    ) -> Optional[dict]:
        """
        监听订单簿（微观结构策略用）

        返回 {"bids": [[price, amount], ...], "asks": [[price, amount], ...]}
        """
        try:
            orderbook = await exchange.watch_order_book(symbol, limit)
            return orderbook
        except Exception as e:
            logger.error(f"DataFeed watch_order_book error [{symbol}]: {e}")
            await asyncio.sleep(1)
            return None

    async def fetch_ohlcv(
        self, exchange, symbol: str, timeframe: str,
        since: Optional[int] = None, limit: int = 200,
    ) -> list[Bar]:
        """
        REST 获取历史K线（回测数据下载 + 初始化用）
        """
        try:
            ohlcvs = await exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=limit
            )
            bars = []
            for ts, o, h, l, c, v in ohlcvs:
                bars.append(Bar(
                    symbol=symbol,
                    exchange=exchange.name,
                    timeframe=timeframe,
                    timestamp=ts,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=v,
                    is_closed=True,
                ))
            return bars
        except Exception as e:
            logger.error(f"DataFeed fetch_ohlcv error [{symbol}]: {e}")
            return []

    async def fetch_funding_rate(self, exchange, symbol: str) -> Optional[dict]:
        """
        REST 获取当前资金费率
        """
        try:
            result = await exchange.fetch_funding_rate(symbol)
            return result
        except Exception as e:
            logger.error(f"DataFeed fetch_funding_rate error [{symbol}]: {e}")
            return None
