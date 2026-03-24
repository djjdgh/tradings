"""
交易引擎 — 系统核心，协调数据流、策略、风控和执行
"""

from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger

from src.core.event import (
    AccountSnapshot, Bar, EventType, Order, Position, RiskAlert,
    Signal, SignalAction, Side, Tick,
)
from src.data.feed import DataFeed
from src.data.store import DataStore
from src.exchange.base import BaseExchange
from src.execution.order_manager import OrderManager
from src.monitor.notifier import Notifier
from src.risk.manager import RiskManager
from src.strategy.base import BaseStrategy


class TradingEngine:
    """
    实盘交易引擎

    事件流:
    DataFeed → on_bar/on_tick → Strategy → Signal → RiskManager → OrderManager → Exchange
    """

    def __init__(
        self,
        exchanges: dict[str, BaseExchange],
        strategies: dict[str, BaseStrategy],
        risk_manager: RiskManager,
        order_manager: OrderManager,
        data_feed: DataFeed,
        data_store: DataStore,
        notifier: Optional[Notifier] = None,
        config: dict | None = None,
    ):
        self.exchanges = exchanges
        self.strategies = strategies
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.data_feed = data_feed
        self.data_store = data_store
        self.notifier = notifier
        self.config = config or {}

        self._running = False
        self._positions: dict[str, Position] = {}  # symbol -> Position
        self._account: dict[str, AccountSnapshot] = {}  # exchange -> snapshot

    async def start(self) -> None:
        """启动交易引擎"""
        logger.info("Trading engine starting...")

        # 初始化数据库
        await self.data_store.initialize()

        # 连接交易所
        for name, exchange in self.exchanges.items():
            await exchange.connect()
            logger.info(f"Connected to exchange: {name}")

        # 同步账户和持仓
        await self._sync_account()
        await self._sync_positions()

        self._running = True
        logger.info("Trading engine started")

        if self.notifier:
            await self.notifier.send("Trading engine started successfully")

        # 启动数据订阅和事件处理
        await self._run_event_loop()

    async def stop(self) -> None:
        """停止交易引擎"""
        logger.info("Trading engine stopping...")
        self._running = False

        for name, exchange in self.exchanges.items():
            await exchange.close()
            logger.info(f"Disconnected from exchange: {name}")

        if self.notifier:
            await self.notifier.send("Trading engine stopped")

        logger.info("Trading engine stopped")

    async def _run_event_loop(self) -> None:
        """主事件循环 — 订阅行情并处理"""
        symbols_config = self.config.get("symbols", [])

        tasks = []
        for sym_cfg in symbols_config:
            symbol = sym_cfg["symbol"]
            exchange_name = sym_cfg["exchange"]
            timeframe = sym_cfg["timeframe"]

            # 订阅K线数据
            task = asyncio.create_task(
                self._watch_bars(symbol, exchange_name, timeframe)
            )
            tasks.append(task)

            # 订阅Tick数据（用于实时止盈止损）
            tick_task = asyncio.create_task(
                self._watch_ticks(symbol, exchange_name)
            )
            tasks.append(tick_task)

        # 定时同步账户
        tasks.append(asyncio.create_task(self._periodic_sync()))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Event loop cancelled")
        except Exception as e:
            logger.exception(f"Event loop error: {e}")
            if self.notifier:
                await self.notifier.send(f"CRITICAL: Event loop error: {e}")

    async def _watch_bars(
        self, symbol: str, exchange_name: str, timeframe: str
    ) -> None:
        """监听K线闭合事件"""
        exchange = self.exchanges[exchange_name]

        while self._running:
            try:
                bar = await self.data_feed.watch_ohlcv(
                    exchange, symbol, timeframe
                )
                if bar and bar.is_closed:
                    await self._on_bar(bar)
            except Exception as e:
                logger.error(f"Watch bars error [{symbol}]: {e}")
                await asyncio.sleep(5)  # 错误后短暂等待再重试

    async def _watch_ticks(self, symbol: str, exchange_name: str) -> None:
        """监听实时价格（用于止盈止损检查）"""
        exchange = self.exchanges[exchange_name]

        while self._running:
            try:
                tick = await self.data_feed.watch_ticker(exchange, symbol)
                if tick:
                    await self._on_tick(tick)
            except Exception as e:
                logger.error(f"Watch ticks error [{symbol}]: {e}")
                await asyncio.sleep(5)

    async def _on_bar(self, bar: Bar) -> None:
        """处理K线闭合事件"""
        logger.debug(f"Bar: {bar.symbol} {bar.timeframe} close={bar.close}")

        # 存储K线
        await self.data_store.save_bar(bar)

        # 更新持仓统计
        if bar.symbol in self._positions:
            pos = self._positions[bar.symbol]
            pos.bars_held += 1
            pos.highest_price = max(pos.highest_price, bar.high)
            pos.lowest_price = min(pos.lowest_price, bar.low)

        # 查找对应策略并计算信号
        for strategy_name, strategy in self.strategies.items():
            if not strategy.should_process(bar.symbol, bar.timeframe):
                continue

            try:
                # 获取历史K线供策略计算
                bars = await self.data_store.get_bars(
                    bar.symbol, bar.exchange, bar.timeframe,
                    limit=strategy.required_bars
                )

                signal = await strategy.on_bar(bar, bars)

                if signal:
                    await self._process_signal(signal)
            except Exception as e:
                logger.error(
                    f"Strategy error [{strategy_name}][{bar.symbol}]: {e}"
                )

    async def _on_tick(self, tick: Tick) -> None:
        """处理实时价格 — 检查止盈止损"""
        if tick.symbol not in self._positions:
            return

        position = self._positions[tick.symbol]

        # 检查止盈止损
        close_signal = self.risk_manager.check_stop_loss(position, tick)
        if close_signal:
            logger.warning(
                f"Stop triggered for {tick.symbol}: {close_signal.action}"
            )
            await self._process_signal(close_signal)

    async def _process_signal(self, signal: Signal) -> None:
        """处理交易信号 — 风控检查 → 下单"""
        logger.info(
            f"Signal: {signal.symbol} {signal.action.value} "
            f"price={signal.price} strength={signal.strength}"
        )

        # 风控检查
        risk_result = self.risk_manager.evaluate_signal(
            signal=signal,
            positions=self._positions,
            account=self._get_current_account(signal.exchange),
        )

        if not risk_result.approved:
            logger.warning(
                f"Signal rejected by risk manager: {risk_result.reason}"
            )
            if self.notifier:
                await self.notifier.send(
                    f"Signal rejected: {signal.symbol} "
                    f"{signal.action.value} - {risk_result.reason}"
                )
            return

        # 计算仓位大小
        position_size = risk_result.position_size

        # 构建并提交订单
        if signal.action in (SignalAction.OPEN_LONG, SignalAction.OPEN_SHORT):
            side = (
                Side.LONG if signal.action == SignalAction.OPEN_LONG
                else Side.SHORT
            )
            order = await self.order_manager.open_position(
                exchange=self.exchanges[signal.exchange],
                symbol=signal.symbol,
                side=side,
                amount=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy=signal.strategy,
            )
        else:
            order = await self.order_manager.close_position(
                exchange=self.exchanges[signal.exchange],
                symbol=signal.symbol,
                strategy=signal.strategy,
            )

        if order:
            await self.data_store.save_order(order)
            if self.notifier:
                await self.notifier.send(
                    f"Order: {order.symbol} {order.side.value} "
                    f"amount={order.amount} price={order.filled_price}"
                )

        # 同步持仓
        await self._sync_positions()

    async def _sync_account(self) -> None:
        """同步所有交易所的账户信息"""
        for name, exchange in self.exchanges.items():
            try:
                snapshot = await exchange.fetch_account()
                self._account[name] = snapshot
                await self.data_store.save_account_snapshot(snapshot)
            except Exception as e:
                logger.error(f"Sync account error [{name}]: {e}")

    async def _sync_positions(self) -> None:
        """同步所有交易所的持仓"""
        for name, exchange in self.exchanges.items():
            try:
                positions = await exchange.fetch_positions()
                for pos in positions:
                    key = pos.symbol
                    if pos.amount > 0:
                        # 保留已有的追踪数据
                        if key in self._positions:
                            pos.bars_held = self._positions[key].bars_held
                            pos.highest_price = max(
                                self._positions[key].highest_price,
                                pos.entry_price
                            )
                            pos.lowest_price = min(
                                self._positions[key].lowest_price,
                                pos.entry_price
                            )
                        self._positions[key] = pos
                    elif key in self._positions:
                        del self._positions[key]
            except Exception as e:
                logger.error(f"Sync positions error [{name}]: {e}")

    async def _periodic_sync(self) -> None:
        """定期同步账户和持仓"""
        while self._running:
            await asyncio.sleep(60)  # 每分钟同步一次
            await self._sync_account()

            # 检查全局风控
            for name, account in self._account.items():
                alert = self.risk_manager.check_account(account)
                if alert:
                    logger.critical(f"Risk alert: {alert.message}")
                    if self.notifier:
                        await self.notifier.send(
                            f"RISK ALERT [{alert.level}]: {alert.message}"
                        )
                    if alert.action == "halt":
                        logger.critical("HALTING TRADING ENGINE")
                        await self.stop()
                        return

    def _get_current_account(
        self, exchange_name: str
    ) -> Optional[AccountSnapshot]:
        """获取当前账户快照"""
        return self._account.get(exchange_name)
