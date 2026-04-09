"""
数据存储 — 数据库读写操作
"""

from __future__ import annotations

import json
from typing import Optional

from loguru import logger
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from src.core.event import (
    AccountSnapshot, Bar, Order, Signal,
)
from src.data.models import (
    Base, BarModel, OrderModel, AccountSnapshotModel,
    FundingRateModel, SignalModel,
)


class DataStore:
    """数据存储层 — 异步 SQLAlchemy"""

    def __init__(self, db_url: str = "sqlite+aiosqlite:///data/db/tradings.db",
                 echo: bool = False):
        self._engine = create_async_engine(db_url, echo=echo)
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self) -> None:
        """创建所有表"""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized")

    async def close(self) -> None:
        """关闭数据库连接"""
        await self._engine.dispose()

    # ─── Bar 操作 ───

    async def save_bar(self, bar: Bar) -> None:
        """保存K线数据（忽略重复）"""
        async with self._session_factory() as session:
            # 使用 INSERT OR IGNORE 语义
            existing = await session.execute(
                select(BarModel).where(
                    BarModel.symbol == bar.symbol,
                    BarModel.exchange == bar.exchange,
                    BarModel.timeframe == bar.timeframe,
                    BarModel.timestamp == bar.timestamp,
                )
            )
            if existing.scalar_one_or_none() is not None:
                return

            model = BarModel(
                symbol=bar.symbol,
                exchange=bar.exchange,
                timeframe=bar.timeframe,
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
            session.add(model)
            await session.commit()

    async def get_bars(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        limit: int = 200,
    ) -> list[Bar]:
        """获取最近的K线数据"""
        async with self._session_factory() as session:
            result = await session.execute(
                select(BarModel)
                .where(
                    BarModel.symbol == symbol,
                    BarModel.exchange == exchange,
                    BarModel.timeframe == timeframe,
                )
                .order_by(desc(BarModel.timestamp))
                .limit(limit)
            )
            rows = result.scalars().all()

        # 按时间正序返回
        bars = [
            Bar(
                symbol=r.symbol,
                exchange=r.exchange,
                timeframe=r.timeframe,
                timestamp=r.timestamp,
                open=r.open,
                high=r.high,
                low=r.low,
                close=r.close,
                volume=r.volume,
                is_closed=True,
            )
            for r in reversed(rows)
        ]
        return bars

    # ─── Order 操作 ───

    async def save_order(self, order: Order) -> None:
        """保存订单记录"""
        async with self._session_factory() as session:
            model = OrderModel(
                order_id=order.id,
                symbol=order.symbol,
                exchange=order.exchange,
                side=order.side.value,
                order_type=order.order_type.value,
                amount=order.amount,
                price=order.price,
                status=order.status.value,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                strategy=order.strategy,
                filled_amount=order.filled_amount,
                filled_price=order.filled_price,
                fee=order.fee,
                exchange_order_id=order.exchange_order_id,
                timestamp=order.timestamp,
            )
            session.add(model)
            await session.commit()

    # ─── Account Snapshot 操作 ───

    async def save_account_snapshot(self, snapshot: AccountSnapshot) -> None:
        """保存账户快照"""
        async with self._session_factory() as session:
            model = AccountSnapshotModel(
                exchange=snapshot.exchange,
                timestamp=snapshot.timestamp,
                total_equity=snapshot.total_equity,
                available_balance=snapshot.available_balance,
                unrealized_pnl=snapshot.unrealized_pnl,
                margin_used=snapshot.margin_used,
            )
            session.add(model)
            await session.commit()

    async def get_account_snapshots(
        self, exchange: str, limit: int = 100
    ) -> list[dict]:
        """获取账户历史快照"""
        async with self._session_factory() as session:
            result = await session.execute(
                select(AccountSnapshotModel)
                .where(AccountSnapshotModel.exchange == exchange)
                .order_by(desc(AccountSnapshotModel.timestamp))
                .limit(limit)
            )
            rows = result.scalars().all()

        return [
            {
                "timestamp": r.timestamp,
                "total_equity": r.total_equity,
                "available_balance": r.available_balance,
                "unrealized_pnl": r.unrealized_pnl,
                "margin_used": r.margin_used,
            }
            for r in reversed(rows)
        ]

    # ─── Funding Rate 操作 ───

    async def save_funding_rate(
        self,
        symbol: str,
        exchange: str,
        timestamp: float,
        funding_rate: float,
        next_funding_time: Optional[float] = None,
    ) -> None:
        """保存资金费率"""
        async with self._session_factory() as session:
            existing = await session.execute(
                select(FundingRateModel).where(
                    FundingRateModel.symbol == symbol,
                    FundingRateModel.exchange == exchange,
                    FundingRateModel.timestamp == timestamp,
                )
            )
            if existing.scalar_one_or_none() is not None:
                return

            model = FundingRateModel(
                symbol=symbol,
                exchange=exchange,
                timestamp=timestamp,
                funding_rate=funding_rate,
                next_funding_time=next_funding_time,
            )
            session.add(model)
            await session.commit()

    async def get_funding_rates(
        self, symbol: str, exchange: str, limit: int = 50
    ) -> list[dict]:
        """获取资金费率历史"""
        async with self._session_factory() as session:
            result = await session.execute(
                select(FundingRateModel)
                .where(
                    FundingRateModel.symbol == symbol,
                    FundingRateModel.exchange == exchange,
                )
                .order_by(desc(FundingRateModel.timestamp))
                .limit(limit)
            )
            rows = result.scalars().all()

        return [
            {
                "timestamp": r.timestamp,
                "funding_rate": r.funding_rate,
                "next_funding_time": r.next_funding_time,
            }
            for r in reversed(rows)
        ]

    # ─── Signal 操作 ───

    async def save_signal(
        self,
        signal: Signal,
        approved: bool = True,
        reject_reason: str = "",
    ) -> None:
        """保存信号记录"""
        async with self._session_factory() as session:
            model = SignalModel(
                symbol=signal.symbol,
                exchange=signal.exchange,
                strategy=signal.strategy,
                action=signal.action.value,
                price=signal.price,
                strength=signal.strength,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                metadata_json=json.dumps(signal.metadata) if signal.metadata else None,
                approved=1 if approved else 0,
                reject_reason=reject_reason,
                timestamp=signal.timestamp,
            )
            session.add(model)
            await session.commit()

    async def get_peak_equity(self, exchange: str) -> float:
        """获取历史最高净值（CPPI 回撤控制用）"""
        async with self._session_factory() as session:
            from sqlalchemy import func
            result = await session.execute(
                select(func.max(AccountSnapshotModel.total_equity))
                .where(AccountSnapshotModel.exchange == exchange)
            )
            peak = result.scalar()
            return peak or 0.0
