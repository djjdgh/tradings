"""
数据库模型 — SQLAlchemy ORM 定义
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, UniqueConstraint, Index,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class BarModel(Base):
    """K线数据"""
    __tablename__ = "bars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(50), nullable=False)
    exchange = Column(String(30), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(Float, nullable=False)  # Unix 毫秒
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("symbol", "exchange", "timeframe", "timestamp",
                         name="uq_bar"),
        Index("ix_bars_lookup", "symbol", "exchange", "timeframe", "timestamp"),
    )


class OrderModel(Base):
    """订单记录"""
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(64), nullable=False, unique=True)
    symbol = Column(String(50), nullable=False)
    exchange = Column(String(30), nullable=False)
    side = Column(String(10), nullable=False)       # long / short
    order_type = Column(String(10), nullable=False)  # market / limit
    amount = Column(Float, nullable=False)
    price = Column(Float)
    status = Column(String(20), nullable=False)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    strategy = Column(String(50), default="")
    filled_amount = Column(Float, default=0.0)
    filled_price = Column(Float, default=0.0)
    fee = Column(Float, default=0.0)
    exchange_order_id = Column(String(64))
    timestamp = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_orders_symbol", "symbol", "exchange"),
    )


class TradeModel(Base):
    """成交记录"""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(50), nullable=False)
    exchange = Column(String(30), nullable=False)
    side = Column(String(10), nullable=False)
    amount = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    realized_pnl = Column(Float, default=0.0)
    fee = Column(Float, default=0.0)
    strategy = Column(String(50), default="")
    entry_time = Column(Float, nullable=False)
    exit_time = Column(Float)
    bars_held = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class AccountSnapshotModel(Base):
    """账户净值快照"""
    __tablename__ = "account_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    exchange = Column(String(30), nullable=False)
    timestamp = Column(Float, nullable=False)
    total_equity = Column(Float, nullable=False)
    available_balance = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0.0)
    margin_used = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_snapshots_time", "exchange", "timestamp"),
    )


class FundingRateModel(Base):
    """资金费率历史"""
    __tablename__ = "funding_rates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(50), nullable=False)
    exchange = Column(String(30), nullable=False)
    timestamp = Column(Float, nullable=False)
    funding_rate = Column(Float, nullable=False)
    next_funding_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("symbol", "exchange", "timestamp",
                         name="uq_funding_rate"),
        Index("ix_funding_lookup", "symbol", "exchange", "timestamp"),
    )


class SignalModel(Base):
    """信号记录（回溯分析用）"""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(50), nullable=False)
    exchange = Column(String(30), nullable=False)
    strategy = Column(String(50), nullable=False)
    action = Column(String(20), nullable=False)
    price = Column(Float, nullable=False)
    strength = Column(Float, default=1.0)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    metadata_json = Column(Text)       # JSON 序列化的额外数据
    approved = Column(Integer, default=1)  # 是否被风控批准
    reject_reason = Column(String(200))
    timestamp = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_signals_lookup", "symbol", "strategy", "timestamp"),
    )
