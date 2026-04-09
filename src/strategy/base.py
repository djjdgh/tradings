"""
策略抽象基类 — 定义策略统一接口

匹配 engine.py 中的调用：
- strategy.should_process(symbol, timeframe) -> bool
- strategy.on_bar(bar, bars) -> Optional[Signal]
- strategy.required_bars -> int
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from src.core.event import Bar, Signal, Tick


class BaseStrategy(ABC):
    """策略抽象基类"""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self._symbols: set[str] = set()
        self._timeframes: set[str] = set()

    @property
    def required_bars(self) -> int:
        """策略需要的历史K线数量"""
        return 200

    def should_process(self, symbol: str, timeframe: str) -> bool:
        """
        判断是否处理此品种和周期

        engine.py 在 _on_bar 中调用此方法来分派K线到对应策略
        """
        if self._symbols and symbol not in self._symbols:
            return False
        if self._timeframes and timeframe not in self._timeframes:
            return False
        return True

    def add_symbol(self, symbol: str) -> None:
        self._symbols.add(symbol)

    def add_timeframe(self, timeframe: str) -> None:
        self._timeframes.add(timeframe)

    @abstractmethod
    async def on_bar(
        self, bar: Bar, history: list[Bar]
    ) -> Optional[Signal]:
        """
        K线闭合时调用，返回交易信号或 None

        Args:
            bar: 最新闭合的K线
            history: 历史K线列表（按时间正序）

        Returns:
            Signal 或 None
        """
        ...

    async def on_tick(self, tick: Tick) -> None:
        """实时价格更新（可选覆写）"""
        pass

    @staticmethod
    def bars_to_dataframe(bars: list[Bar]) -> pd.DataFrame:
        """将K线列表转换为 DataFrame"""
        if not bars:
            return pd.DataFrame()

        data = {
            "timestamp": [b.timestamp for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
