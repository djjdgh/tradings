"""
资金费率反向策略

核心逻辑（纯合约端，适合小资金）：
- 监控永续合约的资金费率（每 8 小时结算）
- 当 funding rate 极端偏高 → 做空（收取费率 + 预期价格回归）
- 当 funding rate 极端偏低/负 → 做多（收取费率 + 预期价格回归）
- 本质：资金费率作为情绪反转信号

参考：Perpetual Futures Funding Rate Research (Deribit/Paradigm)
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from src.core.event import Bar, Signal, SignalAction
from src.strategy.base import BaseStrategy


class FundingRateStrategy(BaseStrategy):
    """资金费率反向策略"""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

        self._high_threshold = config.get("extreme_high_threshold", 0.0005)
        self._low_threshold = config.get("extreme_low_threshold", -0.0003)
        self._exit_threshold = config.get("exit_threshold", 0.0001)
        self._max_position_pct = config.get("max_position_pct", 0.25)
        self._max_holding_periods = config.get("max_holding_periods", 6)
        self._atr_period = config.get("atr_period", 14)
        self._sl_atr = config.get("stop_loss_atr", 2.5)

        # 内部状态
        self._current_funding_rate: dict[str, float] = {}  # symbol -> rate
        self._entry_funding_rate: dict[str, float] = {}    # symbol -> rate at entry

    @property
    def required_bars(self) -> int:
        return 50

    def update_funding_rate(self, symbol: str, rate: float) -> None:
        """外部调用更新资金费率"""
        self._current_funding_rate[symbol] = rate

    async def on_bar(
        self, bar: Bar, history: list[Bar]
    ) -> Optional[Signal]:
        """
        K线闭合时检查资金费率信号

        注意：资金费率数据需要通过 update_funding_rate() 预先更新
        """
        symbol = bar.symbol
        rate = self._current_funding_rate.get(symbol)

        if rate is None:
            return None

        df = self.bars_to_dataframe(history)
        if len(df) < self._atr_period + 1:
            return None

        # 计算 ATR
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        atr = self._compute_atr(high, low, close, self._atr_period)

        if atr <= 0:
            return None

        curr_close = float(close[-1])

        # ─── 开仓信号 ───

        # 费率极端偏高 → 做空（市场过度看多）
        if rate >= self._high_threshold:
            strength = min(abs(rate) / self._high_threshold, 1.0)
            self._entry_funding_rate[symbol] = rate

            return Signal(
                symbol=symbol,
                exchange=bar.exchange,
                strategy=self.name,
                action=SignalAction.OPEN_SHORT,
                price=curr_close,
                strength=strength * self._max_position_pct,
                stop_loss=curr_close + self._sl_atr * atr,
                take_profit=curr_close - self._sl_atr * atr * 1.5,
                metadata={
                    "funding_rate": rate,
                    "signal_type": "extreme_high",
                },
            )

        # 费率极端偏低/负值 → 做多（市场过度看空）
        if rate <= self._low_threshold:
            strength = min(abs(rate) / abs(self._low_threshold), 1.0)
            self._entry_funding_rate[symbol] = rate

            return Signal(
                symbol=symbol,
                exchange=bar.exchange,
                strategy=self.name,
                action=SignalAction.OPEN_LONG,
                price=curr_close,
                strength=strength * self._max_position_pct,
                stop_loss=curr_close - self._sl_atr * atr,
                take_profit=curr_close + self._sl_atr * atr * 1.5,
                metadata={
                    "funding_rate": rate,
                    "signal_type": "extreme_low",
                },
            )

        # ─── 平仓信号（费率回归正常） ───
        entry_rate = self._entry_funding_rate.get(symbol)
        if entry_rate is not None:
            # 如果之前做空（因为费率高），现在费率回到正常 → 平仓
            if entry_rate >= self._high_threshold and abs(rate) <= self._exit_threshold:
                del self._entry_funding_rate[symbol]
                return Signal(
                    symbol=symbol,
                    exchange=bar.exchange,
                    strategy=self.name,
                    action=SignalAction.CLOSE_SHORT,
                    price=curr_close,
                    metadata={"funding_rate": rate, "signal_type": "exit_normalized"},
                )

            # 如果之前做多（因为费率低），现在费率回到正常 → 平仓
            if entry_rate <= self._low_threshold and abs(rate) <= self._exit_threshold:
                del self._entry_funding_rate[symbol]
                return Signal(
                    symbol=symbol,
                    exchange=bar.exchange,
                    strategy=self.name,
                    action=SignalAction.CLOSE_LONG,
                    price=curr_close,
                    metadata={"funding_rate": rate, "signal_type": "exit_normalized"},
                )

        return None

    @staticmethod
    def _compute_atr(
        high, low, close, period: int
    ) -> float:
        """计算 ATR"""
        if len(high) < period + 1:
            return 0.0

        tr_list = []
        for i in range(1, len(high)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
            tr_list.append(tr)

        if len(tr_list) < period:
            return 0.0

        import numpy as np
        return float(np.mean(tr_list[-period:]))
