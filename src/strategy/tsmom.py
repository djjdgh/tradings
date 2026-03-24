"""
多尺度趋势跟踪策略 (TSMOM)

核心逻辑：
- 计算多个时间尺度的 EWMA 信号 (span = 8, 16, 32, 64 天)
- 每个尺度产生 [-1, +1] 信号
- 综合信号 = 各尺度信号的加权平均
- 信号 > 0 做多，< 0 做空，绝对值决定信号强度

参考：
- Moskowitz, Ooi, Pedersen (2012) — Time Series Momentum
- Lemperiere, Deremble, Bouchaud (2014) — Two Centuries of Trend Following
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger

from src.core.event import Bar, Signal, SignalAction
from src.strategy.base import BaseStrategy


class TSMOMStrategy(BaseStrategy):
    """多尺度时序动量策略"""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

        self._ewma_spans = config.get("ewma_spans", [8, 16, 32, 64])
        self._weights = config.get("signal_weights", None)
        self._atr_period = config.get("atr_period", 14)
        self._sl_atr = config.get("stop_loss_atr", 3.0)
        self._tp_atr = config.get("take_profit_atr", 6.0)
        self._signal_threshold = config.get("signal_threshold", 0.1)

        # 默认等权
        if self._weights is None:
            n = len(self._ewma_spans)
            self._weights = [1.0 / n] * n

    @property
    def required_bars(self) -> int:
        return max(self._ewma_spans) * 2 + 20

    async def on_bar(
        self, bar: Bar, history: list[Bar]
    ) -> Optional[Signal]:
        """计算多尺度趋势信号"""
        df = self.bars_to_dataframe(history)

        if len(df) < self.required_bars:
            return None

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # ─── 计算各尺度 EWMA 信号 ───
        signals = []
        for span in self._ewma_spans:
            signal_val = self._compute_ewma_signal(close, span)
            signals.append(signal_val)

        # ─── 加权综合信号 ───
        combined_signal = sum(
            s * w for s, w in zip(signals, self._weights)
        )

        # 信号强度 = |combined_signal|，归一化到 [0, 1]
        strength = min(abs(combined_signal), 1.0)

        # 信号阈值过滤
        if abs(combined_signal) < self._signal_threshold:
            return None

        # ─── ATR 止损止盈 ───
        curr_close = float(close[-1])
        atr = self._compute_atr(high, low, close, self._atr_period)

        if atr <= 0:
            return None

        # ─── 生成信号 ───
        if combined_signal > 0:
            action = SignalAction.OPEN_LONG
            stop_loss = curr_close - self._sl_atr * atr
            take_profit = curr_close + self._tp_atr * atr
        else:
            action = SignalAction.OPEN_SHORT
            stop_loss = curr_close + self._sl_atr * atr
            take_profit = curr_close - self._tp_atr * atr

        return Signal(
            symbol=bar.symbol,
            exchange=bar.exchange,
            strategy=self.name,
            action=action,
            price=curr_close,
            strength=strength,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "combined_signal": combined_signal,
                "individual_signals": dict(zip(
                    [str(s) for s in self._ewma_spans], signals
                )),
                "atr": atr,
            },
        )

    @staticmethod
    def _compute_ewma_signal(prices: np.ndarray, span: int) -> float:
        """
        计算单尺度 EWMA 趋势信号

        signal = (price - EWMA) / std(price - EWMA)
        归一化到 [-1, +1] 范围

        使用 EWMA crossover: 快速 EWMA (span) vs 慢速 EWMA (2×span)
        """
        if len(prices) < span * 2:
            return 0.0

        alpha_fast = 2.0 / (span + 1)
        alpha_slow = 2.0 / (span * 2 + 1)

        # 计算 EWMA
        ewma_fast = prices[0]
        ewma_slow = prices[0]

        for p in prices[1:]:
            ewma_fast = alpha_fast * p + (1 - alpha_fast) * ewma_fast
            ewma_slow = alpha_slow * p + (1 - alpha_slow) * ewma_slow

        # 信号 = 快速 - 慢速，归一化
        diff = ewma_fast - ewma_slow
        # 用价格的标准差归一化
        price_std = float(np.std(prices[-span:]))
        if price_std <= 0:
            return 0.0

        normalized = diff / price_std

        # 限制到 [-1, +1]
        return max(-1.0, min(1.0, normalized))

    @staticmethod
    def _compute_atr(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int,
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

        return float(np.mean(tr_list[-period:]))
