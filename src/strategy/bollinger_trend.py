"""
布林带趋势突破策略

进场条件：
- 价格突破布林带上轨 + 成交量放大 → 做多
- 价格跌破布林带下轨 + 成交量放大 → 做空
- 大周期趋势方向过滤

出场条件：
- ATR 追踪止盈
- ATR 固定止损
- 时间止损

参考：config/strategies/bollinger_trend.yaml
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.core.event import Bar, Signal, SignalAction
from src.strategy.base import BaseStrategy


class BollingerTrendStrategy(BaseStrategy):
    """布林带趋势突破策略"""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

        bb_cfg = config.get("bollinger", {})
        self._bb_period = bb_cfg.get("period", 20)
        self._bb_std = bb_cfg.get("std_dev", 2.0)

        vol_cfg = config.get("volume", {})
        self._vol_enabled = vol_cfg.get("enabled", True)
        self._vol_period = vol_cfg.get("period", 20)
        self._vol_threshold = vol_cfg.get("threshold", 1.5)

        trend_cfg = config.get("trend_filter", {})
        self._trend_enabled = trend_cfg.get("enabled", True)
        self._trend_ma_period = trend_cfg.get("ma_period", 50)

        exit_cfg = config.get("exit", {})
        self._atr_period = exit_cfg.get("atr_period", 14)
        self._sl_atr = exit_cfg.get("stop_loss_atr", 2.0)
        self._trailing_atr = exit_cfg.get("trailing_stop_atr", 1.5)
        self._max_holding = exit_cfg.get("max_holding_bars", 48)
        self._tp_atr = exit_cfg.get("take_profit_atr", 4.0)

    @property
    def required_bars(self) -> int:
        return max(self._bb_period, self._trend_ma_period, self._vol_period) + 50

    async def on_bar(
        self, bar: Bar, history: list[Bar]
    ) -> Optional[Signal]:
        """计算布林带突破信号"""
        df = self.bars_to_dataframe(history)

        if len(df) < self.required_bars:
            return None

        # ─── 计算指标 ───
        close = df["close"]
        volume = df["volume"]

        # 布林带
        bb_ma = close.rolling(self._bb_period).mean()
        bb_std = close.rolling(self._bb_period).std()
        bb_upper = bb_ma + self._bb_std * bb_std
        bb_lower = bb_ma - self._bb_std * bb_std

        # ATR
        high = df["high"]
        low = df["low"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self._atr_period).mean()

        # 成交量均值
        vol_ma = volume.rolling(self._vol_period).mean()

        # 大周期趋势 MA
        trend_ma = close.rolling(self._trend_ma_period).mean()

        # ─── 获取最新值 ───
        curr_close = float(close.iloc[-1])
        curr_upper = float(bb_upper.iloc[-1])
        curr_lower = float(bb_lower.iloc[-1])
        curr_atr = float(atr.iloc[-1])
        curr_vol = float(volume.iloc[-1])
        curr_vol_ma = float(vol_ma.iloc[-1])
        curr_trend_ma = float(trend_ma.iloc[-1])

        if np.isnan(curr_upper) or np.isnan(curr_atr) or curr_atr <= 0:
            return None

        # ─── 成交量确认 ───
        vol_confirmed = True
        if self._vol_enabled and curr_vol_ma > 0:
            vol_confirmed = curr_vol > curr_vol_ma * self._vol_threshold

        if not vol_confirmed:
            return None

        # ─── 信号判断 ───
        signal_action = None

        # 突破上轨 → 做多
        if curr_close > curr_upper:
            if self._trend_enabled and curr_close < curr_trend_ma:
                return None  # 大周期趋势向下，不做多
            signal_action = SignalAction.OPEN_LONG
            stop_loss = curr_close - self._sl_atr * curr_atr
            take_profit = curr_close + self._tp_atr * curr_atr

        # 跌破下轨 → 做空
        elif curr_close < curr_lower:
            if self._trend_enabled and curr_close > curr_trend_ma:
                return None  # 大周期趋势向上，不做空
            signal_action = SignalAction.OPEN_SHORT
            stop_loss = curr_close + self._sl_atr * curr_atr
            take_profit = curr_close - self._tp_atr * curr_atr

        if signal_action is None:
            return None

        # ─── 计算信号强度 ───
        # 价格偏离布林带中轨的程度
        if bb_std.iloc[-1] > 0:
            z_score = abs(curr_close - float(bb_ma.iloc[-1])) / float(bb_std.iloc[-1])
            strength = min(z_score / 3.0, 1.0)  # z-score 3 时强度为 1.0
        else:
            strength = 0.5

        return Signal(
            symbol=bar.symbol,
            exchange=bar.exchange,
            strategy=self.name,
            action=signal_action,
            price=curr_close,
            strength=strength,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "bb_upper": curr_upper,
                "bb_lower": curr_lower,
                "atr": curr_atr,
                "volume_ratio": curr_vol / curr_vol_ma if curr_vol_ma > 0 else 0,
            },
        )
