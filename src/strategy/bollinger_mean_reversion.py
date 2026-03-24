"""
布林带均值回归策略

进场条件：
- 价格突破布林带上轨 + RSI 超买 → 做空（预期回归中轨）
- 价格跌破布林带下轨 + RSI 超卖 → 做多（预期回归中轨）
- ADX < 阈值时才交易（强趋势市不适合均值回归）

出场条件：
- 止盈：价格回归到布林带中轨
- 止损：ATR 倍数
- 时间止损

参考：config/strategies/bollinger_mean_reversion.yaml
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.core.event import Bar, Signal, SignalAction
from src.strategy.base import BaseStrategy


class BollingerMeanReversionStrategy(BaseStrategy):
    """布林带均值回归策略 — 震荡市专用"""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

        bb_cfg = config.get("bollinger", {})
        self._bb_period = bb_cfg.get("period", 20)
        self._bb_std = bb_cfg.get("std_dev", 2.0)

        rsi_cfg = config.get("rsi", {})
        self._rsi_period = rsi_cfg.get("period", 14)
        self._rsi_overbought = rsi_cfg.get("overbought", 70)
        self._rsi_oversold = rsi_cfg.get("oversold", 30)

        adx_cfg = config.get("adx_filter", {})
        self._adx_period = adx_cfg.get("period", 14)
        self._max_adx = adx_cfg.get("max_adx", 30.0)

        exit_cfg = config.get("exit", {})
        self._atr_period = exit_cfg.get("atr_period", 14)
        self._sl_atr = exit_cfg.get("stop_loss_atr", 3.0)
        self._tp_ratio = exit_cfg.get("take_profit_ratio", 0.6)  # TP = 回归中轨距离的 60%
        self._max_holding = exit_cfg.get("max_holding_bars", 36)

    @property
    def required_bars(self) -> int:
        return max(self._bb_period, self._rsi_period, self._adx_period * 2) + 50

    async def on_bar(
        self, bar: Bar, history: list[Bar]
    ) -> Optional[Signal]:
        df = self.bars_to_dataframe(history)

        if len(df) < self.required_bars:
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ─── 布林带 ───
        bb_ma = close.rolling(self._bb_period).mean()
        bb_std = close.rolling(self._bb_period).std()
        bb_upper = bb_ma + self._bb_std * bb_std
        bb_lower = bb_ma - self._bb_std * bb_std

        # ─── RSI ───
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self._rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self._rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # ─── ATR ───
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(self._atr_period).mean()

        # ─── ADX ───
        adx = self._compute_adx(df, self._adx_period)

        # ─── 获取最新值 ───
        curr_close = float(close.iloc[-1])
        curr_upper = float(bb_upper.iloc[-1])
        curr_lower = float(bb_lower.iloc[-1])
        curr_ma = float(bb_ma.iloc[-1])
        curr_std = float(bb_std.iloc[-1])
        curr_atr = float(atr.iloc[-1])
        curr_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
        curr_adx = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0

        if np.isnan(curr_upper) or np.isnan(curr_atr) or curr_atr <= 0:
            return None

        # ─── ADX 过滤：强趋势市不交易 ───
        if curr_adx > self._max_adx:
            return None

        # ─── 均值回归信号 ───
        signal_action = None
        stop_loss = None
        take_profit = None

        # 价格 > 上轨 + RSI 超买 → 做空（预期回归）
        if curr_close > curr_upper and curr_rsi > self._rsi_overbought:
            signal_action = SignalAction.OPEN_SHORT
            stop_loss = curr_close + self._sl_atr * curr_atr
            # TP = 当前价向中轨回归 tp_ratio 的距离（比中轨更近，更容易触及）
            take_profit = curr_close - self._tp_ratio * (curr_close - curr_ma)

        # 价格 < 下轨 + RSI 超卖 → 做多（预期回归）
        elif curr_close < curr_lower and curr_rsi < self._rsi_oversold:
            signal_action = SignalAction.OPEN_LONG
            stop_loss = curr_close - self._sl_atr * curr_atr
            take_profit = curr_close + self._tp_ratio * (curr_ma - curr_close)

        if signal_action is None:
            return None

        # ─── 信号强度 ───
        # RSI 极端程度 × 价格偏离程度
        if curr_std > 0:
            z_score = abs(curr_close - curr_ma) / curr_std
            rsi_extremity = abs(curr_rsi - 50) / 50  # 0~1
            strength = min(z_score / 3.0 * rsi_extremity * 2, 1.0)
        else:
            strength = 0.5

        return Signal(
            symbol=bar.symbol,
            exchange=bar.exchange,
            strategy=self.name,
            action=signal_action,
            price=curr_close,
            strength=max(strength, 0.3),
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "bb_upper": curr_upper,
                "bb_lower": curr_lower,
                "bb_ma": curr_ma,
                "rsi": curr_rsi,
                "adx": curr_adx,
                "atr": curr_atr,
                "type": "mean_reversion",
                "max_holding_bars": self._max_holding,
            },
        )

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)

        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
        adx = dx.ewm(alpha=1 / period, min_periods=period).mean()

        return adx
