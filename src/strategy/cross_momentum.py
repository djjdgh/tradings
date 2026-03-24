"""
跨币种动量轮动策略

核心逻辑：
- 对一组加密货币计算动量分数 = annualized_slope × R²
- 每周排序，做多 Top N（趋势最强的币种）
- R² 过滤确保趋势平滑（排除暴涨暴跌）

参考：
- Liu, Tsyvinski, Wu (2022) — Common Risk Factors in Cryptocurrency
- Clenow (2015) — Stocks on the Move
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger

from src.core.event import Bar, Signal, SignalAction
from src.strategy.base import BaseStrategy


class CrossMomentumStrategy(BaseStrategy):
    """跨币种动量轮动策略"""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

        self._lookback_days = config.get("lookback_days", 30)
        self._top_n = config.get("top_n", 3)
        self._min_r_squared = config.get("min_r_squared", 0.6)
        self._atr_period = config.get("atr_period", 14)
        self._sl_atr = config.get("stop_loss_atr", 3.0)
        self._tp_atr = config.get("take_profit_atr", 6.0)

        # 存储各币种的动量分数
        self._momentum_scores: dict[str, dict] = {}

    @property
    def required_bars(self) -> int:
        return self._lookback_days + 20

    async def on_bar(
        self, bar: Bar, history: list[Bar]
    ) -> Optional[Signal]:
        """
        计算动量分数并生成信号

        每次K线闭合时更新该品种的动量分数。
        信号生成基于该品种在所有已跟踪品种中的排名。
        """
        df = self.bars_to_dataframe(history)

        if len(df) < self._lookback_days:
            return None

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # ─── 计算动量分数 ───
        score = self._compute_momentum_score(
            close[-self._lookback_days:]
        )

        self._momentum_scores[bar.symbol] = {
            "score": score["score"],
            "slope": score["slope"],
            "r_squared": score["r_squared"],
            "price": float(close[-1]),
        }

        # ─── 排名 ───
        # 只有当我们有足够多的品种数据时才生成信号
        if len(self._momentum_scores) < 2:
            return None

        # 按分数排序
        ranked = sorted(
            self._momentum_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True,
        )

        # 找到当前品种的排名
        rank = next(
            (i for i, (sym, _) in enumerate(ranked) if sym == bar.symbol),
            None,
        )

        if rank is None:
            return None

        curr_close = float(close[-1])
        curr_score = self._momentum_scores[bar.symbol]

        # R² 过滤
        if curr_score["r_squared"] < self._min_r_squared:
            return None

        # ATR
        atr = self._compute_atr(high, low, close, self._atr_period)
        if atr <= 0:
            return None

        # ─── 信号生成 ───
        total_symbols = len(self._momentum_scores)

        # Top N → 做多
        if rank < self._top_n and curr_score["slope"] > 0:
            strength = curr_score["r_squared"]  # R² 作为信号强度

            return Signal(
                symbol=bar.symbol,
                exchange=bar.exchange,
                strategy=self.name,
                action=SignalAction.OPEN_LONG,
                price=curr_close,
                strength=strength,
                stop_loss=curr_close - self._sl_atr * atr,
                take_profit=curr_close + self._tp_atr * atr,
                metadata={
                    "rank": rank + 1,
                    "total": total_symbols,
                    "momentum_score": curr_score["score"],
                    "slope": curr_score["slope"],
                    "r_squared": curr_score["r_squared"],
                },
            )

        return None

    @staticmethod
    def _compute_momentum_score(prices: np.ndarray) -> dict:
        """
        计算动量分数 = annualized_slope × R²

        使用 log-price 的线性回归
        """
        if len(prices) < 5:
            return {"score": 0, "slope": 0, "r_squared": 0}

        log_prices = np.log(prices)
        x = np.arange(len(log_prices))

        # 线性回归
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(log_prices)
        sum_xy = np.sum(x * log_prices)
        sum_x2 = np.sum(x * x)

        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            return {"score": 0, "slope": 0, "r_squared": 0}

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        # R²
        y_pred = slope * x + intercept
        ss_res = np.sum((log_prices - y_pred) ** 2)
        ss_tot = np.sum((log_prices - np.mean(log_prices)) ** 2)

        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # 年化斜率（假设日线数据，365天）
        annualized_slope = slope * 365

        # 动量分数
        score = annualized_slope * r_squared

        return {
            "score": float(score),
            "slope": float(annualized_slope),
            "r_squared": float(max(0, r_squared)),
        }

    @staticmethod
    def _compute_atr(high, low, close, period: int) -> float:
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
