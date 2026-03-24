"""
波动率估计 — EWMA + HAR-RV 双模型

参考：
- Brownlees, Engle, Kelly (2011) — EWMA
- Andersen, Bollerslev, Diebold, Labys (2003) — HAR-RV
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VolatilityEstimator:
    """
    双速 EWMA 波动率估计器

    快速 EWMA（λ=0.94）：响应迅速，用于紧急减仓判断
    慢速 EWMA（λ=0.97）：平滑稳定，用于常规仓位计算
    """

    lambda_fast: float = 0.94
    lambda_slow: float = 0.97
    min_periods: int = 10          # 最少需要的观测数
    annualization_factor: float = 365.0  # 加密市场 365 天

    # 内部状态
    _var_fast: float = 0.0
    _var_slow: float = 0.0
    _count: int = 0
    _prev_price: float = 0.0
    _returns: list[float] = field(default_factory=list)

    def update(self, price: float) -> None:
        """
        用新价格更新波动率估计

        Args:
            price: 最新价格（通常是K线收盘价）
        """
        if self._prev_price > 0:
            ret = math.log(price / self._prev_price)
            self._returns.append(ret)

            if self._count == 0:
                # 初始化：用第一个收益率的平方
                self._var_fast = ret * ret
                self._var_slow = ret * ret
            else:
                self._var_fast = (
                    self.lambda_fast * self._var_fast
                    + (1 - self.lambda_fast) * ret * ret
                )
                self._var_slow = (
                    self.lambda_slow * self._var_slow
                    + (1 - self.lambda_slow) * ret * ret
                )

            self._count += 1

            # 只保留最近 200 个收益率（HAR-RV 用）
            if len(self._returns) > 200:
                self._returns = self._returns[-200:]

        self._prev_price = price

    @property
    def is_ready(self) -> bool:
        """是否有足够的数据"""
        return self._count >= self.min_periods

    @property
    def daily_vol_fast(self) -> float:
        """快速 EWMA 日波动率"""
        if not self.is_ready:
            return 0.0
        return math.sqrt(self._var_fast)

    @property
    def daily_vol_slow(self) -> float:
        """慢速 EWMA 日波动率"""
        if not self.is_ready:
            return 0.0
        return math.sqrt(self._var_slow)

    @property
    def annual_vol_fast(self) -> float:
        """快速 EWMA 年化波动率"""
        return self.daily_vol_fast * math.sqrt(self.annualization_factor)

    @property
    def annual_vol_slow(self) -> float:
        """慢速 EWMA 年化波动率"""
        return self.daily_vol_slow * math.sqrt(self.annualization_factor)

    @property
    def annual_vol(self) -> float:
        """主要使用的年化波动率（慢速 EWMA）"""
        return self.annual_vol_slow

    def vol_spike_ratio(self) -> float:
        """
        波动率飙升比率 = 快速 / 慢速

        > 2.0 表示波动率急剧上升（可能需要紧急减仓）
        """
        if self.daily_vol_slow <= 0:
            return 1.0
        return self.daily_vol_fast / self.daily_vol_slow

    def har_rv(self) -> float:
        """
        HAR-RV 波动率估计

        RV_t = c + β1×RV_daily + β2×RV_weekly + β3×RV_monthly

        使用简化版：直接计算不同时间窗口的已实现波动率并加权平均
        """
        if len(self._returns) < 22:
            return self.annual_vol

        returns = np.array(self._returns)

        # 日 RV（最近 1 天的收益率方差）
        rv_daily = float(np.var(returns[-1:])) if len(returns) >= 1 else 0
        # 周 RV（最近 5 天）
        rv_weekly = float(np.var(returns[-5:])) if len(returns) >= 5 else rv_daily
        # 月 RV（最近 22 天）
        rv_monthly = float(np.var(returns[-22:])) if len(returns) >= 22 else rv_weekly

        # 简化 HAR 权重（等权）
        rv_combined = (rv_daily + rv_weekly + rv_monthly) / 3
        daily_vol = math.sqrt(rv_combined) if rv_combined > 0 else 0

        return daily_vol * math.sqrt(self.annualization_factor)

    def reset(self) -> None:
        """重置状态"""
        self._var_fast = 0.0
        self._var_slow = 0.0
        self._count = 0
        self._prev_price = 0.0
        self._returns = []
