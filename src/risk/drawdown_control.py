"""
CPPI 回撤控制 — 动态风险预算

当回撤加深时系统性缩减仓位，保护本金。

参考：Grossman, Zhou (1993) — CPPI Framework
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass
class DrawdownConfig:
    """回撤控制配置"""
    max_drawdown_pct: float = 0.35    # 最大回撤容忍度 (35%)
    cppi_multiplier: float = 3.0      # CPPI 乘数
    warning_drawdown_pct: float = 0.20  # 预警回撤 (20%)
    critical_drawdown_pct: float = 0.30 # 危险回撤 (30%)


class DrawdownControl:
    """
    CPPI 回撤控制器

    exposure_multiplier = min(m × cushion / equity, 1.0)

    其中：
    - cushion = equity - floor
    - floor = peak × (1 - max_drawdown)
    - m = CPPI 乘数
    """

    def __init__(self, config: DrawdownConfig, initial_capital: float = 0.0):
        self.config = config
        self._peak_equity = initial_capital
        self._initial_capital = initial_capital

    def update_equity(self, equity: float) -> None:
        """更新当前净值，追踪峰值"""
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._initial_capital <= 0:
            self._initial_capital = equity

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def floor(self) -> float:
        """最低可接受净值"""
        return self._peak_equity * (1 - self.config.max_drawdown_pct)

    def current_drawdown(self, equity: float) -> float:
        """当前回撤幅度 (0-1)"""
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, 1 - equity / self._peak_equity)

    def exposure_multiplier(self, equity: float) -> float:
        """
        计算仓位暴露乘数 (0-1)

        所有策略的仓位 × 这个乘数。
        回撤越深，乘数越小。
        """
        if self._peak_equity <= 0 or equity <= 0:
            return 1.0

        cushion = equity - self.floor

        if cushion <= 0:
            # 回撤已超过阈值，完全停止
            logger.critical(
                f"Drawdown exceeded max! equity={equity:.2f}, "
                f"floor={self.floor:.2f}"
            )
            return 0.0

        multiplier = self.config.cppi_multiplier * cushion / equity
        return min(multiplier, 1.0)

    def check_drawdown(self, equity: float) -> dict:
        """
        检查回撤状态

        Returns:
            {"level": "normal"|"warning"|"critical"|"halt",
             "drawdown": float,
             "multiplier": float}
        """
        self.update_equity(equity)
        dd = self.current_drawdown(equity)
        mult = self.exposure_multiplier(equity)

        if dd >= self.config.max_drawdown_pct:
            level = "halt"
        elif dd >= self.config.critical_drawdown_pct:
            level = "critical"
        elif dd >= self.config.warning_drawdown_pct:
            level = "warning"
        else:
            level = "normal"

        return {
            "level": level,
            "drawdown": dd,
            "multiplier": mult,
            "peak_equity": self._peak_equity,
            "floor": self.floor,
        }

    def set_peak_equity(self, peak: float) -> None:
        """手动设置峰值净值（从数据库恢复时用）"""
        self._peak_equity = peak
