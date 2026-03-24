"""
熔断机制 — 极端情况下的自动保护

多层防护：
1. 单日最大亏损
2. 连续止损次数
3. 波动率飙升
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class CircuitBreakerConfig:
    """熔断配置"""
    max_daily_loss_pct: float = 0.08      # 单日最大亏损 (8%)
    consecutive_loss_limit: int = 5        # 连续止损次数限制
    cooldown_seconds: float = 14400.0      # 熔断冷却时间 (4小时)
    vol_spike_threshold: float = 3.0       # 波动率飙升阈值 (快/慢 > 3x)
    force_close_on_spike: bool = True      # 波动率飙升时强制平仓


class CircuitBreaker:
    """
    熔断器

    触发条件任一满足即熔断：
    1. 当日累计亏损超过阈值
    2. 连续止损次数超过限制
    3. 波动率短期飙升超过阈值
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self._tripped = False
        self._trip_time: float = 0.0
        self._trip_reason: str = ""
        self._consecutive_losses: int = 0
        self._daily_pnl: float = 0.0
        self._daily_start_equity: float = 0.0
        self._last_reset_day: str = ""

    @property
    def is_tripped(self) -> bool:
        """是否处于熔断状态"""
        if not self._tripped:
            return False

        # 检查冷却时间是否已过
        if time.time() - self._trip_time > self.config.cooldown_seconds:
            self._tripped = False
            self._trip_reason = ""
            logger.info("Circuit breaker cooldown expired, resuming trading")
            return False

        return True

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    def reset_daily(self, equity: float) -> None:
        """每日重置（在新的交易日开始时调用）"""
        today = time.strftime("%Y-%m-%d")
        if today != self._last_reset_day:
            self._daily_pnl = 0.0
            self._daily_start_equity = equity
            self._last_reset_day = today

    def on_trade_result(self, pnl: float, equity: float) -> bool:
        """
        记录交易结果

        Args:
            pnl: 本次交易盈亏
            equity: 当前净值

        Returns:
            是否触发熔断
        """
        self.reset_daily(equity)

        # 更新日内盈亏
        self._daily_pnl += pnl

        # 更新连续亏损计数
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # 检查单日最大亏损
        if self._daily_start_equity > 0:
            daily_loss_pct = abs(self._daily_pnl) / self._daily_start_equity
            if self._daily_pnl < 0 and daily_loss_pct >= self.config.max_daily_loss_pct:
                self._trip(
                    f"Daily loss limit hit: {daily_loss_pct:.1%} "
                    f"(limit: {self.config.max_daily_loss_pct:.1%})"
                )
                return True

        # 检查连续止损
        if self._consecutive_losses >= self.config.consecutive_loss_limit:
            self._trip(
                f"Consecutive losses: {self._consecutive_losses} "
                f"(limit: {self.config.consecutive_loss_limit})"
            )
            return True

        return False

    def check_vol_spike(self, vol_spike_ratio: float) -> bool:
        """
        检查波动率飙升

        Args:
            vol_spike_ratio: 快速波动率 / 慢速波动率

        Returns:
            是否触发熔断
        """
        if vol_spike_ratio >= self.config.vol_spike_threshold:
            self._trip(
                f"Volatility spike: {vol_spike_ratio:.1f}x "
                f"(threshold: {self.config.vol_spike_threshold:.1f}x)"
            )
            return True
        return False

    def _trip(self, reason: str) -> None:
        """触发熔断"""
        self._tripped = True
        self._trip_time = time.time()
        self._trip_reason = reason
        logger.critical(f"CIRCUIT BREAKER TRIPPED: {reason}")

    def manual_reset(self) -> None:
        """手动重置熔断器"""
        self._tripped = False
        self._trip_reason = ""
        self._consecutive_losses = 0
        logger.info("Circuit breaker manually reset")

    def status(self) -> dict:
        """获取熔断器状态"""
        remaining = 0.0
        if self._tripped:
            remaining = max(
                0,
                self.config.cooldown_seconds - (time.time() - self._trip_time)
            )

        return {
            "tripped": self.is_tripped,
            "reason": self._trip_reason,
            "consecutive_losses": self._consecutive_losses,
            "daily_pnl": self._daily_pnl,
            "cooldown_remaining_s": remaining,
        }
