"""
风控管理器 — 统一入口，协调仓位管理、回撤控制、熔断

匹配 engine.py 中的调用接口：
- evaluate_signal(signal, positions, account) -> RiskResult
- check_stop_loss(position, tick) -> Optional[Signal]
- check_account(account) -> Optional[RiskAlert]
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from src.core.event import (
    AccountSnapshot, Position, RiskAlert, Signal, SignalAction, Tick,
)
from src.risk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.risk.drawdown_control import DrawdownControl, DrawdownConfig
from src.risk.position_sizer import PositionSizer, PositionSizerConfig


@dataclass
class RiskResult:
    """风控评估结果"""
    approved: bool
    position_size: float = 0.0
    reason: str = ""


class RiskManager:
    """
    风控管理器

    多层防御：
    1. 熔断检查 — 极端情况直接拒绝
    2. 回撤检查 — CPPI 动态缩减仓位
    3. 仓位限制 — 波动率目标 + 风险限制
    4. 止盈止损 — 实时价格检查
    """

    def __init__(self, config: dict):
        risk_cfg = config.get("risk", {})
        vol_cfg = risk_cfg.get("volatility", {})
        cb_cfg = risk_cfg.get("circuit_breaker", {})

        # 仓位管理器
        self.position_sizer = PositionSizer(PositionSizerConfig(
            target_annual_vol=vol_cfg.get("target_annual_vol", 0.80),
            max_leverage=risk_cfg.get("max_leverage", 5.0),
            min_leverage=risk_cfg.get("min_leverage", 0.1),
            max_risk_per_trade_pct=risk_cfg.get("max_risk_per_trade_pct", 0.03),
            max_total_position_pct=risk_cfg.get("max_total_position_pct", 0.60),
        ))

        # 回撤控制器
        self.drawdown_control = DrawdownControl(DrawdownConfig(
            max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 0.35),
        ))

        # 熔断器
        self.circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
            max_daily_loss_pct=risk_cfg.get("max_daily_loss_pct", 0.08),
            consecutive_loss_limit=risk_cfg.get("consecutive_loss_limit", 5),
            cooldown_seconds=risk_cfg.get("cooldown_hours", 4) * 3600,
            vol_spike_threshold=cb_cfg.get("atr_spike_threshold", 3.0),
            force_close_on_spike=cb_cfg.get("force_close_on_spike", True),
        ))

        self._config = risk_cfg

    def evaluate_signal(
        self,
        signal: Signal,
        positions: dict[str, Position],
        account: Optional[AccountSnapshot],
    ) -> RiskResult:
        """
        评估交易信号

        Returns:
            RiskResult with approved=True/False and position_size
        """
        # ─── 平仓信号直接放行 ───
        if signal.action in (SignalAction.CLOSE_LONG, SignalAction.CLOSE_SHORT):
            pos = positions.get(signal.symbol)
            return RiskResult(
                approved=True,
                position_size=pos.amount if pos else 0.0,
            )

        # ─── 1. 熔断检查 ───
        if self.circuit_breaker.is_tripped:
            return RiskResult(
                approved=False,
                reason=f"Circuit breaker active: {self.circuit_breaker.trip_reason}",
            )

        # ─── 2. 账户信息检查 ───
        if account is None:
            return RiskResult(approved=False, reason="No account data")

        capital = account.available_balance
        if capital <= 0:
            return RiskResult(approved=False, reason="No available balance")

        # ─── 3. 回撤检查 ───
        self.drawdown_control.update_equity(account.total_equity)
        dd_status = self.drawdown_control.check_drawdown(account.total_equity)
        dd_multiplier = dd_status["multiplier"]

        if dd_status["level"] == "halt":
            return RiskResult(
                approved=False,
                reason=f"Max drawdown reached: {dd_status['drawdown']:.1%}",
            )

        # ─── 4. 已有持仓检查 ───
        if signal.symbol in positions:
            existing = positions[signal.symbol]
            # 不允许同一品种的同向加仓（简化处理）
            if (signal.action == SignalAction.OPEN_LONG and existing.side.value == "long") or \
               (signal.action == SignalAction.OPEN_SHORT and existing.side.value == "short"):
                return RiskResult(
                    approved=False,
                    reason=f"Already have {existing.side.value} position on {signal.symbol}",
                )

        # ─── 5. 总仓位检查 ───
        total_position_value = sum(
            p.amount * p.entry_price for p in positions.values()
        )
        max_position_value = account.total_equity * self._config.get(
            "max_total_position_pct", 0.60
        )
        if total_position_value >= max_position_value:
            return RiskResult(
                approved=False,
                reason=f"Total position limit reached: "
                       f"{total_position_value:.2f} >= {max_position_value:.2f}",
            )

        # ─── 6. 波动率飙升检查 ───
        vol_info = self.position_sizer.get_vol_info(signal.symbol)
        if vol_info.get("ready") and vol_info.get("vol_spike_ratio", 1.0) > 2.0:
            logger.warning(
                f"Elevated vol spike for {signal.symbol}: "
                f"{vol_info['vol_spike_ratio']:.1f}x"
            )
            # 不完全拒绝，但缩减仓位
            dd_multiplier *= 0.5

        # ─── 7. 计算仓位大小 ───
        stop_loss_distance = 0.0
        if signal.stop_loss and signal.price:
            stop_loss_distance = abs(signal.price - signal.stop_loss)

        raw_size = self.position_sizer.calculate_position_size(
            symbol=signal.symbol,
            price=signal.price,
            capital=capital,
            signal_strength=signal.strength,
            stop_loss_distance=stop_loss_distance,
        )

        # 应用回撤控制乘数
        final_size = raw_size * dd_multiplier

        if final_size <= 0:
            return RiskResult(
                approved=False,
                reason="Position size too small after adjustments",
            )

        logger.info(
            f"Risk approved: {signal.symbol} size={final_size:.6f} "
            f"(raw={raw_size:.6f}, dd_mult={dd_multiplier:.2f})"
        )

        return RiskResult(approved=True, position_size=final_size)

    def check_stop_loss(
        self, position: Position, tick: Tick
    ) -> Optional[Signal]:
        """
        实时止盈止损检查

        检查条件：
        1. 固定止损
        2. 固定止盈
        3. 追踪止盈（Trailing Stop）
        """
        if position.amount <= 0:
            return None

        price = tick.last
        is_long = position.side.value == "long"

        # ─── 固定止损 ───
        if position.stop_loss:
            if (is_long and price <= position.stop_loss) or \
               (not is_long and price >= position.stop_loss):
                action = SignalAction.CLOSE_LONG if is_long else SignalAction.CLOSE_SHORT
                return Signal(
                    symbol=position.symbol,
                    exchange=position.exchange,
                    strategy=position.strategy,
                    action=action,
                    price=price,
                    strength=1.0,
                    metadata={"reason": "stop_loss", "sl_price": position.stop_loss},
                )

        # ─── 固定止盈 ───
        if position.take_profit:
            if (is_long and price >= position.take_profit) or \
               (not is_long and price <= position.take_profit):
                action = SignalAction.CLOSE_LONG if is_long else SignalAction.CLOSE_SHORT
                return Signal(
                    symbol=position.symbol,
                    exchange=position.exchange,
                    strategy=position.strategy,
                    action=action,
                    price=price,
                    strength=1.0,
                    metadata={"reason": "take_profit", "tp_price": position.take_profit},
                )

        # ─── 追踪止盈 (Trailing Stop) ───
        # 使用 highest/lowest price 追踪
        trailing_pct = 0.03  # 3% 追踪止盈距离
        if is_long and position.highest_price > 0:
            trailing_stop = position.highest_price * (1 - trailing_pct)
            # 只有在已经盈利超过一定比例后才启用追踪止盈
            profit_pct = (position.highest_price - position.entry_price) / position.entry_price
            if profit_pct > 0.02 and price <= trailing_stop:
                return Signal(
                    symbol=position.symbol,
                    exchange=position.exchange,
                    strategy=position.strategy,
                    action=SignalAction.CLOSE_LONG,
                    price=price,
                    strength=1.0,
                    metadata={
                        "reason": "trailing_stop",
                        "highest": position.highest_price,
                        "trailing_stop": trailing_stop,
                    },
                )

        if not is_long and position.lowest_price < float('inf'):
            trailing_stop = position.lowest_price * (1 + trailing_pct)
            profit_pct = (position.entry_price - position.lowest_price) / position.entry_price
            if profit_pct > 0.02 and price >= trailing_stop:
                return Signal(
                    symbol=position.symbol,
                    exchange=position.exchange,
                    strategy=position.strategy,
                    action=SignalAction.CLOSE_SHORT,
                    price=price,
                    strength=1.0,
                    metadata={
                        "reason": "trailing_stop",
                        "lowest": position.lowest_price,
                        "trailing_stop": trailing_stop,
                    },
                )

        return None

    def check_account(
        self, account: AccountSnapshot
    ) -> Optional[RiskAlert]:
        """
        账户级风控检查

        在 engine.py 的 _periodic_sync 中调用
        """
        self.drawdown_control.update_equity(account.total_equity)
        self.circuit_breaker.reset_daily(account.total_equity)

        dd_status = self.drawdown_control.check_drawdown(account.total_equity)

        if dd_status["level"] == "halt":
            return RiskAlert(
                level="emergency",
                message=f"Max drawdown reached: {dd_status['drawdown']:.1%}. "
                        f"Peak: {dd_status['peak_equity']:.2f}, "
                        f"Current: {account.total_equity:.2f}",
                action="halt",
            )

        if dd_status["level"] == "critical":
            return RiskAlert(
                level="critical",
                message=f"Critical drawdown: {dd_status['drawdown']:.1%}. "
                        f"Exposure reduced to {dd_status['multiplier']:.0%}",
                action="reduce",
            )

        if dd_status["level"] == "warning":
            return RiskAlert(
                level="warning",
                message=f"Drawdown warning: {dd_status['drawdown']:.1%}",
                action="reduce",
            )

        return None

    def on_trade_closed(self, pnl: float, equity: float) -> None:
        """交易平仓后更新风控状态"""
        self.circuit_breaker.on_trade_result(pnl, equity)
