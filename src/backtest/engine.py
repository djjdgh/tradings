"""
回测引擎 — 复用策略代码，模拟撮合

设计原则：策略代码在回测和实盘中完全相同。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.core.event import (
    AccountSnapshot, Bar, Order, OrderStatus, OrderType,
    Position, Side, Signal, SignalAction,
)
from src.risk.manager import RiskManager
from src.strategy.base import BaseStrategy


@dataclass
class BacktestTrade:
    """回测交易记录"""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    amount: float
    pnl: float
    fee: float
    entry_time: float
    exit_time: float
    bars_held: int
    strategy: str


@dataclass
class BacktestResult:
    """回测结果"""
    initial_capital: float
    final_equity: float
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_holding_bars: float
    equity_curve: list[dict] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)
    monthly_returns: dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """
    回测引擎

    逐根K线回放历史数据，调用策略 on_bar() 获取信号，
    模拟撮合并计算盈亏。
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.0004,  # Taker 0.04%
        slippage_pct: float = 0.0005,      # 0.05%
    ):
        self._initial_capital = initial_capital
        self._commission_rate = commission_rate
        self._slippage_pct = slippage_pct

    async def run(
        self,
        strategy: BaseStrategy,
        bars: list[Bar],
        risk_manager: Optional[RiskManager] = None,
    ) -> BacktestResult:
        """
        运行回测

        Args:
            strategy: 策略实例
            bars: 历史K线数据（按时间正序）
            risk_manager: 风控管理器（可选）

        Returns:
            BacktestResult
        """
        if not bars:
            return BacktestResult(
                initial_capital=self._initial_capital,
                final_equity=self._initial_capital,
                total_return=0, annual_return=0, max_drawdown=0,
                sharpe_ratio=0, sortino_ratio=0, win_rate=0,
                profit_factor=0, total_trades=0, avg_holding_bars=0,
            )

        # ─── 初始化状态 ───
        equity = self._initial_capital
        peak_equity = equity
        max_drawdown = 0.0
        position: Optional[dict] = None  # 当前持仓
        trades: list[BacktestTrade] = []
        equity_curve: list[dict] = []
        daily_returns: list[float] = []
        prev_equity = equity

        required_bars = strategy.required_bars
        symbol = bars[0].symbol
        exchange = bars[0].exchange

        logger.info(
            f"Backtest starting: {symbol} | "
            f"{len(bars)} bars | "
            f"capital={self._initial_capital}"
        )

        # ─── 逐根K线回放 ───
        for i in range(required_bars, len(bars)):
            bar = bars[i]
            history = bars[max(0, i - required_bars):i + 1]

            # 更新持仓浮动盈亏
            if position is not None:
                position["bars_held"] += 1
                if position["side"] == "long":
                    position["unrealized_pnl"] = (
                        (bar.close - position["entry_price"]) * position["amount"]
                    )
                else:
                    position["unrealized_pnl"] = (
                        (position["entry_price"] - bar.close) * position["amount"]
                    )

                # 追踪最高/最低价
                position["highest"] = max(position["highest"], bar.high)
                position["lowest"] = min(position["lowest"], bar.low)

                # ─── 止损止盈检查 ───
                closed = self._check_stops(position, bar)
                if closed:
                    trade = self._close_position(position, closed["price"], bar.timestamp)
                    trades.append(trade)
                    equity += trade.pnl - trade.fee
                    position = None

            # ─── 策略信号 ───
            signal = await strategy.on_bar(bar, history)

            if signal and signal.action in (SignalAction.OPEN_LONG, SignalAction.OPEN_SHORT):
                # 风控检查
                if risk_manager:
                    positions_dict = {}
                    if position:
                        pos_obj = Position(
                            symbol=symbol, exchange=exchange,
                            side=Side.LONG if position["side"] == "long" else Side.SHORT,
                            amount=position["amount"],
                            entry_price=position["entry_price"],
                        )
                        positions_dict[symbol] = pos_obj

                    account = AccountSnapshot(
                        exchange=exchange, timestamp=bar.timestamp,
                        total_equity=equity, available_balance=equity,
                        unrealized_pnl=0, margin_used=0,
                    )

                    risk_result = risk_manager.evaluate_signal(
                        signal, positions_dict, account
                    )

                    if not risk_result.approved:
                        continue

                    amount = risk_result.position_size
                else:
                    amount = (equity * 0.3) / bar.close  # 默认 30% 仓位

                # 没有持仓时开仓
                if position is None:
                    entry_price = self._apply_slippage(
                        bar.close,
                        signal.action == SignalAction.OPEN_LONG,
                    )
                    fee = entry_price * amount * self._commission_rate

                    position = {
                        "side": "long" if signal.action == SignalAction.OPEN_LONG else "short",
                        "entry_price": entry_price,
                        "amount": amount,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "entry_time": bar.timestamp,
                        "bars_held": 0,
                        "unrealized_pnl": 0.0,
                        "highest": bar.high,
                        "lowest": bar.low,
                        "strategy": signal.strategy,
                        "entry_fee": fee,
                    }
                    equity -= fee

            elif signal and signal.action in (SignalAction.CLOSE_LONG, SignalAction.CLOSE_SHORT):
                if position is not None:
                    trade = self._close_position(position, bar.close, bar.timestamp)
                    trades.append(trade)
                    equity += trade.pnl - trade.fee
                    position = None

            # ─── 更新净值曲线 ───
            current_equity = equity
            if position:
                current_equity += position["unrealized_pnl"]

            equity_curve.append({
                "timestamp": bar.timestamp,
                "equity": current_equity,
            })

            # 回撤
            if current_equity > peak_equity:
                peak_equity = current_equity
            dd = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, dd)

            # 日收益率
            if prev_equity > 0:
                daily_returns.append((current_equity - prev_equity) / prev_equity)
            prev_equity = current_equity

            # 更新风控波动率
            if risk_manager:
                risk_manager.position_sizer.update_price(symbol, bar.close)

        # ─── 平掉剩余持仓 ───
        if position is not None:
            trade = self._close_position(position, bars[-1].close, bars[-1].timestamp)
            trades.append(trade)
            equity += trade.pnl - trade.fee

        # ─── 计算统计指标 ───
        final_equity = equity
        total_return = (final_equity - self._initial_capital) / self._initial_capital

        # 年化收益
        if len(bars) > 1:
            days = (bars[-1].timestamp - bars[0].timestamp) / (1000 * 86400)
            annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0
        else:
            annual_return = 0

        # 夏普/索提诺
        import numpy as np
        returns_arr = np.array(daily_returns) if daily_returns else np.array([0])
        sharpe = self._compute_sharpe(returns_arr)
        sortino = self._compute_sortino(returns_arr)

        # 胜率/盈亏比
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        win_rate = len(winning) / len(trades) if trades else 0

        total_profit = sum(t.pnl for t in winning)
        total_loss = abs(sum(t.pnl for t in losing))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        avg_holding = (
            sum(t.bars_held for t in trades) / len(trades) if trades else 0
        )

        result = BacktestResult(
            initial_capital=self._initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_holding_bars=avg_holding,
            equity_curve=equity_curve,
            trades=trades,
        )

        logger.info(
            f"Backtest complete: return={total_return:.1%}, "
            f"max_dd={max_drawdown:.1%}, sharpe={sharpe:.2f}, "
            f"trades={len(trades)}, win_rate={win_rate:.1%}"
        )

        return result

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """模拟滑点"""
        if is_buy:
            return price * (1 + self._slippage_pct)
        else:
            return price * (1 - self._slippage_pct)

    def _close_position(
        self, position: dict, exit_price: float, timestamp: float
    ) -> BacktestTrade:
        """平仓并计算盈亏"""
        is_long = position["side"] == "long"
        slipped_exit = self._apply_slippage(exit_price, not is_long)
        exit_fee = slipped_exit * position["amount"] * self._commission_rate

        if is_long:
            pnl = (slipped_exit - position["entry_price"]) * position["amount"]
        else:
            pnl = (position["entry_price"] - slipped_exit) * position["amount"]

        total_fee = position.get("entry_fee", 0) + exit_fee

        return BacktestTrade(
            symbol="",
            side=position["side"],
            entry_price=position["entry_price"],
            exit_price=slipped_exit,
            amount=position["amount"],
            pnl=pnl,
            fee=total_fee,
            entry_time=position["entry_time"],
            exit_time=timestamp,
            bars_held=position["bars_held"],
            strategy=position.get("strategy", ""),
        )

    def _check_stops(self, position: dict, bar: Bar) -> Optional[dict]:
        """检查止损止盈"""
        is_long = position["side"] == "long"

        # 固定止损
        sl = position.get("stop_loss")
        if sl:
            if (is_long and bar.low <= sl) or (not is_long and bar.high >= sl):
                return {"price": sl, "reason": "stop_loss"}

        # 固定止盈
        tp = position.get("take_profit")
        if tp:
            if (is_long and bar.high >= tp) or (not is_long and bar.low <= tp):
                return {"price": tp, "reason": "take_profit"}

        return None

    @staticmethod
    def _compute_sharpe(returns: "np.ndarray", risk_free: float = 0.0) -> float:
        """计算年化夏普比率"""
        import numpy as np
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free / 365
        std = float(np.std(excess, ddof=1))
        if std <= 0:
            return 0.0
        return float(np.mean(excess)) / std * (365 ** 0.5)

    @staticmethod
    def _compute_sortino(returns: "np.ndarray", risk_free: float = 0.0) -> float:
        """计算年化索提诺比率"""
        import numpy as np
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free / 365
        downside = excess[excess < 0]
        if len(downside) < 1:
            return float('inf')
        downside_std = float(np.std(downside, ddof=1))
        if downside_std <= 0:
            return 0.0
        return float(np.mean(excess)) / downside_std * (365 ** 0.5)
