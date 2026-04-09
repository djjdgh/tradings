"""
回测报告生成 — 统计指标 + 文本报告
"""

from __future__ import annotations

from src.backtest.engine import BacktestResult


def generate_report(result: BacktestResult, strategy_name: str = "") -> str:
    """
    生成回测报告文本

    Args:
        result: 回测结果
        strategy_name: 策略名称

    Returns:
        格式化的报告文本
    """
    lines = [
        "=" * 60,
        f"  回测报告 — {strategy_name}" if strategy_name else "  回测报告",
        "=" * 60,
        "",
        "─── 收益指标 ───",
        f"  初始资金:     {result.initial_capital:>12,.2f} USDT",
        f"  最终净值:     {result.final_equity:>12,.2f} USDT",
        f"  总收益率:     {result.total_return:>12.2%}",
        f"  年化收益率:   {result.annual_return:>12.2%}",
        "",
        "─── 风险指标 ───",
        f"  最大回撤:     {result.max_drawdown:>12.2%}",
        f"  夏普比率:     {result.sharpe_ratio:>12.2f}",
        f"  索提诺比率:   {result.sortino_ratio:>12.2f}",
        "",
        "─── 交易统计 ───",
        f"  总交易次数:   {result.total_trades:>12d}",
        f"  胜率:         {result.win_rate:>12.2%}",
        f"  盈亏比:       {result.profit_factor:>12.2f}",
        f"  平均持仓:     {result.avg_holding_bars:>12.1f} 根K线",
        "",
    ]

    # 交易明细（最近 20 笔）
    if result.trades:
        lines.append("─── 最近交易明细 ───")
        lines.append(f"  {'方向':<6} {'入场价':>10} {'出场价':>10} "
                     f"{'盈亏':>10} {'持仓':>6}")
        lines.append("  " + "-" * 48)

        for trade in result.trades[-20:]:
            side_str = "多" if trade.side == "long" else "空"
            pnl_str = f"{trade.pnl:+.2f}"
            lines.append(
                f"  {side_str:<6} {trade.entry_price:>10.2f} "
                f"{trade.exit_price:>10.2f} {pnl_str:>10} "
                f"{trade.bars_held:>6d}"
            )

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
