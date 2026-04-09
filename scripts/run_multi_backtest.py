"""
多策略多品种组合回测

用法:
    PYTHONPATH=. python scripts/run_multi_backtest.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

sys.path.insert(0, ".")

from src.backtest.data_loader import DataLoader
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.report import generate_report
from src.risk.manager import RiskManager
from src.strategy.bollinger_trend import BollingerTrendStrategy
from src.strategy.cross_momentum import CrossMomentumStrategy
from src.strategy.tsmom import TSMOMStrategy


def load_strategy_config(name: str) -> dict:
    path = Path(f"config/strategies/{name}.yaml")
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def load_main_config() -> dict:
    for p in ["config/settings.yaml", "config/settings.example.yaml"]:
        path = Path(p)
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
    return {}


async def run_all():
    loader = DataLoader("data/klines")
    config = load_main_config()
    exchange = "binanceusdm"
    capital = 5000.0

    symbols = [
        "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
        "DOGE/USDT:USDT", "LINK/USDT:USDT", "AVAX/USDT:USDT",
    ]

    # 加载所有品种数据
    all_bars = {}
    for sym in symbols:
        bars = loader.load_bars(sym, exchange, "1d")
        if bars:
            all_bars[sym] = bars

    if not all_bars:
        print("No data found! Run generate_test_data.py first.")
        return

    print("=" * 70)
    print("  多策略多品种组合回测")
    print(f"  品种: {len(all_bars)} | 初始资金: {capital} USDT")
    print("=" * 70)

    all_results: dict[str, list[BacktestResult]] = {}

    # ─── 1. TSMOM 策略：每个品种单独回测 ───
    print("\n" + "─" * 70)
    print("  策略 1: TSMOM 多尺度趋势跟踪")
    print("─" * 70)

    tsmom_cfg = load_strategy_config("tsmom")
    tsmom_results = []

    for sym, bars in all_bars.items():
        strategy = TSMOMStrategy("tsmom", tsmom_cfg)
        strategy.add_symbol(sym)
        strategy.add_timeframe("1d")

        risk_manager = RiskManager(config)
        engine = BacktestEngine(initial_capital=capital, commission_rate=0.0004, slippage_pct=0.0005)

        result = await engine.run(strategy, bars, risk_manager)
        tsmom_results.append(result)

        ret_str = f"{result.total_return:+.1%}"
        dd_str = f"{result.max_drawdown:.1%}"
        sr_str = f"{result.sharpe_ratio:.2f}"
        print(f"  {sym:<20} return={ret_str:>7}  dd={dd_str:>6}  sharpe={sr_str:>6}  trades={result.total_trades}")

    all_results["tsmom"] = tsmom_results

    # ─── 2. 布林带策略 ───
    print("\n" + "─" * 70)
    print("  策略 2: 布林带趋势突破")
    print("─" * 70)

    bb_cfg = load_strategy_config("bollinger_trend")
    bb_results = []

    for sym in ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]:
        if sym not in all_bars:
            continue
        bars = all_bars[sym]

        strategy = BollingerTrendStrategy("bollinger_trend", bb_cfg)
        strategy.add_symbol(sym)
        strategy.add_timeframe("1d")

        risk_manager = RiskManager(config)
        engine = BacktestEngine(initial_capital=capital, commission_rate=0.0004, slippage_pct=0.0005)

        result = await engine.run(strategy, bars, risk_manager)
        bb_results.append(result)

        ret_str = f"{result.total_return:+.1%}"
        dd_str = f"{result.max_drawdown:.1%}"
        sr_str = f"{result.sharpe_ratio:.2f}"
        print(f"  {sym:<20} return={ret_str:>7}  dd={dd_str:>6}  sharpe={sr_str:>6}  trades={result.total_trades}")

    all_results["bollinger"] = bb_results

    # ─── 3. 跨币种动量：模拟多品种排名 ───
    print("\n" + "─" * 70)
    print("  策略 3: 跨币种动量轮动 (多品种排名)")
    print("─" * 70)

    cm_cfg = load_strategy_config("cross_momentum")
    cm_strategy = CrossMomentumStrategy("cross_momentum", cm_cfg)
    for sym in all_bars:
        cm_strategy.add_symbol(sym)
    cm_strategy.add_timeframe("1d")

    # 逐根K线回放所有品种，让策略积累动量分数
    risk_manager = RiskManager(config)
    engine = BacktestEngine(initial_capital=capital, commission_rate=0.0004, slippage_pct=0.0005)

    # 先用所有品种的数据更新动量分数
    min_len = min(len(bars) for bars in all_bars.values())
    for sym, bars in all_bars.items():
        if len(bars) >= cm_strategy.required_bars:
            # 预计算动量分数
            from src.strategy.cross_momentum import CrossMomentumStrategy as CMS
            close_arr = np.array([b.close for b in bars[-cm_strategy._lookback_days:]])
            score = CMS._compute_momentum_score(close_arr)
            cm_strategy._momentum_scores[sym] = {
                "score": score["score"],
                "slope": score["slope"],
                "r_squared": score["r_squared"],
                "price": bars[-1].close,
            }

    # 用 BTC 做回测（此时有排名数据）
    cm_results = []
    for sym in ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]:
        if sym not in all_bars:
            continue

        cm_strat = CrossMomentumStrategy("cross_momentum", cm_cfg)
        cm_strat.add_symbol(sym)
        cm_strat.add_timeframe("1d")
        # 注入所有品种的动量分数
        cm_strat._momentum_scores = dict(cm_strategy._momentum_scores)

        rm = RiskManager(config)
        eng = BacktestEngine(initial_capital=capital, commission_rate=0.0004, slippage_pct=0.0005)
        result = await eng.run(cm_strat, all_bars[sym], rm)
        cm_results.append(result)

        ret_str = f"{result.total_return:+.1%}"
        dd_str = f"{result.max_drawdown:.1%}"
        sr_str = f"{result.sharpe_ratio:.2f}"
        print(f"  {sym:<20} return={ret_str:>7}  dd={dd_str:>6}  sharpe={sr_str:>6}  trades={result.total_trades}")

    all_results["cross_momentum"] = cm_results

    # ─── 组合汇总 ───
    print("\n" + "=" * 70)
    print("  组合汇总")
    print("=" * 70)

    total_final = 0
    total_initial = 0
    worst_dd = 0
    total_trades = 0

    for strat_name, results in all_results.items():
        for r in results:
            total_initial += r.initial_capital
            total_final += r.final_equity
            worst_dd = max(worst_dd, r.max_drawdown)
            total_trades += r.total_trades

    combined_return = (total_final - total_initial) / total_initial if total_initial > 0 else 0

    print(f"  总投入资金:      {total_initial:>12,.2f} USDT (分配到各策略+品种)")
    print(f"  总最终净值:      {total_final:>12,.2f} USDT")
    print(f"  组合总收益率:    {combined_return:>12.2%}")
    print(f"  最大单策略回撤:  {worst_dd:>12.2%}")
    print(f"  总交易次数:      {total_trades:>12d}")
    print()

    # 各策略平均表现
    print("  各策略平均表现:")
    for strat_name, results in all_results.items():
        if not results:
            continue
        avg_ret = np.mean([r.total_return for r in results])
        avg_dd = np.mean([r.max_drawdown for r in results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in results])
        total_t = sum(r.total_trades for r in results)
        print(f"    {strat_name:<20} avg_return={avg_ret:+.2%}  avg_dd={avg_dd:.2%}  avg_sharpe={avg_sharpe:.2f}  trades={total_t}")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_all())
