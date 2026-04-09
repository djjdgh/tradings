"""
回测运行脚本

用法:
    python scripts/run_backtest.py --strategy bollinger_trend --symbol BTC/USDT:USDT
    python scripts/run_backtest.py --strategy tsmom --symbol BTC/USDT:USDT --capital 5000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml
from loguru import logger

from src.backtest.data_loader import DataLoader
from src.backtest.engine import BacktestEngine
from src.backtest.report import generate_report
from src.risk.manager import RiskManager
from src.strategy.bollinger_trend import BollingerTrendStrategy
from src.strategy.cross_momentum import CrossMomentumStrategy
from src.strategy.funding_rate import FundingRateStrategy
from src.strategy.tsmom import TSMOMStrategy


STRATEGY_MAP = {
    "bollinger_trend": BollingerTrendStrategy,
    "tsmom": TSMOMStrategy,
    "funding_rate": FundingRateStrategy,
    "cross_momentum": CrossMomentumStrategy,
}


def load_strategy_config(name: str) -> dict:
    """加载策略配置"""
    path = Path(f"config/strategies/{name}.yaml")
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def load_main_config() -> dict:
    """加载主配置"""
    path = Path("config/settings.yaml")
    if not path.exists():
        path = Path("config/settings.example.yaml")
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


async def run_backtest(
    strategy_name: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    capital: float,
    start_date: str | None,
    end_date: str | None,
    use_risk_manager: bool,
) -> None:
    """运行回测"""

    # 加载策略
    strategy_cls = STRATEGY_MAP.get(strategy_name)
    if strategy_cls is None:
        logger.error(f"Unknown strategy: {strategy_name}")
        logger.info(f"Available: {list(STRATEGY_MAP.keys())}")
        return

    strategy_config = load_strategy_config(strategy_name)
    strategy = strategy_cls(name=strategy_name, config=strategy_config)
    strategy.add_symbol(symbol)
    strategy.add_timeframe(timeframe)

    # 加载数据
    loader = DataLoader("data/klines")
    bars = loader.load_bars(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )

    if not bars:
        logger.error(
            f"No data found for {symbol} {timeframe}. "
            f"Run download_data.py first."
        )
        return

    # 风控
    risk_manager = None
    if use_risk_manager:
        main_config = load_main_config()
        risk_manager = RiskManager(main_config)

    # 回测
    main_config = load_main_config()
    bt_cfg = main_config.get("backtest", {})

    engine = BacktestEngine(
        initial_capital=capital,
        commission_rate=bt_cfg.get("commission_rate", 0.0004),
        slippage_pct=bt_cfg.get("slippage_pct", 0.0005),
    )

    result = await engine.run(
        strategy=strategy,
        bars=bars,
        risk_manager=risk_manager,
    )

    # 输出报告
    report = generate_report(result, strategy_name)
    print(report)


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--strategy", required=True, help="Strategy name")
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Trading pair")
    parser.add_argument("--exchange", default="binanceusdm", help="Exchange")
    parser.add_argument("--timeframe", default="1d", help="Timeframe")
    parser.add_argument("--capital", type=float, default=5000, help="Initial capital")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-risk", action="store_true", help="Disable risk manager")

    args = parser.parse_args()

    asyncio.run(run_backtest(
        strategy_name=args.strategy,
        symbol=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
        capital=args.capital,
        start_date=args.start,
        end_date=args.end,
        use_risk_manager=not args.no_risk,
    ))


if __name__ == "__main__":
    main()
