"""
主入口 — 加载配置、创建组件、启动交易引擎
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import yaml
from loguru import logger

from src.core.engine import TradingEngine
from src.data.feed import DataFeed
from src.data.store import DataStore
from src.exchange.factory import create_exchanges
from src.execution.order_manager import OrderManager
from src.monitor.notifier import Notifier
from src.risk.manager import RiskManager
from src.strategy.base import BaseStrategy
from src.strategy.bollinger_trend import BollingerTrendStrategy
from src.strategy.cross_momentum import CrossMomentumStrategy
from src.strategy.funding_rate import FundingRateStrategy
from src.strategy.tsmom import TSMOMStrategy


# 策略注册表
STRATEGY_MAP: dict[str, type[BaseStrategy]] = {
    "bollinger_trend": BollingerTrendStrategy,
    "tsmom": TSMOMStrategy,
    "funding_rate": FundingRateStrategy,
    "cross_momentum": CrossMomentumStrategy,
}


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """加载主配置文件"""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded from {config_path}")
    return config


def load_strategy_config(strategy_name: str) -> dict:
    """加载策略配置文件"""
    path = Path(f"config/strategies/{strategy_name}.yaml")
    if not path.exists():
        logger.warning(f"Strategy config not found: {path}, using defaults")
        return {}

    with open(path) as f:
        return yaml.safe_load(f) or {}


def setup_logging(config: dict) -> None:
    """配置日志"""
    log_cfg = config.get("logging", {})
    log_dir = log_cfg.get("dir", "logs")
    log_level = log_cfg.get("level", "INFO")
    log_rotation = log_cfg.get("rotation", "1 day")
    log_retention = log_cfg.get("retention", "30 days")
    log_format = log_cfg.get(
        "format",
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
        "{name}:{function}:{line} | {message}"
    )

    # 移除默认 handler
    logger.remove()

    # 控制台输出
    logger.add(sys.stderr, level=log_level, format=log_format)

    # 文件输出
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.add(
        f"{log_dir}/tradings.log",
        level=log_level,
        format=log_format,
        rotation=log_rotation,
        retention=log_retention,
        compression="gz",
    )


def create_strategies(config: dict) -> dict[str, BaseStrategy]:
    """根据配置创建策略实例"""
    strategies: dict[str, BaseStrategy] = {}
    symbols_config = config.get("symbols", [])

    # 收集需要的策略
    strategy_symbols: dict[str, list[dict]] = {}
    for sym_cfg in symbols_config:
        strategy_name = sym_cfg.get("strategy", "")
        if strategy_name not in strategy_symbols:
            strategy_symbols[strategy_name] = []
        strategy_symbols[strategy_name].append(sym_cfg)

    # 创建策略实例
    for strategy_name, sym_list in strategy_symbols.items():
        strategy_cls = STRATEGY_MAP.get(strategy_name)
        if strategy_cls is None:
            logger.warning(f"Unknown strategy: {strategy_name}, skipping")
            continue

        strategy_config = load_strategy_config(strategy_name)
        strategy = strategy_cls(name=strategy_name, config=strategy_config)

        # 注册品种和周期
        for sym_cfg in sym_list:
            strategy.add_symbol(sym_cfg["symbol"])
            strategy.add_timeframe(sym_cfg["timeframe"])

        strategies[strategy_name] = strategy
        logger.info(
            f"Strategy created: {strategy_name} "
            f"({len(sym_list)} symbols)"
        )

    return strategies


async def main() -> None:
    """主函数"""
    # 加载配置
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/settings.yaml"
    config = load_config(config_path)

    # 配置日志
    setup_logging(config)

    logger.info("=" * 60)
    logger.info("Crypto Futures Quantitative Trading System")
    logger.info(f"Mode: {config.get('mode', 'paper')}")
    logger.info("=" * 60)

    # 确保数据目录存在
    Path("data/db").mkdir(parents=True, exist_ok=True)
    Path("data/klines").mkdir(parents=True, exist_ok=True)

    # 创建组件
    exchanges = create_exchanges(config.get("exchanges", {}))
    strategies = create_strategies(config)
    risk_manager = RiskManager(config)
    order_manager = OrderManager()
    data_feed = DataFeed()
    data_store = DataStore(
        db_url=config.get("database", {}).get(
            "url", "sqlite+aiosqlite:///data/db/tradings.db"
        ),
        echo=config.get("database", {}).get("echo", False),
    )
    notifier = Notifier(config.get("notification", {}))

    # 创建并启动引擎
    engine = TradingEngine(
        exchanges=exchanges,
        strategies=strategies,
        risk_manager=risk_manager,
        order_manager=order_manager,
        data_feed=data_feed,
        data_store=data_store,
        notifier=notifier,
        config=config,
    )

    try:
        await engine.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
    finally:
        await engine.stop()
        await data_store.close()


if __name__ == "__main__":
    asyncio.run(main())
