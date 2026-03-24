"""
交易所工厂 — 根据配置创建交易所实例
"""

from __future__ import annotations

from loguru import logger

from src.exchange.base import BaseExchange
from src.exchange.binance_futures import BinanceFutures


# 支持的交易所映射
EXCHANGE_MAP: dict[str, type[BaseExchange]] = {
    "binance_futures": BinanceFutures,
}


def create_exchanges(exchanges_config: dict) -> dict[str, BaseExchange]:
    """
    根据配置创建所有启用的交易所实例

    Args:
        exchanges_config: settings.yaml 中的 exchanges 配置段

    Returns:
        {exchange_name: exchange_instance}
    """
    exchanges: dict[str, BaseExchange] = {}

    for name, cfg in exchanges_config.items():
        if not cfg.get("enabled", False):
            logger.info(f"Exchange {name} is disabled, skipping")
            continue

        exchange_cls = EXCHANGE_MAP.get(name)
        if exchange_cls is None:
            logger.warning(f"Unknown exchange type: {name}, skipping")
            continue

        exchanges[name] = exchange_cls(name=name, config=cfg)
        logger.info(f"Created exchange: {name}")

    return exchanges
