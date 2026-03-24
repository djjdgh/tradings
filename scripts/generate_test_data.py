"""
生成合成历史数据用于回测测试

基于 GBM (Geometric Brownian Motion) + 波动率聚集 生成逼真的加密货币价格序列
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.backtest.data_loader import DataLoader
from src.core.event import Bar


def generate_crypto_prices(
    initial_price: float,
    days: int,
    annual_drift: float = 0.30,     # 年化漂移 30%
    annual_vol: float = 0.70,       # 年化波动率 70%
    vol_persistence: float = 0.95,  # 波动率聚集（GARCH效应）
    jump_prob: float = 0.02,        # 跳跃概率 2%/天
    jump_size: float = 0.05,        # 跳跃幅度 5%
    seed: int = 42,
) -> np.ndarray:
    """
    生成逼真的加密货币日线价格序列

    使用 GBM + 随机波动率 + 跳跃过程
    """
    rng = np.random.RandomState(seed)
    dt = 1.0 / 365.0
    daily_drift = annual_drift * dt
    base_daily_vol = annual_vol * np.sqrt(dt)

    prices = [initial_price]
    current_vol = base_daily_vol

    for _ in range(days - 1):
        # 波动率聚集（简化 GARCH）
        vol_shock = rng.normal(0, 0.1)
        current_vol = (
            vol_persistence * current_vol
            + (1 - vol_persistence) * base_daily_vol
            + 0.005 * vol_shock
        )
        current_vol = max(current_vol, base_daily_vol * 0.3)

        # 正常收益
        ret = daily_drift + current_vol * rng.normal()

        # 跳跃过程
        if rng.random() < jump_prob:
            jump = rng.choice([-1, 1]) * jump_size * rng.random()
            ret += jump

        new_price = prices[-1] * np.exp(ret)
        prices.append(new_price)

    return np.array(prices)


def prices_to_ohlcv(
    daily_prices: np.ndarray,
    start_timestamp_ms: int,
    symbol: str,
    exchange: str,
    timeframe: str = "1d",
    seed: int = 42,
) -> list[Bar]:
    """将日线收盘价转换为完整的 OHLCV K线"""
    rng = np.random.RandomState(seed + 1)
    bars = []
    ms_per_day = 86400 * 1000

    for i, close in enumerate(daily_prices):
        # 生成 OHLCV
        intraday_vol = close * 0.015 * (0.5 + rng.random())
        open_price = close * (1 + rng.normal(0, 0.005))
        high = max(open_price, close) + abs(rng.normal(0, 1)) * intraday_vol
        low = min(open_price, close) - abs(rng.normal(0, 1)) * intraday_vol

        # 确保 OHLC 关系正确
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # 成交量（与波动率正相关）
        price_range_pct = (high - low) / close
        base_volume = 50000 + rng.random() * 30000
        volume = base_volume * (1 + price_range_pct * 20)

        bars.append(Bar(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            timestamp=start_timestamp_ms + i * ms_per_day,
            open=round(open_price, 2),
            high=round(high, 2),
            low=round(low, 2),
            close=round(close, 2),
            volume=round(volume, 2),
        ))

    return bars


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test data")
    parser.add_argument("--days", type=int, default=365, help="Days of data")
    args = parser.parse_args()

    days = args.days
    exchange = "binanceusdm"
    # 起始时间: 2025-03-24 (一年前)
    start_ts = int(pd.Timestamp("2025-03-24").timestamp() * 1000)

    loader = DataLoader("data/klines")

    # ─── BTC ───
    print("Generating BTC/USDT:USDT data...")
    btc_prices = generate_crypto_prices(
        initial_price=65000, days=days,
        annual_drift=0.40, annual_vol=0.65, seed=42,
    )
    btc_bars = prices_to_ohlcv(
        btc_prices, start_ts, "BTC/USDT:USDT", exchange, seed=42,
    )
    loader.save_bars(btc_bars, "BTC/USDT:USDT", exchange, "1d")
    print(f"  BTC: {btc_bars[0].close:.0f} → {btc_bars[-1].close:.0f} "
          f"({(btc_bars[-1].close/btc_bars[0].close - 1):.1%})")

    # ─── ETH ───
    print("Generating ETH/USDT:USDT data...")
    eth_prices = generate_crypto_prices(
        initial_price=3200, days=days,
        annual_drift=0.50, annual_vol=0.80, seed=123,
    )
    eth_bars = prices_to_ohlcv(
        eth_prices, start_ts, "ETH/USDT:USDT", exchange, seed=123,
    )
    loader.save_bars(eth_bars, "ETH/USDT:USDT", exchange, "1d")
    print(f"  ETH: {eth_bars[0].close:.0f} → {eth_bars[-1].close:.0f} "
          f"({(eth_bars[-1].close/eth_bars[0].close - 1):.1%})")

    # ─── SOL ───
    print("Generating SOL/USDT:USDT data...")
    sol_prices = generate_crypto_prices(
        initial_price=180, days=days,
        annual_drift=0.60, annual_vol=0.90, seed=456,
    )
    sol_bars = prices_to_ohlcv(
        sol_prices, start_ts, "SOL/USDT:USDT", exchange, seed=456,
    )
    loader.save_bars(sol_bars, "SOL/USDT:USDT", exchange, "1d")
    print(f"  SOL: {sol_bars[0].close:.0f} → {sol_bars[-1].close:.0f} "
          f"({(sol_bars[-1].close/sol_bars[0].close - 1):.1%})")

    # ─── DOGE ───
    print("Generating DOGE/USDT:USDT data...")
    doge_prices = generate_crypto_prices(
        initial_price=0.15, days=days,
        annual_drift=0.20, annual_vol=1.00, seed=789,
    )
    doge_bars = prices_to_ohlcv(
        doge_prices, start_ts, "DOGE/USDT:USDT", exchange, seed=789,
    )
    loader.save_bars(doge_bars, "DOGE/USDT:USDT", exchange, "1d")
    print(f"  DOGE: {doge_bars[0].close:.4f} → {doge_bars[-1].close:.4f} "
          f"({(doge_bars[-1].close/doge_bars[0].close - 1):.1%})")

    # ─── LINK ───
    print("Generating LINK/USDT:USDT data...")
    link_prices = generate_crypto_prices(
        initial_price=15, days=days,
        annual_drift=0.35, annual_vol=0.85, seed=321,
    )
    link_bars = prices_to_ohlcv(
        link_prices, start_ts, "LINK/USDT:USDT", exchange, seed=321,
    )
    loader.save_bars(link_bars, "LINK/USDT:USDT", exchange, "1d")
    print(f"  LINK: {link_bars[0].close:.2f} → {link_bars[-1].close:.2f} "
          f"({(link_bars[-1].close/link_bars[0].close - 1):.1%})")

    # ─── AVAX ───
    print("Generating AVAX/USDT:USDT data...")
    avax_prices = generate_crypto_prices(
        initial_price=35, days=days,
        annual_drift=0.30, annual_vol=0.90, seed=654,
    )
    avax_bars = prices_to_ohlcv(
        avax_prices, start_ts, "AVAX/USDT:USDT", exchange, seed=654,
    )
    loader.save_bars(avax_bars, "AVAX/USDT:USDT", exchange, "1d")
    print(f"  AVAX: {avax_bars[0].close:.2f} → {avax_bars[-1].close:.2f} "
          f"({(avax_bars[-1].close/avax_bars[0].close - 1):.1%})")

    print(f"\nDone! Generated {days} days of data for 6 symbols.")
    print("Data saved to data/klines/")


if __name__ == "__main__":
    main()
