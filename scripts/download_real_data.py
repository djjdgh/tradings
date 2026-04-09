"""
真实历史数据下载脚本（在本地/服务器运行）

此脚本需要在能访问 Binance API 的环境中运行。
Cloud IDE 环境通常封锁了交易所 API，请在以下环境运行：
  - 你的本地电脑
  - 腾讯云服务器
  - 任何能访问 api.binance.com 的网络

用法:
    # 安装依赖
    pip install ccxt pandas

    # 下载全部品种（BTC/ETH/SOL 等 6 个币种，日线 + 15分钟线）
    python scripts/download_real_data.py --all

    # 下载单个品种
    python scripts/download_real_data.py --symbol BTC/USDT:USDT --timeframe 1d --days 365

    # 下载完成后，将 data/klines/ 目录上传到 Cloud IDE 或服务器
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

try:
    import ccxt.async_support as ccxt
except ImportError:
    print("请先安装 ccxt: pip install ccxt")
    sys.exit(1)


SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "DOGE/USDT:USDT",
    "LINK/USDT:USDT",
    "AVAX/USDT:USDT",
]

TIMEFRAMES = {
    "1d": 86400 * 1000,
    "4h": 4 * 3600 * 1000,
    "1h": 3600 * 1000,
    "15m": 15 * 60 * 1000,
}


async def download_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    days: int,
    data_dir: str,
) -> int:
    """下载单个品种的历史数据"""
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_data = []
    batch_size = 1000

    print(f"  Downloading {symbol} {timeframe} (last {days} days)...")

    while True:
        try:
            ohlcvs = await exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=batch_size
            )
        except Exception as e:
            print(f"    Error: {e}, retrying in 5s...")
            await asyncio.sleep(5)
            continue

        if not ohlcvs:
            break

        all_data.extend(ohlcvs)
        since = int(ohlcvs[-1][0]) + 1

        if len(ohlcvs) < batch_size:
            break

        # Rate limit
        await asyncio.sleep(0.3)

    if not all_data:
        print(f"    No data for {symbol} {timeframe}")
        return 0

    # 去重
    seen = set()
    unique_data = []
    for row in all_data:
        if row[0] not in seen:
            seen.add(row[0])
            unique_data.append(row)
    unique_data.sort(key=lambda x: x[0])

    # 保存 CSV
    df = pd.DataFrame(unique_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    exchange_name = "binanceusdm"
    filename = f"{safe_symbol}_{exchange_name}_{timeframe}.csv"

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(data_dir) / filename
    df.to_csv(filepath, index=False)

    start_dt = datetime.utcfromtimestamp(unique_data[0][0] / 1000).strftime("%Y-%m-%d")
    end_dt = datetime.utcfromtimestamp(unique_data[-1][0] / 1000).strftime("%Y-%m-%d")
    print(f"    Saved {len(unique_data)} bars ({start_dt} → {end_dt}) → {filepath}")

    return len(unique_data)


async def download_all(days: int, data_dir: str, timeframes: list[str]):
    """下载所有品种的所有周期"""
    exchange = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 30000})

    try:
        print(f"Connecting to Binance Futures...")
        await exchange.load_markets()
        print(f"Connected! {len(exchange.markets)} markets available\n")

        total_bars = 0
        for tf in timeframes:
            print(f"\n{'='*50}")
            print(f"Timeframe: {tf}")
            print(f"{'='*50}")
            for symbol in SYMBOLS:
                if symbol not in exchange.markets:
                    print(f"  {symbol} not found, skipping")
                    continue
                count = await download_symbol(exchange, symbol, tf, days, data_dir)
                total_bars += count
                await asyncio.sleep(1)  # Rate limit between symbols

        print(f"\n{'='*50}")
        print(f"Download complete! Total: {total_bars} bars")
        print(f"Data saved to: {data_dir}/")
        print(f"\n下载完成后，将 {data_dir}/ 目录复制到你的项目中即可回测。")

    finally:
        await exchange.close()


async def download_single(symbol: str, timeframe: str, days: int, data_dir: str):
    """下载单个品种"""
    exchange = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 30000})
    try:
        await exchange.load_markets()
        await download_symbol(exchange, symbol, timeframe, days, data_dir)
    finally:
        await exchange.close()


def main():
    parser = argparse.ArgumentParser(
        description="Download real historical kline data from Binance Futures"
    )
    parser.add_argument("--all", action="store_true",
                        help="Download all symbols and timeframes")
    parser.add_argument("--symbol", default="BTC/USDT:USDT",
                        help="Single symbol to download")
    parser.add_argument("--timeframe", default="1d",
                        help="Timeframe (1d, 4h, 1h, 15m)")
    parser.add_argument("--days", type=int, default=365,
                        help="Days of history")
    parser.add_argument("--data-dir", default="data/klines",
                        help="Output directory")

    args = parser.parse_args()

    if args.all:
        # 下载全部：日线 365 天 + 15 分钟线 90 天
        print("=" * 50)
        print("下载全部品种真实历史数据")
        print(f"品种: {', '.join(SYMBOLS)}")
        print(f"日线: 365 天 | 15 分钟线: 90 天")
        print("=" * 50)

        async def run_all():
            exchange = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 30000})
            try:
                await exchange.load_markets()
                print(f"Connected! {len(exchange.markets)} markets\n")

                total = 0
                # 日线 365 天
                print("\n>>> 日线数据 (1d, 365 天) <<<")
                for sym in SYMBOLS:
                    if sym in exchange.markets:
                        count = await download_symbol(exchange, sym, "1d", 365, args.data_dir)
                        total += count
                        await asyncio.sleep(0.5)

                # 15 分钟线 90 天
                print("\n>>> 15 分钟数据 (15m, 90 天) <<<")
                for sym in SYMBOLS[:3]:  # BTC, ETH, SOL
                    if sym in exchange.markets:
                        count = await download_symbol(exchange, sym, "15m", 90, args.data_dir)
                        total += count
                        await asyncio.sleep(0.5)

                # 1 小时线 180 天（资金费率策略用）
                print("\n>>> 1 小时数据 (1h, 180 天) <<<")
                for sym in SYMBOLS[:2]:  # BTC, ETH
                    if sym in exchange.markets:
                        count = await download_symbol(exchange, sym, "1h", 180, args.data_dir)
                        total += count
                        await asyncio.sleep(0.5)

                print(f"\n总计下载 {total} 根K线")
            finally:
                await exchange.close()

        asyncio.run(run_all())
    else:
        asyncio.run(download_single(args.symbol, args.timeframe, args.days, args.data_dir))


if __name__ == "__main__":
    main()
