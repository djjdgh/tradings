"""
历史K线数据下载脚本

用法:
    python scripts/download_data.py --symbol BTC/USDT:USDT --timeframe 1d --days 365
    python scripts/download_data.py --symbol ETH/USDT:USDT --timeframe 15m --days 90
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime, timedelta

import ccxt.pro as ccxtpro
from loguru import logger

from src.backtest.data_loader import DataLoader
from src.core.event import Bar


async def download(
    symbol: str,
    timeframe: str,
    days: int,
    exchange_id: str = "binanceusdm",
    data_dir: str = "data/klines",
) -> None:
    """下载历史K线数据"""

    exchange = ccxtpro.binanceusdm({"enableRateLimit": True})

    try:
        await exchange.load_markets()

        # 计算起始时间
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
        all_bars: list[Bar] = []
        batch_size = 1000

        logger.info(
            f"Downloading {symbol} {timeframe} | "
            f"Last {days} days | Exchange: {exchange_id}"
        )

        while True:
            ohlcvs = await exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=batch_size
            )

            if not ohlcvs:
                break

            for ts, o, h, l, c, v in ohlcvs:
                all_bars.append(Bar(
                    symbol=symbol,
                    exchange=exchange_id,
                    timeframe=timeframe,
                    timestamp=ts,
                    open=o, high=h, low=l, close=c, volume=v,
                    is_closed=True,
                ))

            logger.info(f"  Downloaded {len(all_bars)} bars so far...")

            # 下一批的起始时间
            since = int(ohlcvs[-1][0]) + 1

            if len(ohlcvs) < batch_size:
                break  # 没有更多数据了

            await asyncio.sleep(0.5)  # Rate limit

        # 保存到本地
        loader = DataLoader(data_dir)
        loader.save_bars(all_bars, symbol, exchange_id, timeframe)

        logger.info(
            f"Download complete: {len(all_bars)} bars saved "
            f"({all_bars[0].timestamp} → {all_bars[-1].timestamp})"
        )

    finally:
        await exchange.close()


def main():
    parser = argparse.ArgumentParser(description="Download historical kline data")
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Trading pair")
    parser.add_argument("--timeframe", default="1d", help="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)")
    parser.add_argument("--days", type=int, default=365, help="Number of days to download")
    parser.add_argument("--exchange", default="binanceusdm", help="Exchange ID")
    parser.add_argument("--data-dir", default="data/klines", help="Data directory")

    args = parser.parse_args()

    asyncio.run(download(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        exchange_id=args.exchange,
        data_dir=args.data_dir,
    ))


if __name__ == "__main__":
    main()
