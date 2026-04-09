"""
下载历史资金费率数据

用法:
    python scripts/download_funding_rates.py
"""

import asyncio, os, sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

try:
    import ccxt.async_support as ccxt
except ImportError:
    print("pip install ccxt"); sys.exit(1)

SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]


async def download_funding_rates(symbol, exchange, days=365):
    """下载历史资金费率"""
    print(f"  Downloading funding rates for {symbol} ({days} days)...")
    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_rates = []

    while True:
        try:
            rates = await exchange.fetch_funding_rate_history(
                symbol, since=since, limit=1000
            )
        except Exception as e:
            print(f"    Error: {e}")
            break

        if not rates:
            break

        for r in rates:
            all_rates.append({
                "timestamp": r.get("timestamp", 0),
                "funding_rate": r.get("fundingRate", 0),
                "datetime": r.get("datetime", ""),
            })

        since = int(rates[-1]["timestamp"]) + 1
        if len(rates) < 1000:
            break
        await asyncio.sleep(0.3)

    if not all_rates:
        print(f"    No funding rate data for {symbol}")
        return

    df = pd.DataFrame(all_rates)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    safe = symbol.replace("/", "_").replace(":", "_")
    Path("data/klines").mkdir(parents=True, exist_ok=True)
    fp = f"data/klines/{safe}_binanceusdm_funding.csv"
    df.to_csv(fp, index=False)

    start = df["datetime"].iloc[0][:10]
    end = df["datetime"].iloc[-1][:10]
    print(f"    Saved {len(df)} records ({start} → {end}) → {fp}")


async def main():
    exchange = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 30000})
    try:
        await exchange.load_markets()
        print(f"Connected!\n")
        for sym in SYMBOLS:
            await download_funding_rates(sym, exchange, days=365)
            await asyncio.sleep(0.5)
        print("\nDone!")
    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
