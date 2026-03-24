#!/bin/bash
# 下载 4h K线数据（用于 TSMOM 短周期回测）
# 用法: cd ~/tradings-repo && bash scripts/download_4h_data.sh

set -e

SYMBOLS=("BTC/USDT:USDT" "ETH/USDT:USDT" "SOL/USDT:USDT" "DOGE/USDT:USDT" "LINK/USDT:USDT" "AVAX/USDT:USDT")
DAYS=365

echo "=== 下载 4h K线数据 ==="

for sym in "${SYMBOLS[@]}"; do
    echo "  Downloading $sym 4h (last ${DAYS} days)..."
    PYTHONPATH=. python scripts/download_data.py --symbol "$sym" --timeframe 4h --days $DAYS
    sleep 1
done

echo "=== 完成 ==="
