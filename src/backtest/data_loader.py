"""
回测数据加载 — 从本地文件或交易所 API 加载历史K线
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.core.event import Bar


class DataLoader:
    """历史数据加载器"""

    def __init__(self, data_dir: str = "data/klines"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def load_bars(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[Bar]:
        """
        从本地 CSV 文件加载K线数据

        文件命名：{symbol}_{exchange}_{timeframe}.csv
        格式：timestamp,open,high,low,close,volume
        """
        # 标准化文件名
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        filename = f"{safe_symbol}_{exchange}_{timeframe}.csv"
        filepath = self._data_dir / filename

        if not filepath.exists():
            logger.warning(f"Data file not found: {filepath}")
            return []

        df = pd.read_csv(filepath)

        # 标准化列名
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower in ("timestamp", "time", "date"):
                col_map[col] = "timestamp"
            elif lower == "open":
                col_map[col] = "open"
            elif lower == "high":
                col_map[col] = "high"
            elif lower == "low":
                col_map[col] = "low"
            elif lower == "close":
                col_map[col] = "close"
            elif lower in ("volume", "vol"):
                col_map[col] = "volume"

        df = df.rename(columns=col_map)

        required = ["timestamp", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required):
            logger.error(f"Missing columns in {filepath}: need {required}")
            return []

        # 时间过滤
        if start_date:
            start_ts = pd.Timestamp(start_date).timestamp() * 1000
            df = df[df["timestamp"] >= start_ts]
        if end_date:
            end_ts = pd.Timestamp(end_date).timestamp() * 1000
            df = df[df["timestamp"] <= end_ts]

        # 排序
        df = df.sort_values("timestamp").reset_index(drop=True)

        bars = [
            Bar(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                timestamp=float(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                is_closed=True,
            )
            for _, row in df.iterrows()
        ]

        logger.info(f"Loaded {len(bars)} bars from {filepath}")
        return bars

    def save_bars(
        self,
        bars: list[Bar],
        symbol: str,
        exchange: str,
        timeframe: str,
    ) -> None:
        """保存K线数据到本地 CSV"""
        if not bars:
            return

        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        filename = f"{safe_symbol}_{exchange}_{timeframe}.csv"
        filepath = self._data_dir / filename

        data = {
            "timestamp": [b.timestamp for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(bars)} bars to {filepath}")
