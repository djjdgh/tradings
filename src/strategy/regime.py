"""
市场 Regime 检测 — 自动识别趋势/震荡/高波动状态

检测方法：
1. ADX (Average Directional Index) — 趋势强度
2. 波动率百分位 — 当前波动率在历史中的位置
3. 价格相对于均线的位置 — 趋势方向

根据 regime 动态调整策略权重：
- 趋势市 → TSMOM 权重增大，均值回归降低
- 震荡市 → 均值回归/资金费率增大，趋势策略降低
- 高波动 → 所有策略缩减仓位
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


class RegimeDetector:
    """
    市场 regime 检测器

    输入：OHLCV DataFrame
    输出：当前 regime + 各策略权重
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
        vol_lookback: int = 60,
        vol_high_percentile: float = 80.0,
        ma_period: int = 50,
    ):
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.vol_lookback = vol_lookback
        self.vol_high_percentile = vol_high_percentile
        self.ma_period = ma_period

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """
        检测当前市场 regime

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            MarketRegime enum
        """
        if len(df) < max(self.adx_period * 2, self.vol_lookback, self.ma_period) + 10:
            return MarketRegime.RANGING

        # 1. 计算 ADX
        adx = self._compute_adx(df, self.adx_period)
        current_adx = adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0

        # 2. 计算波动率百分位
        returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        rolling_vol = returns.rolling(self.adx_period).std()
        if len(rolling_vol.dropna()) >= self.vol_lookback:
            current_vol = rolling_vol.iloc[-1]
            vol_percentile = (
                (rolling_vol.iloc[-self.vol_lookback:] <= current_vol).sum()
                / self.vol_lookback * 100
            )
        else:
            vol_percentile = 50.0

        # 3. 价格相对于 MA 的位置
        ma = df["close"].rolling(self.ma_period).mean()
        price_above_ma = df["close"].iloc[-1] > ma.iloc[-1]

        # ─── Regime 判断 ───

        # 高波动优先
        if vol_percentile >= self.vol_high_percentile:
            return MarketRegime.HIGH_VOLATILITY

        # ADX 判断趋势 vs 震荡
        if current_adx >= self.adx_trend_threshold:
            if price_above_ma:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN

        if current_adx <= self.adx_range_threshold:
            return MarketRegime.RANGING

        # 中间地带，根据价格位置判断
        if price_above_ma:
            return MarketRegime.TRENDING_UP
        else:
            return MarketRegime.TRENDING_DOWN

    def get_strategy_weights(self, regime: MarketRegime) -> dict[str, float]:
        """
        根据 regime 返回各策略权重

        权重用于缩放仓位大小：
        weight=1.0 表示正常仓位，0.5 表示减半，0.0 表示不交易
        """
        weights = {
            MarketRegime.TRENDING_UP: {
                "tsmom": 1.0,
                "bollinger_trend": 0.8,
                "funding_rate": 0.6,
                "cross_momentum": 0.8,
                "mean_reversion": 0.2,
            },
            MarketRegime.TRENDING_DOWN: {
                "tsmom": 1.0,
                "bollinger_trend": 0.8,
                "funding_rate": 0.6,
                "cross_momentum": 0.8,
                "mean_reversion": 0.2,
            },
            MarketRegime.RANGING: {
                "tsmom": 0.5,            # 0.2→0.5: 震荡市仍允许 TSMOM（减仓但不禁止）
                "bollinger_trend": 0.3,
                "funding_rate": 1.0,
                "cross_momentum": 0.5,
                "mean_reversion": 1.0,
            },
            MarketRegime.HIGH_VOLATILITY: {
                "tsmom": 0.5,
                "bollinger_trend": 0.3,
                "funding_rate": 0.3,
                "cross_momentum": 0.3,
                "mean_reversion": 0.3,
            },
        }
        return weights.get(regime, weights[MarketRegime.RANGING])

    def get_regime_info(self, df: pd.DataFrame) -> dict:
        """返回 regime 详细信息"""
        if len(df) < max(self.adx_period * 2, self.vol_lookback, self.ma_period) + 10:
            return {"regime": "ranging", "adx": 0, "vol_pct": 50}

        adx = self._compute_adx(df, self.adx_period)
        returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        rolling_vol = returns.rolling(self.adx_period).std()

        current_adx = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0
        current_vol = float(rolling_vol.iloc[-1]) if not np.isnan(rolling_vol.iloc[-1]) else 0

        if len(rolling_vol.dropna()) >= self.vol_lookback:
            vol_pct = float(
                (rolling_vol.iloc[-self.vol_lookback:] <= current_vol).sum()
                / self.vol_lookback * 100
            )
        else:
            vol_pct = 50.0

        regime = self.detect(df)

        return {
            "regime": regime.value,
            "adx": round(current_adx, 1),
            "vol_percentile": round(vol_pct, 1),
            "daily_vol": round(current_vol * 100, 2),
        }

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算 ADX"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # +DM / -DM
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)

        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        # Smoothed averages (Wilder's smoothing)
        atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr)

        # DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
        adx = dx.ewm(alpha=1 / period, min_periods=period).mean()

        return adx
