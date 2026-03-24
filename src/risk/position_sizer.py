"""
波动率目标仓位管理 — Carver 框架

核心公式：
position_size = (target_vol × capital) / (instrument_vol × price)

参考：Robert Carver - "Systematic Trading" (2015)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from loguru import logger

from src.risk.volatility import VolatilityEstimator


@dataclass
class PositionSizerConfig:
    """仓位管理配置"""
    target_annual_vol: float = 0.80      # 目标年化波动率 (激进: 80%)
    max_leverage: float = 5.0            # 最大杠杆倍数
    min_leverage: float = 0.1            # 最小杠杆倍数
    max_risk_per_trade_pct: float = 0.03 # 单笔最大风险 (3%)
    max_total_position_pct: float = 0.60 # 最大总仓位占比
    rebalance_threshold: float = 0.20    # 仓位偏离 20% 才重平衡


class PositionSizer:
    """
    波动率目标仓位管理器

    根据 Carver 的框架，通过目标波动率和当前波动率的比值
    动态调整仓位大小，使组合波动率保持在目标水平。
    """

    def __init__(self, config: PositionSizerConfig):
        self.config = config
        self._vol_estimators: dict[str, VolatilityEstimator] = {}

    def get_or_create_estimator(
        self,
        symbol: str,
        lambda_fast: float = 0.94,
        lambda_slow: float = 0.97,
    ) -> VolatilityEstimator:
        """获取或创建指定品种的波动率估计器"""
        if symbol not in self._vol_estimators:
            self._vol_estimators[symbol] = VolatilityEstimator(
                lambda_fast=lambda_fast,
                lambda_slow=lambda_slow,
            )
        return self._vol_estimators[symbol]

    def update_price(self, symbol: str, price: float) -> None:
        """更新价格（K线闭合时调用）"""
        estimator = self.get_or_create_estimator(symbol)
        estimator.update(price)

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        capital: float,
        signal_strength: float = 1.0,
        stop_loss_distance: float = 0.0,
    ) -> float:
        """
        计算目标仓位大小（合约数量）

        Args:
            symbol: 交易品种
            price: 当前价格
            capital: 可用资金
            signal_strength: 信号强度 (0-1)
            stop_loss_distance: 止损距离（价格单位），用于风险限制

        Returns:
            target_amount: 合约数量
        """
        if price <= 0 or capital <= 0:
            return 0.0

        estimator = self.get_or_create_estimator(symbol)

        if not estimator.is_ready:
            # 波动率数据不足时，使用保守的默认仓位
            default_amount = (capital * 0.1) / price  # 10% 仓位
            logger.debug(f"Vol not ready for {symbol}, using default size")
            return default_amount

        # ─── 方法 1：波动率目标法 (Carver) ───
        instrument_vol = estimator.annual_vol
        if instrument_vol <= 0:
            instrument_vol = 0.80  # 默认 80% 年化波动率

        # 目标杠杆 = target_vol / instrument_vol
        target_leverage = self.config.target_annual_vol / instrument_vol

        # 限制杠杆范围
        target_leverage = max(
            self.config.min_leverage,
            min(target_leverage, self.config.max_leverage)
        )

        # 基础仓位 = 杠杆 × 资金 / 价格
        vol_based_amount = (target_leverage * capital * signal_strength) / price

        # ─── 方法 2：风险限制法（如果有止损距离） ───
        if stop_loss_distance > 0:
            max_loss = capital * self.config.max_risk_per_trade_pct
            risk_based_amount = max_loss / stop_loss_distance
            # 取两种方法的较小值
            target_amount = min(vol_based_amount, risk_based_amount)
        else:
            target_amount = vol_based_amount

        # ─── 总仓位限制 ───
        max_amount = (capital * self.config.max_total_position_pct) / price
        target_amount = min(target_amount, max_amount)

        logger.debug(
            f"PositionSizer [{symbol}]: vol={instrument_vol:.2%}, "
            f"leverage={target_leverage:.2f}x, "
            f"amount={target_amount:.6f}, signal={signal_strength:.2f}"
        )

        return max(target_amount, 0.0)

    def get_vol_info(self, symbol: str) -> dict:
        """获取指定品种的波动率信息"""
        estimator = self._vol_estimators.get(symbol)
        if estimator is None or not estimator.is_ready:
            return {"ready": False}

        return {
            "ready": True,
            "annual_vol_fast": estimator.annual_vol_fast,
            "annual_vol_slow": estimator.annual_vol_slow,
            "daily_vol_fast": estimator.daily_vol_fast,
            "daily_vol_slow": estimator.daily_vol_slow,
            "vol_spike_ratio": estimator.vol_spike_ratio(),
            "har_rv": estimator.har_rv(),
        }
