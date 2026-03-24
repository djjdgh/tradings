"""
优化版回测 — Regime 检测 + 参数优化 + 资金费率策略

用法:
    PYTHONPATH=. python scripts/run_optimized_backtest.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, ".")

from src.backtest.data_loader import DataLoader
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.report import generate_report
from src.core.event import Bar, Signal, SignalAction
from src.risk.manager import RiskManager
from src.strategy.base import BaseStrategy
from src.strategy.bollinger_trend import BollingerTrendStrategy
from src.strategy.regime import MarketRegime, RegimeDetector
from src.strategy.tsmom import TSMOMStrategy


# ─── Regime-aware wrapper ───
class RegimeAwareStrategy(BaseStrategy):
    """包装策略，根据 regime 动态调整信号强度"""

    def __init__(self, strategy: BaseStrategy, strategy_type: str):
        super().__init__(strategy.name, strategy.config)
        self._inner = strategy
        self._type = strategy_type
        self._regime_detector = RegimeDetector()
        self._symbols = strategy._symbols
        self._timeframes = strategy._timeframes
        self._cooldown: dict[str, int] = {}  # symbol -> bars remaining
        self._cooldown_bars = 8  # 止损后冷却 8 根K线

    @property
    def required_bars(self) -> int:
        return max(self._inner.required_bars, 200)

    async def on_bar(self, bar: Bar, history: list[Bar]) -> Signal | None:
        # 冷却期检查
        key = bar.symbol
        if key in self._cooldown and self._cooldown[key] > 0:
            self._cooldown[key] -= 1
            return None

        # Regime 检测
        df = self.bars_to_dataframe(history)
        regime = self._regime_detector.detect(df)
        weights = self._regime_detector.get_strategy_weights(regime)
        weight = weights.get(self._type, 0.5)

        # 权重太低则不交易
        if weight < 0.3:
            return None

        # 调用内部策略
        signal = await self._inner.on_bar(bar, history)

        if signal is None:
            return None

        # 应用 regime 权重到信号强度
        signal.strength *= weight
        signal.metadata["regime"] = regime.value
        signal.metadata["regime_weight"] = weight

        return signal

    def on_stop_loss(self, symbol: str):
        """止损触发时设置冷却期"""
        self._cooldown[symbol] = self._cooldown_bars


# ─── Funding Rate Strategy (backtest version) ───
class FundingRateBacktestStrategy(BaseStrategy):
    """
    资金费率回测策略

    使用历史资金费率数据，在费率极端时反向开仓
    """

    def __init__(self, name: str, config: dict, funding_data: pd.DataFrame):
        super().__init__(name, config)
        self._funding = funding_data
        self._high_threshold = config.get("extreme_high_threshold", 0.0005)
        self._low_threshold = config.get("extreme_low_threshold", -0.0003)
        self._atr_period = config.get("atr_period", 14)
        self._sl_atr = config.get("stop_loss_atr", 2.5)
        self._tp_atr = config.get("take_profit_atr", 3.5)

    @property
    def required_bars(self) -> int:
        return 50

    async def on_bar(self, bar: Bar, history: list[Bar]) -> Signal | None:
        # 找到最近的资金费率
        rate = self._get_latest_funding_rate(bar.timestamp)
        if rate is None:
            return None

        df = self.bars_to_dataframe(history)
        if len(df) < self._atr_period + 1:
            return None

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        atr = self._compute_atr(high, low, close, self._atr_period)
        if atr <= 0:
            return None

        price = float(close[-1])

        # 费率极端偏高 → 做空
        if rate >= self._high_threshold:
            strength = min(abs(rate) / self._high_threshold, 1.0) * 0.25
            return Signal(
                symbol=bar.symbol, exchange=bar.exchange,
                strategy=self.name, action=SignalAction.OPEN_SHORT,
                price=price, strength=strength,
                stop_loss=price + self._sl_atr * atr,
                take_profit=price - self._tp_atr * atr,
                metadata={"funding_rate": rate, "type": "high_funding"},
            )

        # 费率极端偏低 → 做多
        if rate <= self._low_threshold:
            strength = min(abs(rate) / abs(self._low_threshold), 1.0) * 0.25
            return Signal(
                symbol=bar.symbol, exchange=bar.exchange,
                strategy=self.name, action=SignalAction.OPEN_LONG,
                price=price, strength=strength,
                stop_loss=price - self._sl_atr * atr,
                take_profit=price + self._tp_atr * atr,
                metadata={"funding_rate": rate, "type": "low_funding"},
            )

        return None

    def _get_latest_funding_rate(self, timestamp: float) -> float | None:
        """获取最近的资金费率"""
        if self._funding.empty:
            return None
        mask = self._funding["timestamp"] <= timestamp
        if not mask.any():
            return None
        return float(self._funding.loc[mask, "funding_rate"].iloc[-1])

    @staticmethod
    def _compute_atr(high, low, close, period):
        tr_list = [max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
                   for i in range(1, len(high))]
        return float(np.mean(tr_list[-period:])) if len(tr_list) >= period else 0


# ─── Parameter sweep ───
async def param_sweep_bollinger(bars, symbol, config):
    """布林带参数扫描"""
    print(f"\n  {'─'*60}")
    print(f"  布林带参数扫描 — {symbol}")
    print(f"  {'─'*60}")
    print(f"  {'std':>5} {'vol_t':>6} {'ma':>5} {'sl_atr':>7} | {'ret':>7} {'dd':>6} {'sharpe':>7} {'trades':>7} {'wr':>6}")
    print(f"  {'─'*65}")

    best_sharpe = -999
    best_params = {}

    for std_dev in [2.0, 2.5, 3.0]:
        for vol_threshold in [1.5, 2.0, 2.5]:
            for trend_ma in [50, 100]:
                for sl_atr in [2.0, 3.0, 4.0]:
                    cfg = {
                        "bollinger": {"period": 20, "std_dev": std_dev},
                        "volume": {"enabled": True, "period": 20, "threshold": vol_threshold},
                        "trend_filter": {"enabled": True, "ma_period": trend_ma},
                        "exit": {"atr_period": 14, "stop_loss_atr": sl_atr,
                                 "trailing_stop_atr": sl_atr * 0.7,
                                 "max_holding_bars": 96, "take_profit_atr": sl_atr * 2},
                    }

                    strategy = BollingerTrendStrategy("bb", cfg)
                    strategy.add_symbol(symbol)
                    strategy.add_timeframe("15m")

                    rm = RiskManager(config)
                    engine = BacktestEngine(initial_capital=5000, commission_rate=0.0004, slippage_pct=0.0005)
                    r = await engine.run(strategy, bars, rm)

                    marker = ""
                    if r["sharpe_ratio"] > best_sharpe:
                        best_sharpe = r["sharpe_ratio"]
                        best_params = {"std": std_dev, "vol_t": vol_threshold,
                                       "ma": trend_ma, "sl_atr": sl_atr, "result": r}
                        marker = " ★"

                    if r["total_trades"] > 5:
                        print(f"  {std_dev:>5.1f} {vol_threshold:>6.1f} {trend_ma:>5d} {sl_atr:>7.1f} | "
                              f"{r['total_return']:>+6.1%} {r['max_drawdown']:>5.1%} "
                              f"{r['sharpe_ratio']:>7.2f} {r['total_trades']:>7d} "
                              f"{r['win_rate']:>5.1%}{marker}")

    if best_params:
        r = best_params["result"]
        print(f"\n  ★ 最优参数: std={best_params['std']}, vol_t={best_params['vol_t']}, "
              f"ma={best_params['ma']}, sl_atr={best_params['sl_atr']}")
        print(f"    收益={r['total_return']:+.1%}, 回撤={r['max_drawdown']:.1%}, "
              f"夏普={r['sharpe_ratio']:.2f}, 交易={r['total_trades']}")

    return best_params


# ─── Main ───
async def main():
    loader = DataLoader("data/klines")
    config = yaml.safe_load(open("config/settings.example.yaml"))

    symbols_daily = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
                     "DOGE/USDT:USDT", "LINK/USDT:USDT", "AVAX/USDT:USDT"]
    symbols_15m = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    exchange = "binanceusdm"

    print("=" * 70)
    print("  优化版回测 — Regime 检测 + 参数优化 + 资金费率")
    print("=" * 70)

    # ═══════════════════════════════════════════
    # 1. Regime 分析
    # ═══════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  1. 市场 Regime 分析")
    print(f"{'='*70}")

    detector = RegimeDetector()
    for sym in symbols_daily:
        bars = loader.load_bars(sym, exchange, "1d")
        if not bars:
            continue
        df = pd.DataFrame([{"open": b.open, "high": b.high, "low": b.low,
                            "close": b.close, "volume": b.volume} for b in bars])
        info = detector.get_regime_info(df)
        print(f"  {sym:<20} regime={info['regime']:<16} "
              f"ADX={info['adx']:>5.1f}  vol_pct={info['vol_percentile']:>5.1f}%")

    # ═══════════════════════════════════════════
    # 2. 布林带参数扫描 (15m)
    # ═══════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  2. 布林带参数扫描 (15m)")
    print(f"{'='*70}")

    for sym in symbols_15m:
        bars = loader.load_bars(sym, exchange, "15m")
        if bars:
            await param_sweep_bollinger(bars, sym, config)

    # ═══════════════════════════════════════════
    # 3. TSMOM 优化版 (日线, Regime-aware)
    # ═══════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  3. TSMOM 优化版 (日线, Regime-aware)")
    print(f"{'='*70}")

    tsmom_cfg = yaml.safe_load(open("config/strategies/tsmom_optimized.yaml"))
    tsmom_results = []

    for sym in symbols_daily:
        bars = loader.load_bars(sym, exchange, "1d")
        if not bars:
            continue

        inner = TSMOMStrategy("tsmom", tsmom_cfg)
        inner.add_symbol(sym)
        inner.add_timeframe("1d")

        strategy = RegimeAwareStrategy(inner, "tsmom")
        strategy.add_symbol(sym)
        strategy.add_timeframe("1d")

        rm = RiskManager(config)
        engine = BacktestEngine(initial_capital=5000, commission_rate=0.0004, slippage_pct=0.0005)
        r = await engine.run(strategy, bars, rm)
        tsmom_results.append(r)

        print(f"  {sym:<20} return={r['total_return']:+7.1%}  dd={r['max_drawdown']:6.1%}  "
              f"sharpe={r['sharpe_ratio']:6.2f}  trades={r['total_trades']:3d}")

    if tsmom_results:
        avg_r = np.mean([r["total_return"] for r in tsmom_results])
        avg_s = np.mean([r["sharpe_ratio"] for r in tsmom_results])
        print(f"  {'AVERAGE':<20} return={avg_r:+7.1%}  sharpe={avg_s:6.2f}")

    # ═══════════════════════════════════════════
    # 4. 布林带优化版 (15m, Regime-aware)
    # ═══════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  4. 布林带优化版 (15m, Regime-aware)")
    print(f"{'='*70}")

    bb_opt_cfg = yaml.safe_load(open("config/strategies/bollinger_trend_optimized.yaml"))
    bb_results = []

    for sym in symbols_15m:
        bars = loader.load_bars(sym, exchange, "15m")
        if not bars:
            continue

        inner = BollingerTrendStrategy("bollinger_trend", bb_opt_cfg)
        inner.add_symbol(sym)
        inner.add_timeframe("15m")

        strategy = RegimeAwareStrategy(inner, "bollinger_trend")
        strategy.add_symbol(sym)
        strategy.add_timeframe("15m")

        rm = RiskManager(config)
        engine = BacktestEngine(initial_capital=5000, commission_rate=0.0004, slippage_pct=0.0005)
        r = await engine.run(strategy, bars, rm)
        bb_results.append(r)

        print(f"  {sym:<20} return={r['total_return']:+7.1%}  dd={r['max_drawdown']:6.1%}  "
              f"sharpe={r['sharpe_ratio']:6.2f}  trades={r['total_trades']:3d}")

    if bb_results:
        avg_r = np.mean([r["total_return"] for r in bb_results])
        avg_s = np.mean([r["sharpe_ratio"] for r in bb_results])
        print(f"  {'AVERAGE':<20} return={avg_r:+7.1%}  sharpe={avg_s:6.2f}")

    # ═══════════════════════════════════════════
    # 5. 资金费率策略 (1h)
    # ═══════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  5. 资金费率反向策略 (1h)")
    print(f"{'='*70}")

    fr_cfg = yaml.safe_load(open("config/strategies/funding_rate.yaml"))
    fr_results = []

    for sym in ["BTC/USDT:USDT", "ETH/USDT:USDT"]:
        # 加载资金费率数据
        safe = sym.replace("/", "_").replace(":", "_")
        fr_path = Path(f"data/klines/{safe}_{exchange}_funding.csv")

        if not fr_path.exists():
            print(f"  {sym}: 无资金费率数据，跳过 (运行 download_funding_rates.py)")
            continue

        funding_df = pd.read_csv(fr_path)

        # 加载 1h K线
        bars = loader.load_bars(sym, exchange, "1h")
        if not bars:
            print(f"  {sym}: 无 1h K线数据")
            continue

        strategy = FundingRateBacktestStrategy("funding_rate", fr_cfg, funding_df)
        strategy.add_symbol(sym)
        strategy.add_timeframe("1h")

        rm = RiskManager(config)
        engine = BacktestEngine(initial_capital=5000, commission_rate=0.0004, slippage_pct=0.0005)
        r = await engine.run(strategy, bars, rm)
        fr_results.append(r)

        print(f"  {sym:<20} return={r['total_return']:+7.1%}  dd={r['max_drawdown']:6.1%}  "
              f"sharpe={r['sharpe_ratio']:6.2f}  trades={r['total_trades']:3d}")

    if fr_results:
        avg_r = np.mean([r["total_return"] for r in fr_results])
        avg_s = np.mean([r["sharpe_ratio"] for r in fr_results])
        print(f"  {'AVERAGE':<20} return={avg_r:+7.1%}  sharpe={avg_s:6.2f}")

    # ═══════════════════════════════════════════
    # 组合汇总
    # ═══════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  组合汇总（优化版 vs 原版对比）")
    print(f"{'='*70}")

    all_results = {
        "TSMOM (regime-aware)": tsmom_results,
        "Bollinger 15m (regime+opt)": bb_results,
        "Funding Rate": fr_results,
    }

    total_i = total_f = 0
    worst_dd = 0

    for name, results in all_results.items():
        if not results:
            continue
        avg_r = np.mean([r["total_return"] for r in results])
        avg_d = np.mean([r["max_drawdown"] for r in results])
        avg_s = np.mean([r["sharpe_ratio"] for r in results])
        total_t = sum(r["total_trades"] for r in results)

        for r in results:
            total_i += r["initial_capital"]
            total_f += r["final_equity"]
            worst_dd = max(worst_dd, r["max_drawdown"])

        print(f"  {name:<30} avg_ret={avg_r:+6.2%}  avg_dd={avg_d:5.1%}  "
              f"avg_sharpe={avg_s:6.2f}  trades={total_t}")

    if total_i > 0:
        combined = (total_f - total_i) / total_i
        print(f"\n  总投入:  {total_i:>10,.0f} USDT")
        print(f"  总净值:  {total_f:>10,.2f} USDT")
        print(f"  组合收益: {combined:>9.2%}")
        print(f"  最大回撤: {worst_dd:>9.2%}")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
