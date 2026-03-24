"""
回测 — TSMOM 参数优化 + 资金费率策略

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
from src.core.event import Bar, Signal, SignalAction
from src.risk.manager import RiskManager
from src.strategy.base import BaseStrategy
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
        self._cooldown: dict[str, int] = {}
        self._cooldown_bars = 8

    @property
    def required_bars(self) -> int:
        return max(self._inner.required_bars, 200)

    async def on_bar(self, bar: Bar, history: list[Bar]) -> Signal | None:
        key = bar.symbol
        if key in self._cooldown and self._cooldown[key] > 0:
            self._cooldown[key] -= 1
            return None

        df = self.bars_to_dataframe(history)
        regime = self._regime_detector.detect(df)
        weights = self._regime_detector.get_strategy_weights(regime)
        weight = weights.get(self._type, 0.5)

        if weight < 0.3:
            return None

        signal = await self._inner.on_bar(bar, history)
        if signal is None:
            return None

        signal.strength *= weight
        signal.metadata["regime"] = regime.value
        signal.metadata["regime_weight"] = weight
        return signal

    def on_stop_loss(self, symbol: str):
        self._cooldown[symbol] = self._cooldown_bars


# ─── Funding Rate Strategy (backtest version) ───
class FundingRateBacktestStrategy(BaseStrategy):
    """资金费率回测策略 — 费率极端时反向开仓"""

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


# ─── TSMOM Parameter Sweep ───
async def param_sweep_tsmom(bars, symbol, config):
    """TSMOM 参数扫描"""
    print(f"\n  {'─'*60}")
    print(f"  TSMOM 参数扫描 — {symbol}")
    print(f"  {'─'*60}")
    print(f"  {'thresh':>7} {'sl_atr':>7} {'tp_atr':>7} | {'ret':>7} {'dd':>6} {'sharpe':>7} {'trades':>7} {'wr':>6} {'pf':>5}")
    print(f"  {'─'*70}")

    best_sharpe = -999
    best_params = {}

    for signal_threshold in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for sl_atr in [3.0, 4.0, 5.0]:
            tp_atr = sl_atr * 2  # 固定 2:1 盈亏比
            cfg = {
                "ewma_spans": [8, 16, 32, 64],
                "signal_weights": [0.25, 0.25, 0.25, 0.25],
                "signal_threshold": signal_threshold,
                "atr_period": 14,
                "stop_loss_atr": sl_atr,
                "take_profit_atr": tp_atr,
            }

            strategy = TSMOMStrategy("tsmom", cfg)
            strategy.add_symbol(symbol)
            strategy.add_timeframe("1d")

            rm = RiskManager(config)
            engine = BacktestEngine(initial_capital=5000, commission_rate=0.0004, slippage_pct=0.0005)
            r = await engine.run(strategy, bars, rm)

            marker = ""
            if r.sharpe_ratio > best_sharpe and r.total_trades >= 3:
                best_sharpe = r.sharpe_ratio
                best_params = {"thresh": signal_threshold, "sl_atr": sl_atr,
                               "tp_atr": tp_atr, "result": r}
                marker = " ★"

            if r.total_trades > 0:
                print(f"  {signal_threshold:>7.2f} {sl_atr:>7.1f} {tp_atr:>7.1f} | "
                      f"{r.total_return:>+6.1%} {r.max_drawdown:>5.1%} "
                      f"{r.sharpe_ratio:>7.2f} {r.total_trades:>7d} "
                      f"{r.win_rate:>5.1%} {r.profit_factor:>5.2f}{marker}")

    if best_params:
        r = best_params["result"]
        print(f"\n  ★ 最优参数: thresh={best_params['thresh']}, "
              f"sl_atr={best_params['sl_atr']}, tp_atr={best_params['tp_atr']}")
        print(f"    收益={r.total_return:+.1%}, 回撤={r.max_drawdown:.1%}, "
              f"夏普={r.sharpe_ratio:.2f}, 交易={r.total_trades}, 胜率={r.win_rate:.1%}")

    return best_params


# ─── Main ───
async def main():
    loader = DataLoader("data/klines")
    config = yaml.safe_load(open("config/settings.example.yaml"))

    symbols_daily = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
                     "DOGE/USDT:USDT", "LINK/USDT:USDT", "AVAX/USDT:USDT"]
    exchange = "binanceusdm"

    print("=" * 70)
    print("  回测 — TSMOM 参数优化 + 资金费率策略")
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
    # 2. TSMOM 参数扫描 (日线)
    # ═══════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  2. TSMOM 参数扫描 (日线, 每币种 15 组合)")
    print(f"{'='*70}")

    best_tsmom_params = {}
    for sym in symbols_daily[:3]:  # BTC, ETH, SOL 先扫描
        bars = loader.load_bars(sym, exchange, "1d")
        if bars:
            bp = await param_sweep_tsmom(bars, sym, config)
            if bp:
                best_tsmom_params[sym] = bp

    # ═══════════════════════════════════════════
    # 3. TSMOM 最优参数全币种回测 (Regime-aware)
    # ═══════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  3. TSMOM 最优参数回测 (日线, Regime-aware, 全币种)")
    print(f"{'='*70}")

    # 用 BTC 的最优参数（或默认优化版）
    if "BTC/USDT:USDT" in best_tsmom_params:
        bp = best_tsmom_params["BTC/USDT:USDT"]
        tsmom_cfg = {
            "ewma_spans": [8, 16, 32, 64],
            "signal_weights": [0.25, 0.25, 0.25, 0.25],
            "signal_threshold": bp["thresh"],
            "atr_period": 14,
            "stop_loss_atr": bp["sl_atr"],
            "take_profit_atr": bp["tp_atr"],
        }
        print(f"  使用 BTC 最优参数: thresh={bp['thresh']}, sl={bp['sl_atr']}, tp={bp['tp_atr']}")
    else:
        tsmom_cfg = yaml.safe_load(open("config/strategies/tsmom_optimized.yaml"))
        print(f"  使用默认优化参数")

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
        tsmom_results.append((sym, r))

        print(f"  {sym:<20} return={r.total_return:+7.1%}  dd={r.max_drawdown:6.1%}  "
              f"sharpe={r.sharpe_ratio:6.2f}  trades={r.total_trades:3d}  "
              f"wr={r.win_rate:5.1%}  pf={r.profit_factor:5.2f}")

    if tsmom_results:
        returns = [r.total_return for _, r in tsmom_results]
        sharpes = [r.sharpe_ratio for _, r in tsmom_results]
        print(f"\n  {'TSMOM 汇总':<20} avg_ret={np.mean(returns):+7.1%}  "
              f"avg_sharpe={np.mean(sharpes):6.2f}  "
              f"best={max(returns):+.1%}  worst={min(returns):+.1%}")

    # ═══════════════════════════════════════════
    # 4. 资金费率策略 (1h)
    # ═══════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  4. 资金费率反向策略 (1h)")
    print(f"{'='*70}")

    fr_cfg = yaml.safe_load(open("config/strategies/funding_rate.yaml"))
    fr_results = []

    for sym in ["BTC/USDT:USDT", "ETH/USDT:USDT"]:
        safe = sym.replace("/", "_").replace(":", "_")
        fr_path = Path(f"data/klines/{safe}_{exchange}_funding.csv")

        if not fr_path.exists():
            print(f"  {sym}: 无资金费率数据，跳过 (运行 download_funding_rates.py)")
            continue

        funding_df = pd.read_csv(fr_path)
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
        fr_results.append((sym, r))

        print(f"  {sym:<20} return={r.total_return:+7.1%}  dd={r.max_drawdown:6.1%}  "
              f"sharpe={r.sharpe_ratio:6.2f}  trades={r.total_trades:3d}  "
              f"wr={r.win_rate:5.1%}  pf={r.profit_factor:5.2f}")

    if fr_results:
        returns = [r.total_return for _, r in fr_results]
        sharpes = [r.sharpe_ratio for _, r in fr_results]
        print(f"\n  {'资金费率汇总':<20} avg_ret={np.mean(returns):+7.1%}  "
              f"avg_sharpe={np.mean(sharpes):6.2f}")

    # ═══════════════════════════════════════════
    # 组合汇总
    # ═══════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  组合汇总")
    print(f"{'='*70}")

    all_strats = {
        "TSMOM (regime-aware)": [r for _, r in tsmom_results],
        "Funding Rate": [r for _, r in fr_results],
    }

    total_i = total_f = 0
    worst_dd = 0

    for name, results in all_strats.items():
        if not results:
            continue
        avg_r = np.mean([r.total_return for r in results])
        avg_d = np.mean([r.max_drawdown for r in results])
        avg_s = np.mean([r.sharpe_ratio for r in results])
        total_t = sum(r.total_trades for r in results)

        for r in results:
            total_i += r.initial_capital
            total_f += r.final_equity
            worst_dd = max(worst_dd, r.max_drawdown)

        print(f"  {name:<30} avg_ret={avg_r:+6.2%}  avg_dd={avg_d:5.1%}  "
              f"avg_sharpe={avg_s:6.2f}  trades={total_t}")

    if total_i > 0:
        combined = (total_f - total_i) / total_i
        print(f"\n  总投入:  {total_i:>10,.0f} USDT")
        print(f"  总净值:  {total_f:>10,.2f} USDT")
        print(f"  组合收益: {combined:>9.2%}")
        print(f"  最大回撤: {worst_dd:>9.2%}")

    # 盈利判断
    profitable_strats = []
    for name, results in all_strats.items():
        if results and np.mean([r.total_return for r in results]) > 0:
            profitable_strats.append(name)

    if profitable_strats:
        print(f"\n  ✓ 有正收益的策略: {', '.join(profitable_strats)}")
    else:
        print(f"\n  ✗ 所有策略均为负收益")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
