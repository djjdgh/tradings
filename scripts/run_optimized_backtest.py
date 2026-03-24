"""
回测 — TSMOM 日线 + 4h 双时间框架优化

用法:
    PYTHONPATH=. python scripts/run_optimized_backtest.py

    首次需先下载 4h 数据:
    bash scripts/download_4h_data.sh
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


# ─── TSMOM Parameter Sweep ───
async def param_sweep_tsmom(bars, symbol, timeframe, config, ewma_spans):
    """TSMOM 参数扫描"""
    print(f"\n  {'─'*60}")
    print(f"  TSMOM 参数扫描 — {symbol} ({timeframe})")
    print(f"  {'─'*60}")
    print(f"  {'thresh':>7} {'sl_atr':>7} {'tp_atr':>7} | {'ret':>7} {'dd':>6} {'sharpe':>7} {'trades':>7} {'wr':>6} {'pf':>5}")
    print(f"  {'─'*70}")

    best_sharpe = -999
    best_params = {}

    for signal_threshold in [0.05, 0.10, 0.15, 0.20, 0.25]:
        for sl_atr in [2.0, 3.0, 4.0, 5.0]:
            tp_atr = sl_atr * 2  # 固定 2:1 盈亏比
            cfg = {
                "ewma_spans": ewma_spans,
                "signal_weights": [1.0 / len(ewma_spans)] * len(ewma_spans),
                "signal_threshold": signal_threshold,
                "atr_period": 14,
                "stop_loss_atr": sl_atr,
                "take_profit_atr": tp_atr,
            }

            strategy = TSMOMStrategy("tsmom", cfg)
            strategy.add_symbol(symbol)
            strategy.add_timeframe(timeframe)

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
        print(f"\n  ★ 最优: thresh={best_params['thresh']}, "
              f"sl={best_params['sl_atr']}, tp={best_params['tp_atr']}")
        print(f"    收益={r.total_return:+.1%}, 回撤={r.max_drawdown:.1%}, "
              f"夏普={r.sharpe_ratio:.2f}, 交易={r.total_trades}, 胜率={r.win_rate:.1%}")

    return best_params


async def run_tsmom_suite(loader, config, symbols, exchange, timeframe, ewma_spans, label):
    """完整的 TSMOM 测试套件: 参数扫描 → 最优参数全币种回测"""

    # ─── 参数扫描（前 3 个币种） ───
    print(f"\n{'='*70}")
    print(f"  {label} — 参数扫描 (每币种 20 组合)")
    print(f"{'='*70}")

    best_params_all = {}
    for sym in symbols[:3]:
        bars = loader.load_bars(sym, exchange, timeframe)
        if bars:
            bp = await param_sweep_tsmom(bars, sym, timeframe, config, ewma_spans)
            if bp:
                best_params_all[sym] = bp

    # ─── 用最优参数跑全币种 (Regime-aware) ───
    print(f"\n{'='*70}")
    print(f"  {label} — Regime-aware 全币种回测")
    print(f"{'='*70}")

    # 选择最佳参数（BTC 优先）
    ref_sym = symbols[0]
    if ref_sym in best_params_all:
        bp = best_params_all[ref_sym]
        tsmom_cfg = {
            "ewma_spans": ewma_spans,
            "signal_weights": [1.0 / len(ewma_spans)] * len(ewma_spans),
            "signal_threshold": bp["thresh"],
            "atr_period": 14,
            "stop_loss_atr": bp["sl_atr"],
            "take_profit_atr": bp["tp_atr"],
        }
        print(f"  参数: thresh={bp['thresh']}, sl={bp['sl_atr']}, tp={bp['tp_atr']}")
    else:
        # 默认参数
        tsmom_cfg = {
            "ewma_spans": ewma_spans,
            "signal_weights": [1.0 / len(ewma_spans)] * len(ewma_spans),
            "signal_threshold": 0.15,
            "atr_period": 14,
            "stop_loss_atr": 3.0,
            "take_profit_atr": 6.0,
        }
        print(f"  使用默认参数 (无数据或扫描无结果)")

    results = []
    for sym in symbols:
        bars = loader.load_bars(sym, exchange, timeframe)
        if not bars:
            print(f"  {sym:<20} 无 {timeframe} 数据")
            continue

        inner = TSMOMStrategy("tsmom", tsmom_cfg)
        inner.add_symbol(sym)
        inner.add_timeframe(timeframe)

        strategy = RegimeAwareStrategy(inner, "tsmom")
        strategy.add_symbol(sym)
        strategy.add_timeframe(timeframe)

        rm = RiskManager(config)
        engine = BacktestEngine(initial_capital=5000, commission_rate=0.0004, slippage_pct=0.0005)
        r = await engine.run(strategy, bars, rm)
        results.append((sym, r))

        print(f"  {sym:<20} return={r.total_return:+7.1%}  dd={r.max_drawdown:6.1%}  "
              f"sharpe={r.sharpe_ratio:6.2f}  trades={r.total_trades:3d}  "
              f"wr={r.win_rate:5.1%}  pf={r.profit_factor:5.2f}")

    if results:
        returns = [r.total_return for _, r in results]
        sharpes = [r.sharpe_ratio for _, r in results]
        total_trades = sum(r.total_trades for _, r in results)
        print(f"\n  {'汇总':<20} avg_ret={np.mean(returns):+7.1%}  "
              f"avg_sharpe={np.mean(sharpes):6.2f}  "
              f"total_trades={total_trades}  "
              f"best={max(returns):+.1%}  worst={min(returns):+.1%}")

    return results


# ─── Main ───
async def main():
    loader = DataLoader("data/klines")
    config = yaml.safe_load(open("config/settings.example.yaml"))

    symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
               "DOGE/USDT:USDT", "LINK/USDT:USDT", "AVAX/USDT:USDT"]
    exchange = "binanceusdm"

    print("=" * 70)
    print("  TSMOM 双时间框架回测 — 日线 + 4h")
    print("=" * 70)

    # ═══════════════════════════════════════════
    # 1. Regime 分析
    # ═══════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  1. 市场 Regime 分析")
    print(f"{'='*70}")

    detector = RegimeDetector()
    for sym in symbols:
        bars = loader.load_bars(sym, exchange, "1d")
        if not bars:
            continue
        df = pd.DataFrame([{"open": b.open, "high": b.high, "low": b.low,
                            "close": b.close, "volume": b.volume} for b in bars])
        info = detector.get_regime_info(df)
        print(f"  {sym:<20} regime={info['regime']:<16} "
              f"ADX={info['adx']:>5.1f}  vol_pct={info['vol_percentile']:>5.1f}%")

    # ═══════════════════════════════════════════
    # 2. TSMOM 日线
    # ═══════════════════════════════════════════
    daily_results = await run_tsmom_suite(
        loader, config, symbols, exchange,
        timeframe="1d",
        ewma_spans=[8, 16, 32, 64],
        label="2. TSMOM 日线",
    )

    # ═══════════════════════════════════════════
    # 3. TSMOM 4h（如果有数据）
    # ═══════════════════════════════════════════
    test_bars = loader.load_bars(symbols[0], exchange, "4h")
    if test_bars:
        h4_results = await run_tsmom_suite(
            loader, config, symbols, exchange,
            timeframe="4h",
            ewma_spans=[12, 24, 48, 96],
            label="3. TSMOM 4h",
        )
    else:
        print(f"\n{'='*70}")
        print("  3. TSMOM 4h — 无数据，请先运行:")
        print("     bash scripts/download_4h_data.sh")
        print(f"{'='*70}")
        h4_results = []

    # ═══════════════════════════════════════════
    # 组合汇总
    # ═══════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  组合汇总")
    print(f"{'='*70}")

    all_strats = {
        "TSMOM 日线": [r for _, r in daily_results],
        "TSMOM 4h": [r for _, r in h4_results] if h4_results else [],
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

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
