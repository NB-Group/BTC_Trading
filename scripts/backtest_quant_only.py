import argparse
import sys
from pathlib import Path

from loguru import logger as LOGGER

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from btc_predictor.strategies import (  # noqa: E402
    backtest_volume_filtered_ema,
    backtest_trend_pullback,
    backtest_volatility_breakout,
)


def main():
    parser = argparse.ArgumentParser(description="量化-only 策略回测器")
    parser.add_argument(
        "--strategy",
        choices=["ema", "trend_pullback", "vol_breakout"],
        default="ema",
        help="选择回测策略: ema / trend_pullback / vol_breakout (默认: ema)",
    )
    parser.add_argument("--symbol", default="BTC/USDT", help="交易对 (默认: BTC/USDT)")
    parser.add_argument("--timeframe", default="6h", help="K线周期 (默认: 6h)")
    parser.add_argument("--fast", type=int, default=5, help="快速EMA周期 (默认: 5)")
    parser.add_argument("--slow", type=int, default=20, help="慢速EMA周期 (默认: 20)")
    parser.add_argument("--volume-window", type=int, default=20, help="成交量均线窗口 (默认: 20)")
    parser.add_argument("--volume-multiplier", type=float, default=2.5, help="成交量过滤倍数 (默认: 2.5)")
    parser.add_argument("--leverage", type=float, default=2.5, help="名义杠杆 (默认: 2.5)")
    parser.add_argument("--fee", type=float, default=0.0006, help="单边手续费比例 (默认: 0.0006)")
    parser.add_argument("--capital", type=float, default=10000, help="初始资金 (默认: 10000)")
    parser.add_argument("--since", type=str, default=None, help="回测起始时间 (ISO8601，如 2024-01-01T00:00:00Z)")
    # Trend pullback specific
    parser.add_argument("--rsi-period", type=int, default=14, help="[趋势回调] RSI周期 (默认: 14)")
    parser.add_argument("--atr-period", type=int, default=14, help="[趋势回调] ATR周期 (默认: 14)")
    parser.add_argument(
        "--max-pullback",
        type=float,
        default=0.04,
        help="[趋势回调] 允许回调幅度 (默认: 0.04 = 4%)",
    )
    parser.add_argument(
        "--rsi-pullback-level",
        type=float,
        default=45.0,
        help="[趋势回调] RSI下限 (默认: 45)",
    )
    parser.add_argument(
        "--rsi-rebound-level",
        type=float,
        default=50.0,
        help="[趋势回调] RSI回升确认水平 (默认: 50)",
    )
    parser.add_argument(
        "--atr-stop-multiple",
        type=float,
        default=1.8,
        help="[趋势回调] ATR 跟踪止损倍数 (默认: 1.8)",
    )
    # Vol breakout specific
    parser.add_argument(
        "--breakout-window",
        type=int,
        default=60,
        help="[波动突破] 高低点回看窗口 (默认: 60)",
    )
    parser.add_argument(
        "--trend-ema-period",
        type=int,
        default=80,
        help="[波动突破] 趋势EMA周期 (默认: 80)",
    )
    parser.add_argument(
        "--atr-buffer",
        type=float,
        default=0.5,
        help="[波动突破] ATR突破缓冲倍数 (默认: 0.5)",
    )
    parser.add_argument(
        "--atr-stop-breakout",
        type=float,
        default=1.2,
        help="[波动突破] ATR止损倍数 (默认: 1.2)",
    )
    args = parser.parse_args()

    if args.strategy == "ema":
        result = backtest_volume_filtered_ema(
            symbol=args.symbol,
            timeframe=args.timeframe,
            fast_period=args.fast,
            slow_period=args.slow,
            volume_window=args.volume_window,
            volume_spike_multiplier=args.volume_multiplier,
            leverage=args.leverage,
            fee_pct=args.fee,
            initial_capital=args.capital,
            since=args.since,
        )
    elif args.strategy == "trend_pullback":
        result = backtest_trend_pullback(
            symbol=args.symbol,
            timeframe=args.timeframe,
            fast_period=args.fast,
            slow_period=args.slow,
            rsi_period=args.rsi_period,
            atr_period=args.atr_period,
            volume_window=args.volume_window,
            max_pullback_pct=args.max_pullback,
            rsi_pullback_level=args.rsi_pullback_level,
            rsi_rebound_level=args.rsi_rebound_level,
            volume_spike_multiplier=args.volume_multiplier,
            atr_stop_multiple=args.atr_stop_multiple,
            leverage=args.leverage,
            fee_pct=args.fee,
            initial_capital=args.capital,
            since=args.since,
        )
    else:
        result = backtest_volatility_breakout(
            symbol=args.symbol,
            timeframe=args.timeframe,
            breakout_window=args.breakout_window,
            atr_period=args.atr_period,
            trend_ema_period=args.trend_ema_period,
            volume_window=args.volume_window,
            atr_buffer=args.atr_buffer,
            volume_spike_multiplier=args.volume_multiplier,
            atr_stop_multiple=args.atr_stop_breakout,
            leverage=args.leverage,
            fee_pct=args.fee,
            initial_capital=args.capital,
            since=args.since,
        )

    if not result:
        LOGGER.error("回测失败，未能生成结果。")
        return

    print(f"\n=== {args.strategy.upper()} 策略回测结果 ===")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
