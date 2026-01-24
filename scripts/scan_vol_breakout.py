import argparse
import csv
import sys
from itertools import product
from pathlib import Path
from typing import List

from loguru import logger as LOGGER

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from btc_predictor.strategies import backtest_volatility_breakout  # noqa: E402


def _parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(',') if v.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(',') if v.strip()]


def scan_vol_breakout(args: argparse.Namespace) -> None:
    breakout_windows = _parse_int_list(args.breakout_windows)
    atr_buffers = _parse_float_list(args.atr_buffers)
    atr_stops = _parse_float_list(args.atr_stop_multiples)
    ema_periods = _parse_int_list(args.trend_ema_periods)
    volume_multipliers = _parse_float_list(args.volume_multipliers)

    param_product = list(
        product(breakout_windows, atr_buffers, atr_stops, ema_periods, volume_multipliers)
    )
    LOGGER.info(f"共有 {len(param_product)} 组参数需要扫描")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'breakout_window', 'atr_buffer', 'atr_stop_multiple', 'trend_ema_period', 'volume_spike_multiplier',
            'final_equity', 'total_return_pct', 'max_drawdown_pct', 'total_trades', 'win_rate_pct',
        ])

        for idx, (breakout_window, atr_buffer, atr_stop, ema_period, vol_mult) in enumerate(param_product, start=1):
            LOGGER.info(
                "[%s/%s] 扫描参数 | breakout=%s atr_buffer=%.2f atr_stop=%.2f ema=%s volume_mult=%.2f",
                idx,
                len(param_product),
                breakout_window,
                atr_buffer,
                atr_stop,
                ema_period,
                vol_mult,
            )
            result = backtest_volatility_breakout(
                symbol=args.symbol,
                timeframe=args.timeframe,
                breakout_window=breakout_window,
                atr_period=args.atr_period,
                trend_ema_period=ema_period,
                volume_window=args.volume_window,
                atr_buffer=atr_buffer,
                volume_spike_multiplier=vol_mult,
                atr_stop_multiple=atr_stop,
                leverage=args.leverage,
                fee_pct=args.fee,
                initial_capital=args.capital,
                since=args.since,
                log_every=args.log_every,
            )
            if not result:
                LOGGER.warning("参数组合失败，跳过")
                continue

            writer.writerow([
                breakout_window,
                atr_buffer,
                atr_stop,
                ema_period,
                vol_mult,
                f"{result['final_equity']:.2f}",
                f"{result['total_return_pct']:.2f}",
                f"{result['max_drawdown_pct']:.2f}",
                result['total_trades'],
                f"{result['win_rate_pct']:.2f}",
            ])

    LOGGER.success(f"扫描完成，结果已写入 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="批量扫描波动突破策略参数")
    parser.add_argument('--symbol', default='BTC/USDT', help='交易对 (默认: BTC/USDT)')
    parser.add_argument('--timeframe', default='4h', help='K线周期 (默认: 4h)')
    parser.add_argument('--atr-period', type=int, default=14, help='ATR周期 (默认: 14)')
    parser.add_argument('--volume-window', type=int, default=30, help='成交量均线窗口 (默认: 30)')
    parser.add_argument('--leverage', type=float, default=2.0, help='名义杠杆 (默认: 2)')
    parser.add_argument('--fee', type=float, default=0.0006, help='单边手续费比例 (默认: 0.0006)')
    parser.add_argument('--capital', type=float, default=10000, help='初始资金 (默认: 10000)')
    parser.add_argument('--since', type=str, default='2020-01-01T00:00:00Z', help='回测开始时间 (默认: 2020-01-01)')
    parser.add_argument('--log-every', type=int, default=0, help='日志输出频率 (默认: 0=关闭)')

    parser.add_argument('--breakout-windows', default='40,60,80', help='扫描的突破窗口列表，逗号分隔')
    parser.add_argument('--atr-buffers', default='0.4,0.5,0.6', help='扫描的ATR缓冲倍数列表，逗号分隔')
    parser.add_argument('--atr-stop-multiples', default='1.0,1.2,1.5', help='扫描的ATR止损倍数列表，逗号分隔')
    parser.add_argument('--trend-ema-periods', default='60,80,100', help='扫描的趋势EMA周期列表，逗号分隔')
    parser.add_argument('--volume-multipliers', default='2.5,3.0,3.5', help='扫描的成交量过滤倍数列表，逗号分隔')

    parser.add_argument('--output', default='results/vol_breakout_scan.csv', help='结果输出文件 (默认: results/vol_breakout_scan.csv)')

    args = parser.parse_args()
    scan_vol_breakout(args)


if __name__ == '__main__':
    main()
