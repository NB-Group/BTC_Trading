import argparse
from datetime import datetime
import os
import sys

# 确保可从任何工作目录运行：将项目根目录加入 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from btc_predictor.utils import setup_logger, LOGGER
from btc_predictor.predict import (
    get_live_trade_signal,
    get_rf4_signal,
    get_bollinger_breakout_signal,
    get_ma_crossover_signal,
)
from btc_predictor.data import get_data
from btc_predictor.config import DATA_CONFIG
from btc_predictor.config import get_model_config


def okx_to_ccxt_symbol(okx_symbol: str) -> str:
    # e.g., 'BTC-USDT-SWAP' -> 'BTC/USDT'
    return okx_symbol.replace('-SWAP', '').replace('-', '/')


def _detect_crosses_by_sign(series_close, series_ma):
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({'close': series_close, 'ma': series_ma}).dropna()
    if df.empty or len(df) < 2:
        return []
    diff = (df['close'] - df['ma']).astype('float64')
    sign = np.sign(diff.values)
    # 将 0 贴线值继承前一非零符号，避免漏检
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i-1]
    changes = (sign[1:] * sign[:-1]) < 0
    pos = np.where(changes)[0] + 1  # 变化发生在当前位置
    events = []
    for p in pos:
        kind = 'GOLDEN' if (sign[p-1] < 0 and sign[p] > 0) else 'DEATH'
        ts = df.index[p]
        price = float(df['close'].iloc[p])
        events.append((ts, kind, price))
    return events


def scan_ma_cross_events(ccxt_symbol: str, model_name: str, lookback: int = 300, ma_override: int | None = None):
    try:
        model_cfg = get_model_config(model_name)
        ma_window = int(ma_override or model_cfg.get('ma_window', 60))

        df = get_data(symbol=ccxt_symbol, timeframe=DATA_CONFIG['timeframe'], limit=ma_window + max(lookback, 200))
        if df is None or df.empty:
            LOGGER.warning("无法获取价格数据，跳过MA交叉扫描。")
            return

        df = df.copy()
        df[f'ma{ma_window}'] = df['close'].rolling(window=ma_window).mean()
        events = _detect_crosses_by_sign(df['close'], df[f'ma{ma_window}'])

        LOGGER.info(f"在最近 {len(df)} 根K线中，检测到 MA{ma_window} 交叉事件 {len(events)} 次：")
        for ts, kind, price in events[-10:]:  # 仅展示最近10次
            LOGGER.info(f"  - {ts.isoformat()} | {kind} | close={price:.2f}")

        # 滑动窗口统计（每 120 根为一段）
        win = 120
        if len(df) > win:
            LOGGER.info(f"滑动窗口统计（窗口={win}）：")
            for start in range(len(df) - win, len(df), win//2):
                seg = df.iloc[max(0, start-win):start]
                if len(seg) < 10:
                    continue
                seg_events = _detect_crosses_by_sign(seg['close'], seg[f'ma{ma_window}'])
                t0 = seg.index[0].isoformat(); t1 = seg.index[-1].isoformat()
                LOGGER.info(f"  - {t0} ~ {t1} | 交叉数: {len(seg_events)}")
    except Exception as e:
        LOGGER.error(f"扫描MA交叉事件失败: {e}")


def main():
    setup_logger()

    parser = argparse.ArgumentParser(description="使用与 main.py 相同API的信号检测测试器")
    parser.add_argument('--symbol', default='BTC-USDT-SWAP', help='OKX合约符号，例如 BTC-USDT-SWAP')
    parser.add_argument('--model', default=None, help='模型名称，默认读取 config.DEFAULTS.model_name')
    parser.add_argument('--lookback', type=int, default=300, help='扫描MA交叉的回看根数')
    parser.add_argument('--ma', type=int, default=None, help='覆盖模型配置的MA窗口，例如 60')
    args = parser.parse_args()

    try:
        import config as root_cfg
        model_name = args.model or root_cfg.DEFAULTS.get('model_name')
    except Exception:
        model_name = args.model or 'btc-crossover-regression-v1'

    okx_symbol = args.symbol
    ccxt_symbol = okx_to_ccxt_symbol(okx_symbol)

    LOGGER.info(f"[测试入口] OKX符号: {okx_symbol} | CCXT符号: {ccxt_symbol} | timeframe={DATA_CONFIG['timeframe']} | model={model_name}")

    # 1) 扫描 MA{ma_window} 交叉事件（与 get_live_trade_signal 一致的交叉定义）
    scan_ma_cross_events(ccxt_symbol, model_name, lookback=args.lookback, ma_override=args.ma)

    # 2) 调用与 main.py 相同的API获取当前时刻信号
    LOGGER.info("\n=== 当前时刻实时信号（与 main.py 相同API）===")
    try:
        live_ml = get_live_trade_signal(model_name=model_name, symbol=ccxt_symbol)
        LOGGER.info(f"live_ml: {live_ml}")
    except Exception as e:
        LOGGER.error(f"live_ml 调用失败: {e}")

    try:
        rf4 = get_rf4_signal(symbol=ccxt_symbol, period=15, order=5)
        LOGGER.info(f"rf4: {rf4}")
    except Exception as e:
        LOGGER.error(f"rf4 调用失败: {e}")

    try:
        bb = get_bollinger_breakout_signal(symbol=ccxt_symbol, window=20, std_dev=2.0)
        LOGGER.info(f"bollinger: {bb}")
    except Exception as e:
        LOGGER.error(f"bollinger 调用失败: {e}")

    try:
        ma = get_ma_crossover_signal(symbol=ccxt_symbol, fast_period=5, slow_period=20)
        LOGGER.info(f"ma_cross: {ma}")
    except Exception as e:
        LOGGER.error(f"ma_cross 调用失败: {e}")


if __name__ == '__main__':
    main()


