import os
from typing import Dict, Any, Optional

import pandas as pd

from btc_predictor.data import get_data
from btc_predictor.utils import LOGGER


def fetch_1s_ohlcv(symbol: str, limit: int = 240) -> pd.DataFrame:
    """
    获取1s级别的OHLCV数据。
    默认拉取最近4分钟(240根)数据，便于做超短线信号。
    """
    try:
        df = get_data(symbol=symbol, timeframe="1s", limit=limit)
        if df is None or df.empty:
            LOGGER.warning(f"[HF] 未获取到 {symbol} 的1s数据")
            return pd.DataFrame()
        return df
    except Exception as e:
        LOGGER.error(f"[HF] 获取1s数据失败: {e}")
        return pd.DataFrame()


def _calc_basic_indicators(df: pd.DataFrame, fast: int, slow: int, mom: int) -> pd.DataFrame:
    """在1s数据上计算快速均线、慢速均线与动量。"""
    df = df.copy()
    df["ma_fast"] = df["close"].rolling(fast, min_periods=fast).mean()
    df["ma_slow"] = df["close"].rolling(slow, min_periods=slow).mean()
    df["momentum"] = df["close"].diff(mom)
    return df


def generate_hf_signal(
    symbol: str,
    limit: int = 240,
    fast: int = 5,
    slow: int = 20,
    momentum_window: int = 10,
    momentum_thresh: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    生成1s级别的高频信号（纯量化，不依赖LLM）。

    规则（保守示例，可再迭代）：
      - BUY: ma_fast > ma_slow 且 momentum > momentum_thresh
      - SELL: ma_fast < ma_slow 且 momentum < -momentum_thresh
      - 否则 HOLD
    """
    df = fetch_1s_ohlcv(symbol, limit=limit)
    if df.empty or len(df) < max(slow, momentum_window):
        return None

    df = _calc_basic_indicators(df, fast=fast, slow=slow, mom=momentum_window)
    latest = df.iloc[-1]

    price = float(latest["close"])
    ma_fast_val = float(latest["ma_fast"])
    ma_slow_val = float(latest["ma_slow"])
    mom_val = float(latest["momentum"])

    signal = "HOLD"
    reason = "均线与动量未形成共振"

    if pd.notna(ma_fast_val) and pd.notna(ma_slow_val) and pd.notna(mom_val):
        if ma_fast_val > ma_slow_val and mom_val > momentum_thresh:
            signal = "BUY"
            reason = f"快线高于慢线且短期动量>阈值({momentum_thresh})"
        elif ma_fast_val < ma_slow_val and mom_val < -momentum_thresh:
            signal = "SELL"
            reason = f"快线低于慢线且短期动量<-阈值({momentum_thresh})"

    return {
        "signal": signal,
        "current_price": price,
        "ma_fast": ma_fast_val,
        "ma_slow": ma_slow_val,
        "momentum": mom_val,
        "params": {
            "limit": limit,
            "fast": fast,
            "slow": slow,
            "momentum_window": momentum_window,
            "momentum_thresh": momentum_thresh,
        },
        "info": reason,
    }


def run_once(symbol: str = "BTC/USDT") -> None:
    """便捷测试入口：打印最新1s高频信号。"""
    res = generate_hf_signal(symbol)
    LOGGER.info(f"[HF] {symbol} 1s 信号: {res}")
    if res:
        print(res)


def backtest_hf(
    symbol: str = "BTC/USDT",
    limit: int = 3600,
    fast: int = 5,
    slow: int = 20,
    momentum_window: int = 10,
    momentum_thresh: float = 0.0,
    fee_pct: float = 0.0005,
) -> Optional[Dict[str, Any]]:
    """
    简易1s级回测：使用同一规则在过去N根K线上模拟。
    - 仅做多/做空单向持仓，反向信号直接反手。
    - 入场/反手与出场均按下一根close成交，忽略滑点。
    - 返回交易统计与最后净值。
    """
    df = fetch_1s_ohlcv(symbol, limit=limit)
    if df.empty or len(df) < max(slow, momentum_window) + 1:
        LOGGER.warning("[HF] 回测数据不足")
        return None

    df = _calc_basic_indicators(df, fast=fast, slow=slow, mom=momentum_window)

    def decide(row):
        if pd.isna(row.ma_fast) or pd.isna(row.ma_slow) or pd.isna(row.momentum):
            return "HOLD"
        if row.ma_fast > row.ma_slow and row.momentum > momentum_thresh:
            return "BUY"
        if row.ma_fast < row.ma_slow and row.momentum < -momentum_thresh:
            return "SELL"
        return "HOLD"

    df["signal_raw"] = df.apply(decide, axis=1)
    # 使用上一根信号作为本根执行指令，避免前视
    df["signal_exec"] = df["signal_raw"].shift(1).fillna("HOLD")

    cash = 1.0  # 以1单位初始资金计
    pos = 0     # 1=多头，-1=空头，0=空仓
    entry = 0.0
    trades = []

    closes = df["close"].values
    signals = df["signal_exec"].values

    for i in range(len(df)):
        sig = signals[i]
        px = float(closes[i])

        # 平仓逻辑
        if pos != 0 and ((sig == "BUY" and pos < 0) or (sig == "SELL" and pos > 0) or sig == "HOLD"):
            pnl = (px - entry) / entry * pos
            cash *= (1 + pnl - fee_pct)
            trades.append(pnl)
            pos = 0
            entry = 0.0

        # 开/反手
        if sig == "BUY" and pos == 0:
            pos = 1
            entry = px
        elif sig == "SELL" and pos == 0:
            pos = -1
            entry = px

    # 若收盘仍有持仓，按最后价平仓
    if pos != 0 and entry > 0:
        px = float(closes[-1])
        pnl = (px - entry) / entry * pos
        cash *= (1 + pnl - fee_pct)
        trades.append(pnl)

    if not trades:
        return {
            "symbol": symbol,
            "trades": 0,
            "win_rate": 0.0,
            "final_equity": cash,
            "pnl_pct": 0.0,
        }

    wins = sum(1 for t in trades if t > 0)
    return {
        "symbol": symbol,
        "trades": len(trades),
        "win_rate": wins / len(trades),
        "final_equity": cash,
        "pnl_pct": (cash - 1.0) * 100,
        "params": {
            "limit": limit,
            "fast": fast,
            "slow": slow,
            "momentum_window": momentum_window,
            "momentum_thresh": momentum_thresh,
            "fee_pct": fee_pct,
        },
    }


if __name__ == "__main__":
    run_once()

