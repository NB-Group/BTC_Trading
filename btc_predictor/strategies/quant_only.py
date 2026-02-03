import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

from btc_predictor.data import get_data
from btc_predictor.utils import LOGGER


@dataclass
class VolumeFilteredSignal:
    signal: str
    action: str
    current_price: float
    timestamp: str
    fast_ema: float
    slow_ema: float
    latest_volume: float
    volume_ma: float
    volume_spike: bool
    strategy: str = "EMA5_20_6H_VolumeFilter"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "action": self.action,
            "current_price": self.current_price,
            "timestamp": self.timestamp,
            "fast_ema": self.fast_ema,
            "slow_ema": self.slow_ema,
            "latest_volume": self.latest_volume,
            "volume_ma": self.volume_ma,
            "volume_spike": self.volume_spike,
            "strategy": self.strategy,
        }


@dataclass
class TrendPullbackSignal:
    signal: str
    action: str
    current_price: float
    timestamp: str
    fast_ema: float
    slow_ema: float
    rsi: float
    atr: float
    volume_spike: bool
    pullback_pct: float
    strategy: str = "TrendPullback_EMA_ATR"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "action": self.action,
            "current_price": self.current_price,
            "timestamp": self.timestamp,
            "fast_ema": self.fast_ema,
            "slow_ema": self.slow_ema,
            "rsi": self.rsi,
            "atr": self.atr,
            "volume_spike": self.volume_spike,
            "pullback_pct": self.pullback_pct,
            "strategy": self.strategy,
        }


@dataclass
class VolatilityBreakoutSignal:
    signal: str
    action: str
    current_price: float
    timestamp: str
    breakout_level: float
    atr: float
    ema_trend: float
    strategy: str = "VolatilityBreakout_ATR"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "action": self.action,
            "current_price": self.current_price,
            "timestamp": self.timestamp,
            "breakout_level": self.breakout_level,
            "atr": self.atr,
            "ema_trend": self.ema_trend,
            "strategy": self.strategy,
        }


def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    if period <= 0:
        raise ValueError("RSI period must be positive.")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def _prepare_ema_dataset(
    symbol: str,
    timeframe: str,
    fast_period: int,
    slow_period: int,
    volume_window: int,
    since: Optional[str] = None,
    limit_padding: int = 5,
) -> Optional[pd.DataFrame]:
    limit = max(slow_period + volume_window + limit_padding, 200)
    price_data = get_data(symbol=symbol, timeframe=timeframe, limit=limit, since=since)
    if price_data is None or price_data.empty:
        LOGGER.warning(f"[{symbol}] 无法在 {timeframe} 时间框架上获取价格数据。")
        return None

    df = price_data.copy()
    df["ema_fast"] = df["close"].ewm(span=fast_period, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_period, adjust=False).mean()
    df["volume_ma"] = df["volume"].rolling(window=volume_window).mean()
    df.dropna(inplace=True)
    if df.empty:
        LOGGER.warning(f"[{symbol}] 计算EMA和成交量均值后数据不足。")
        return None
    return df


def _prepare_breakout_dataset(
    symbol: str,
    timeframe: str,
    breakout_window: int,
    atr_period: int,
    trend_ema_period: int,
    volume_window: int,
    since: Optional[str] = None,
    limit_padding: int = 200,
    data_limit: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    computed_min = breakout_window + atr_period + trend_ema_period + volume_window + limit_padding
    limit = max(data_limit or 0, computed_min, 600)
    price_data = get_data(symbol=symbol, timeframe=timeframe, limit=limit, since=since)
    if price_data is None or price_data.empty:
        LOGGER.warning(f"[{symbol}] 无法获取波动突破所需数据。")
        return None

    df = price_data.copy()
    df["atr"] = _calculate_atr(df, period=atr_period)
    df["rolling_high"] = df["high"].rolling(window=breakout_window).max().shift(1)
    df["rolling_low"] = df["low"].rolling(window=breakout_window).min().shift(1)
    df["ema_trend"] = df["close"].ewm(span=trend_ema_period, adjust=False).mean()
    df["volume_ma"] = df["volume"].rolling(window=volume_window).mean()
    df.dropna(inplace=True)
    if df.empty:
        LOGGER.warning(f"[{symbol}] 波动突破准备数据不足。")
        return None
    return df


def _prepare_trend_pullback_dataset(
    symbol: str,
    timeframe: str,
    fast_period: int,
    slow_period: int,
    rsi_period: int,
    atr_period: int,
    volume_window: int,
    since: Optional[str] = None,
    limit_padding: int = 120,
) -> Optional[pd.DataFrame]:
    limit = max(slow_period + volume_window + rsi_period + atr_period + limit_padding, 400)
    price_data = get_data(symbol=symbol, timeframe=timeframe, limit=limit, since=since)
    if price_data is None or price_data.empty:
        LOGGER.warning(f"[{symbol}] 无法在 {timeframe} 时间框架上获取趋势回调数据。")
        return None

    df = price_data.copy()
    df["ema_fast"] = df["close"].ewm(span=fast_period, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_period, adjust=False).mean()
    df["rsi"] = _calculate_rsi(df["close"], period=rsi_period)
    df["atr"] = _calculate_atr(df, period=atr_period)
    df["volume_ma"] = df["volume"].rolling(window=volume_window).mean()
    df.dropna(inplace=True)

    if df.empty:
        LOGGER.warning(f"[{symbol}] 趋势回调策略计算指标后数据不足。")
        return None
    return df


def calculate_volatility_breakout_signal(
    symbol: str,
    timeframe: str = "4h",
    breakout_window: int = 60,
    atr_period: int = 14,
    trend_ema_period: int = 80,
    volume_window: int = 30,
    atr_buffer: float = 0.5,
    volume_spike_multiplier: float = 3.0,
) -> Optional[Dict[str, Any]]:
    """
    波动率突破策略：
    - price > rolling_high + atr * buffer 且 price > EMA(trend)
    - volume spike 时观望
    - price < rolling_low - atr * buffer 判定 EXIT
    """
    LOGGER.info(
        "[%s] 计算波动突破信号 | timeframe=%s breakout=%s atr=%s ema=%s buffer=%.2f",
        symbol,
        timeframe,
        breakout_window,
        atr_period,
        trend_ema_period,
        atr_buffer,
    )
    df = _prepare_breakout_dataset(
        symbol=symbol,
        timeframe=timeframe,
        breakout_window=breakout_window,
        atr_period=atr_period,
        trend_ema_period=trend_ema_period,
        volume_window=volume_window,
        since=None,
    )
    if df is None or df.empty:
        return None

    latest = df.iloc[-1]
    price = float(latest["close"])
    atr = float(latest["atr"])
    ema_trend = float(latest["ema_trend"])
    breakout_level = float(latest["rolling_high"])
    breakdown_level = float(latest["rolling_low"])
    volume_ma = latest["volume_ma"]
    latest_volume = latest["volume"]

    volume_spike = bool(volume_ma and latest_volume > volume_ma * volume_spike_multiplier)
    breakout = breakout_level and price > breakout_level + atr * atr_buffer and price > ema_trend
    breakdown = breakdown_level and price < breakdown_level - atr * atr_buffer

    signal = "HOLD"
    action = "等待突破或跌破信号"
    if breakout and not volume_spike:
        signal = "BUY"
        action = "价格放量突破近期高点并站上趋势均线，顺势做多"
    elif breakdown:
        signal = "EXIT"
        action = "价格跌破近期低点 - buffer，离场或观望"

    payload = VolatilityBreakoutSignal(
        signal=signal,
        action=action,
        current_price=price,
        timestamp=latest.name.isoformat(),
        breakout_level=float(breakout_level),
        atr=atr,
        ema_trend=ema_trend,
    ).to_dict()
    payload["volume_spike"] = volume_spike
    payload["strategy"] = "VOL_BREAKOUT_4H"
    LOGGER.info("[%s] 波动突破信号: %s", symbol, payload)
    return payload


def backtest_volatility_breakout(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    breakout_window: int = 60,
    atr_period: int = 14,
    trend_ema_period: int = 80,
    volume_window: int = 30,
    atr_buffer: float = 0.5,
    volume_spike_multiplier: float = 3.0,
    atr_stop_multiple: float = 1.2,
    leverage: float = 2.0,
    fee_pct: float = 0.0006,
    initial_capital: float = 10000.0,
    since: Optional[str] = None,
    log_every: int = 120,
) -> Optional[Dict[str, Any]]:
    """
    回测波动率突破策略：放量突破 rolling high 做多，跌破 rolling low 或触发 ATR 止损则离场。
    """
    LOGGER.info(
        "[回测] 启动波动突破策略 | symbol=%s timeframe=%s breakout=%s atr=%s ema=%s buffer=%.2f",
        symbol,
        timeframe,
        breakout_window,
        atr_period,
        trend_ema_period,
        atr_buffer,
    )
    df = _prepare_breakout_dataset(
        symbol=symbol,
        timeframe=timeframe,
        breakout_window=breakout_window,
        atr_period=atr_period,
        trend_ema_period=trend_ema_period,
        volume_window=volume_window,
        since=since,
    )
    if df is None or df.empty:
        return None
    LOGGER.info("[回测] 波动突破数据长度=%s，时间范围 %s → %s", len(df), df.index[0], df.index[-1])

    equity = initial_capital
    peak_equity = equity
    equity_curve: List[float] = [equity]
    trades: List[Dict[str, Any]] = []

    position_open = False
    entry_price = 0.0
    entry_time = None
    stop_price = 0.0

    for idx in range(1, len(df)):
        row = df.iloc[idx]
        price = float(row["close"])
        atr_val = float(row["atr"])
        ema_trend = float(row["ema_trend"])
        breakout_level = row["rolling_high"]
        breakdown_level = row["rolling_low"]
        volume_ma = row["volume_ma"]
        latest_volume = row["volume"]

        volume_spike = bool(volume_ma and latest_volume > volume_ma * volume_spike_multiplier)
        breakout = breakout_level and price > breakout_level + atr_val * atr_buffer and price > ema_trend
        breakdown = bool(breakdown_level and price < breakdown_level - atr_val * atr_buffer)

        if log_every and idx % max(log_every, 1) == 0:
            breakout_display = float(breakout_level) if pd.notna(breakout_level) else float("nan")
            breakdown_display = float(breakdown_level) if pd.notna(breakdown_level) else float("nan")
            LOGGER.info(
                f"[回测] 进度 {idx}/{len(df)} | equity={equity:.2f} pos={position_open} price={price:.2f} "
                f"breakout={breakout_display:.2f} breakdown={breakdown_display:.2f} "
                f"atr={atr_val:.2f} spike={volume_spike}"
            )

        if position_open:
            stop_price = max(stop_price, price - atr_val * atr_stop_multiple)
            exit_stop = price <= stop_price
            exit_break = breakdown or price < ema_trend
            if exit_stop or exit_break:
                change = (price / entry_price) - 1.0
                gross_pnl = equity * leverage * change
                fee_cost = equity * leverage * fee_pct * 2
                pnl = gross_pnl - fee_cost
                equity += pnl
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": row.name,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "pnl": pnl,
                        "return_pct": change * leverage * 100,
                    }
                )
                LOGGER.info(
                    "[回测] 波动突破离场 | entry=%.2f exit=%.2f pnl=%.2f equity=%.2f stop=%.2f reason=%s",
                    entry_price,
                    price,
                    pnl,
                    equity,
                    stop_price,
                    "STOP" if exit_stop else "BREAK",
                )
                position_open = False
                entry_price = 0.0
                entry_time = None
                stop_price = 0.0

        if not position_open and breakout and not volume_spike:
            position_open = True
            entry_price = price
            entry_time = row.name
            stop_price = entry_price - atr_val * atr_stop_multiple
            LOGGER.info(
                "[回测] 波动突破开仓 | price=%.2f breakout=%.2f atr=%.2f ema=%.2f",
                price,
                breakout_level,
                atr_val,
                ema_trend,
            )

        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)

    if position_open:
        last_price = float(df.iloc[-1]["close"])
        change = (last_price / entry_price) - 1.0
        gross_pnl = equity * leverage * change
        fee_cost = equity * leverage * fee_pct
        pnl = gross_pnl - fee_cost
        equity += pnl
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": df.index[-1],
                "entry_price": entry_price,
                "exit_price": last_price,
                "pnl": pnl,
                "return_pct": change * leverage * 100,
            }
        )
        equity_curve.append(equity)

    data_start = df.index[0]
    data_end = df.index[-1]
    data_bars = len(df)
    data_days = (data_end - data_start).total_seconds() / 86400

    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min()) * 100 if not drawdown.empty else 0.0
    # 月化收益率（基于实际数据跨度）
    duration_months = max(
        (df.index[-1] - df.index[0]).total_seconds() / (30 * 24 * 3600), 1e-6
    )
    monthly_return_pct = ((equity / initial_capital) ** (1 / duration_months) - 1) * 100

    trades_df = pd.DataFrame(trades)
    win_rate = float((trades_df["pnl"] > 0).mean() * 100) if not trades_df.empty else 0.0
    avg_gain = float(trades_df.loc[trades_df["pnl"] > 0, "return_pct"].mean()) if not trades_df.empty else 0.0
    avg_loss = float(trades_df.loc[trades_df["pnl"] <= 0, "return_pct"].mean()) if not trades_df.empty else 0.0

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "breakout_window": breakout_window,
        "atr_period": atr_period,
        "trend_ema_period": trend_ema_period,
        "volume_window": volume_window,
        "atr_buffer": atr_buffer,
        "volume_spike_multiplier": volume_spike_multiplier,
        "atr_stop_multiple": atr_stop_multiple,
        "data_start": data_start,
        "data_end": data_end,
        "data_bars": data_bars,
        "data_days": data_days,
        "leverage": leverage,
        "initial_capital": initial_capital,
        "final_equity": equity,
        "total_return_pct": (equity / initial_capital - 1) * 100,
        "monthly_return_pct": monthly_return_pct,
        "max_drawdown_pct": max_drawdown,
        "total_trades": int(len(trades)),
        "win_rate_pct": win_rate,
        "avg_gain_pct": avg_gain,
    }
    LOGGER.info("[回测] 波动突破策略结果: %s", result)
    return result


def calculate_volume_filtered_ema_signal(
    symbol: str,
    timeframe: str = "6h",
    fast_period: int = 5,
    slow_period: int = 20,
    volume_window: int = 20,
    volume_spike_multiplier: float = 2.5,
) -> Optional[Dict[str, Any]]:
    """
    6小时 EMA5/EMA20 金叉做多 + 成交量异常过滤的量化信号。
    - 金叉信号：EMA5 上穿 EMA20 → BUY
    - 死叉信号：EMA5 下穿 EMA20 → EXIT（止损/离场）
    - 成交量过滤：若最新成交量 > rolling_mean(volume) * multiplier，则忽略金叉入场
    """
    LOGGER.info(
        f"[{symbol}] 计算量化-only EMA信号 | timeframe={timeframe} fast={fast_period} slow={slow_period} volume_window={volume_window}"
    )
    df = _prepare_ema_dataset(
        symbol=symbol,
        timeframe=timeframe,
        fast_period=fast_period,
        slow_period=slow_period,
        volume_window=volume_window,
        since=None,
    )
    if df is None or len(df) < 2:
        return None

    LOGGER.info(f"[{symbol}] EMA信号数据长度={len(df)}，最近时间={df.index[-1]}")
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    golden_cross = previous["ema_fast"] <= previous["ema_slow"] and latest["ema_fast"] > latest["ema_slow"]
    death_cross = previous["ema_fast"] >= previous["ema_slow"] and latest["ema_fast"] < latest["ema_slow"]
    volume_ma = latest["volume_ma"]
    latest_volume = latest["volume"]
    volume_spike = bool(volume_ma and latest_volume > volume_ma * volume_spike_multiplier)

    signal = "HOLD"
    action = "等待信号"
    if death_cross:
        signal = "EXIT"
        action = "EMA5 下穿 EMA20，触发死叉止损/离场"
    elif golden_cross:
        if volume_spike:
            signal = "HOLD"
            action = (
                "检测到EMA金叉，但成交量异常放大，触发黑天鹅过滤，暂缓入场"
            )
        else:
            signal = "BUY"
            action = "EMA5 上穿 EMA20，符合做多条件"

    LOGGER.debug(
        f"[{symbol}] EMA状态: fast={latest['ema_fast']:.2f}, slow={latest['ema_slow']:.2f}, "
        f"volume={latest_volume:.2f}, volume_ma={volume_ma:.2f if volume_ma else float('nan')}, "
        f"golden={golden_cross}, death={death_cross}, spike={volume_spike}"
    )

    payload = VolumeFilteredSignal(
        signal=signal,
        action=action,
        current_price=float(latest["close"]),
        timestamp=latest.name.isoformat(),
        fast_ema=float(latest["ema_fast"]),
        slow_ema=float(latest["ema_slow"]),
        latest_volume=float(latest_volume),
        volume_ma=float(volume_ma) if not math.isnan(volume_ma) else float("nan"),
        volume_spike=volume_spike,
    )
    LOGGER.info(f"[{symbol}] 量化-only EMA信号: {payload.to_dict()}")
    return payload.to_dict()


def calculate_trend_pullback_signal(
    symbol: str,
    timeframe: str = "4h",
    fast_period: int = 50,
    slow_period: int = 150,
    rsi_period: int = 14,
    atr_period: int = 14,
    volume_window: int = 30,
    max_pullback_pct: float = 0.04,
    rsi_pullback_level: float = 45.0,
    rsi_rebound_level: float = 50.0,
    volume_spike_multiplier: float = 3.0,
) -> Optional[Dict[str, Any]]:
    """
    趋势回调策略：
    - 长期趋势：EMA_fast > EMA_slow 且 close > EMA_slow
    - 回调：价格较EMA_fast回撤不超过 max_pullback_pct
    - RSI 回踩-反弹：前一根RSI<=rsi_pullback_level，当前>=rsi_rebound_level
    - 成交量异常放大则观望
    """
    LOGGER.info(
        "[%s] 计算趋势回调信号 | timeframe=%s fast=%s slow=%s rsi=%s atr=%s pullback<=%.2f%%",
        symbol,
        timeframe,
        fast_period,
        slow_period,
        rsi_period,
        atr_period,
        max_pullback_pct * 100,
    )
    df = _prepare_trend_pullback_dataset(
        symbol=symbol,
        timeframe=timeframe,
        fast_period=fast_period,
        slow_period=slow_period,
        rsi_period=rsi_period,
        atr_period=atr_period,
        volume_window=volume_window,
        since=None,
    )
    if df is None or len(df) < 2:
        return None

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    ema_fast = float(latest["ema_fast"])
    ema_slow = float(latest["ema_slow"])
    price = float(latest["close"])
    rsi_latest = float(latest["rsi"])
    rsi_prev = float(previous["rsi"])
    atr_latest = float(latest["atr"])
    volume_ma = latest["volume_ma"]
    latest_volume = latest["volume"]

    trend_up = ema_fast > ema_slow and price > ema_slow
    pullback_pct = max((ema_fast - price) / ema_fast, 0.0) if ema_fast else 0.0
    pullback_ok = 0.0 < pullback_pct <= max_pullback_pct
    rsi_rebound = rsi_prev <= rsi_pullback_level and rsi_latest >= rsi_rebound_level
    volume_spike = bool(volume_ma and latest_volume > volume_ma * volume_spike_multiplier)

    signal = "HOLD"
    action = "趋势回调条件未满足，继续观望"
    if trend_up and pullback_ok and rsi_rebound and not volume_spike:
        signal = "BUY"
        action = (
            "EMA趋势向上，价格温和回踩且RSI重新上穿确认，多头试单"
        )
    elif price < ema_slow or not trend_up:
        signal = "EXIT"
        action = "价格跌破慢线或趋势转弱，离场观望"

    payload = TrendPullbackSignal(
        signal=signal,
        action=action,
        current_price=price,
        timestamp=latest.name.isoformat(),
        fast_ema=ema_fast,
        slow_ema=ema_slow,
        rsi=rsi_latest,
        atr=atr_latest,
        volume_spike=volume_spike,
        pullback_pct=pullback_pct,
    )
    LOGGER.info("[%s] 趋势回调信号: %s", symbol, payload.to_dict())
    return payload.to_dict()


def backtest_trend_pullback(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    fast_period: int = 50,
    slow_period: int = 150,
    rsi_period: int = 14,
    atr_period: int = 14,
    volume_window: int = 30,
    max_pullback_pct: float = 0.04,
    rsi_pullback_level: float = 45.0,
    rsi_rebound_level: float = 50.0,
    volume_spike_multiplier: float = 3.0,
    atr_stop_multiple: float = 1.8,
    leverage: float = 2.0,
    fee_pct: float = 0.0006,
    initial_capital: float = 10000.0,
    since: Optional[str] = None,
    log_every: int = 120,
) -> Optional[Dict[str, Any]]:
    """
    回测 EMA 趋势回调策略：趋势向上+温和回调+RSI重新上穿时做多，ATR 跟踪止损。
    """
    LOGGER.info(
        "[回测] 启动趋势回调策略 | symbol=%s timeframe=%s fast=%s slow=%s pullback<=%.2f%%",
        symbol,
        timeframe,
        fast_period,
        slow_period,
        max_pullback_pct * 100,
    )
    df = _prepare_trend_pullback_dataset(
        symbol=symbol,
        timeframe=timeframe,
        fast_period=fast_period,
        slow_period=slow_period,
        rsi_period=rsi_period,
        atr_period=atr_period,
        volume_window=volume_window,
        since=since,
    )
    if df is None or df.empty:
        return None
    LOGGER.info("[回测] 趋势回调数据长度=%s，时间范围 %s → %s", len(df), df.index[0], df.index[-1])

    equity = initial_capital
    peak_equity = equity
    equity_curve: List[float] = [equity]
    trades: List[Dict[str, Any]] = []

    position_open = False
    entry_price = 0.0
    entry_time = None
    stop_price = 0.0

    no_trend = 0
    no_pullback = 0
    no_rsi = 0
    blocked_volume = 0

    for idx in range(1, len(df)):
        prev_row = df.iloc[idx - 1]
        row = df.iloc[idx]

        price = float(row["close"])
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        atr_val = float(row["atr"])
        rsi_prev = float(prev_row["rsi"])
        rsi_latest = float(row["rsi"])
        volume_ma = row["volume_ma"]
        volume_spike = bool(volume_ma and row["volume"] > volume_ma * volume_spike_multiplier)

        trend_up = ema_fast > ema_slow and price > ema_slow
        pullback_pct = max((ema_fast - price) / ema_fast, 0.0) if ema_fast else 0.0
        pullback_ok = 0.0 < pullback_pct <= max_pullback_pct
        rsi_rebound = rsi_prev <= rsi_pullback_level and rsi_latest >= rsi_rebound_level

        if log_every and idx % max(log_every, 1) == 0:
            LOGGER.info(
                f"[回测] 进度 {idx}/{len(df)} | equity={equity:.2f} pos={position_open} price={price:.2f} "
                f"ema_fast={ema_fast:.2f} ema_slow={ema_slow:.2f} pullback={pullback_pct * 100:.2f}% "
                f"rsi={rsi_latest:.2f} volume_spike={volume_spike}"
            )

        # exit logic
        if position_open:
            stop_price = max(stop_price, price - atr_val * atr_stop_multiple)
            exit_trend = price <= ema_slow
            exit_stop = price <= stop_price
            exit_rsi = rsi_latest < rsi_pullback_level - 5
            if exit_trend or exit_stop or exit_rsi:
                change = (price / entry_price) - 1.0
                gross_pnl = equity * leverage * change
                fee_cost = equity * leverage * fee_pct * 2
                pnl = gross_pnl - fee_cost
                equity += pnl
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": row.name,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "pnl": pnl,
                        "return_pct": change * leverage * 100,
                    }
                )
                LOGGER.info(
                    "[回测] 趋势/ATR离场 | entry=%.2f exit=%.2f pnl=%.2f equity=%.2f stop_price=%.2f",
                    entry_price,
                    price,
                    pnl,
                    equity,
                    stop_price,
                )
                position_open = False
                entry_price = 0.0
                entry_time = None
                stop_price = 0.0

        # entry logic
        if not position_open:
            if trend_up and pullback_ok and rsi_rebound and not volume_spike:
                position_open = True
                entry_price = price
                entry_time = row.name
                stop_price = entry_price - atr_val * atr_stop_multiple
                LOGGER.info(
                    "[回测] 趋势回调开仓 | price=%.2f ema_fast=%.2f ema_slow=%.2f rsi=%.2f pullback=%.2f%%",
                    price,
                    ema_fast,
                    ema_slow,
                    rsi_latest,
                    pullback_pct * 100,
                )
            else:
                if not trend_up:
                    no_trend += 1
                elif not pullback_ok:
                    no_pullback += 1
                elif not rsi_rebound:
                    no_rsi += 1
                elif volume_spike:
                    blocked_volume += 1

        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)

    if position_open:
        last_price = float(df.iloc[-1]["close"])
        change = (last_price / entry_price) - 1.0
        gross_pnl = equity * leverage * change
        fee_cost = equity * leverage * fee_pct
        pnl = gross_pnl - fee_cost
        equity += pnl
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": df.index[-1],
                "entry_price": entry_price,
                "exit_price": last_price,
                "pnl": pnl,
                "return_pct": change * leverage * 100,
            }
        )
        equity_curve.append(equity)

    data_start = df.index[0]
    data_end = df.index[-1]
    data_bars = len(df)
    data_days = (data_end - data_start).total_seconds() / 86400

    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min()) * 100 if not drawdown.empty else 0.0
    duration_months = max((df.index[-1] - df.index[0]).total_seconds() / (30 * 24 * 3600), 1e-6)
    monthly_return_pct = ((equity / initial_capital) ** (1 / duration_months) - 1) * 100

    trades_df = pd.DataFrame(trades)
    win_rate = float((trades_df["pnl"] > 0).mean() * 100) if not trades_df.empty else 0.0
    avg_gain = float(trades_df.loc[trades_df["pnl"] > 0, "return_pct"].mean()) if not trades_df.empty else 0.0
    avg_loss = float(trades_df.loc[trades_df["pnl"] <= 0, "return_pct"].mean()) if not trades_df.empty else 0.0

    LOGGER.info(
        "[回测] 触发统计 | trend_fail=%d pullback_fail=%d rsi_fail=%d volume_block=%d",
        no_trend,
        no_pullback,
        no_rsi,
        blocked_volume,
    )

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "fast_period": fast_period,
        "slow_period": slow_period,
        "rsi_period": rsi_period,
        "atr_period": atr_period,
        "max_pullback_pct": max_pullback_pct,
        "rsi_pullback_level": rsi_pullback_level,
        "rsi_rebound_level": rsi_rebound_level,
        "atr_stop_multiple": atr_stop_multiple,
        "volume_spike_multiplier": volume_spike_multiplier,
        "data_start": data_start,
        "data_end": data_end,
        "data_bars": data_bars,
        "data_days": data_days,
        "leverage": leverage,
        "initial_capital": initial_capital,
        "final_equity": equity,
        "total_return_pct": (equity / initial_capital - 1) * 100,
        "monthly_return_pct": monthly_return_pct,
        "max_drawdown_pct": max_drawdown,
        "total_trades": int(len(trades)),
        "win_rate_pct": win_rate,
        "avg_gain_pct": avg_gain,
        "avg_loss_pct": avg_loss,
    }
    LOGGER.info("[回测] 趋势回调策略结果: %s", result)
    return result


def backtest_volume_filtered_ema(
    symbol: str = "BTC/USDT",
    timeframe: str = "6h",
    fast_period: int = 5,
    slow_period: int = 20,
    volume_window: int = 20,
    volume_spike_multiplier: float = 2.5,
    leverage: float = 2.5,
    fee_pct: float = 0.0006,
    initial_capital: float = 10000.0,
    since: Optional[str] = None,
    log_every: int = 150,
) -> Optional[Dict[str, Any]]:
    """
    简易回测：只做多，金叉全仓做多，死叉平仓。成交量异常时忽略入场。
    """
    LOGGER.info(
        f"[回测] 启动 EMA 量化-only 回测 | symbol={symbol} timeframe={timeframe} fast={fast_period} slow={slow_period} "
        f"volume_window={volume_window} volume_multiplier={volume_spike_multiplier} leverage={leverage}"
    )
    df = _prepare_ema_dataset(
        symbol=symbol,
        timeframe=timeframe,
        fast_period=fast_period,
        slow_period=slow_period,
        volume_window=volume_window,
        since=since,
    )
    if df is None or df.empty:
        return None

    LOGGER.info(f"[回测] 数据长度={len(df)}，时间范围 {df.index[0]} → {df.index[-1]}")
    equity = initial_capital
    peak_equity = equity
    equity_curve: List[float] = [equity]
    trades: List[Dict[str, Any]] = []

    position_open = False
    entry_price = 0.0
    entry_time = None

    for current_idx in range(1, len(df)):
        prev_row = df.iloc[current_idx - 1]
        row = df.iloc[current_idx]

        golden_cross = prev_row["ema_fast"] <= prev_row["ema_slow"] and row["ema_fast"] > row["ema_slow"]
        death_cross = prev_row["ema_fast"] >= prev_row["ema_slow"] and row["ema_fast"] < row["ema_slow"]
        volume_spike = bool(row["volume_ma"] and row["volume"] > row["volume_ma"] * volume_spike_multiplier)

        price = float(row["close"])

        if log_every and current_idx % max(log_every, 1) == 0:
            LOGGER.info(
                f"[回测] 进度 {current_idx}/{len(df)} | equity={equity:.2f} | pos_open={position_open} | "
                f"fast={row['ema_fast']:.2f} slow={row['ema_slow']:.2f} spike={volume_spike}"
            )

        if position_open and death_cross:
            change = (price / entry_price) - 1.0
            gross_pnl = equity * leverage * change
            fee_cost = equity * leverage * fee_pct * 2
            pnl = gross_pnl - fee_cost
            equity += pnl
            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": row.name,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl": pnl,
                    "return_pct": change * leverage * 100,
                }
            )
            LOGGER.info(
                f"[回测] 死叉平仓 | 进场={entry_price:.2f} 出场={price:.2f} pnl={pnl:.2f} equity={equity:.2f}"
            )
            position_open = False
            entry_price = 0.0
            entry_time = None

        if not position_open and golden_cross and not volume_spike:
            position_open = True
            entry_price = price
            entry_time = row.name
            LOGGER.info(f"[回测] 金叉开仓 | 价格={price:.2f} 时间={entry_time}")

        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)

    # 强制在结尾平仓
    if position_open:
        last_price = float(df.iloc[-1]["close"])
        change = (last_price / entry_price) - 1.0
        gross_pnl = equity * leverage * change
        fee_cost = equity * leverage * fee_pct
        pnl = gross_pnl - fee_cost
        equity += pnl
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": df.index[-1],
                "entry_price": entry_price,
                "exit_price": last_price,
                "pnl": pnl,
                "return_pct": change * leverage * 100,
            }
        )
        equity_curve.append(equity)

    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min()) * 100 if not drawdown.empty else 0.0
    duration_months = max((df.index[-1] - df.index[0]).total_seconds() / (30 * 24 * 3600), 1e-6)
    monthly_return_pct = ((equity / initial_capital) ** (1 / duration_months) - 1) * 100

    trades_df = pd.DataFrame(trades)
    win_rate = float((trades_df["pnl"] > 0).mean() * 100) if not trades_df.empty else 0.0

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "fast_period": fast_period,
        "slow_period": slow_period,
        "volume_window": volume_window,
        "data_start": data_start,
        "data_end": data_end,
        "data_bars": data_bars,
        "data_days": data_days,
        "volume_spike_multiplier": volume_spike_multiplier,
        "leverage": leverage,
        "initial_capital": initial_capital,
        "final_equity": equity,
        "total_return_pct": (equity / initial_capital - 1) * 100,
        "max_drawdown_pct": max_drawdown,
        "total_trades": int(len(trades)),
        "win_rate_pct": win_rate,
    }

    LOGGER.info(f"[回测] EMA量化-only策略: {result}")
    return result
