import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from btc_predictor.data import get_exchange
from btc_predictor.utils import LOGGER


def fetch_orderbook(symbol: str, depth: int = 5, max_retries: int = 2, retry_delay: float = 0.5) -> Optional[Dict]:
    """
    拉取当前盘口，返回 ccxt order book 结构。
    带重试机制处理限流。
    优化：减少重试次数和延迟，提高响应速度。
    """
    for attempt in range(max_retries):
        try:
            ex = get_exchange()
            if ex is None:
                LOGGER.warning("[HF-OB] 交易所初始化失败，无法获取盘口。")
                return None
            ob = ex.fetch_order_book(symbol, limit=depth)
            return ob
        except Exception as e:
            error_str = str(e)
            # 检查是否是限流错误
            if "50011" in error_str or "Rate limit" in error_str or "Too many requests" in error_str:
                wait_time = retry_delay * (attempt + 1)  # 指数退避，但更短
                if attempt < max_retries - 1:
                    # 记录每次限流和重试
                    LOGGER.warning(
                        f"[HF-OB] API限流 (错误码: 50011) - "
                        f"第 {attempt+1} 次尝试失败，等待 {wait_time:.1f}秒后重试 ({attempt+1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # 最终失败
                    LOGGER.error(
                        f"[HF-OB] 获取盘口失败（限流，已重试{max_retries}次，最终放弃） - "
                        f"错误: {error_str[:100]}"
                    )
                    return None
            else:
                # 非限流错误，直接返回
                if attempt == 0:  # 只在第一次尝试时记录
                    LOGGER.error(f"[HF-OB] 获取盘口失败: {e}")
                return None
    return None


def _sum_depth(levels: List[List[float]], n: int) -> float:
    """累加前 n 档数量。"""
    if not levels:
        return 0.0
    return float(sum(x[1] for x in levels[:n] if len(x) >= 2))


def compute_obi(orderbook: Dict, levels: int = 3) -> Optional[Dict[str, float]]:
    """
    计算 Order Book Imbalance (OBI):
      buy_obi = sum(bid_qty_1..n)
      sell_obi = sum(ask_qty_1..n)
      ratio = buy_obi / sell_obi
    """
    try:
        bids = orderbook.get("bids") or []
        asks = orderbook.get("asks") or []
        buy = _sum_depth(bids, levels)
        sell = _sum_depth(asks, levels)
        if sell == 0:
            return None
        ratio = buy / sell
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        spread_bps = (
            ((best_ask - best_bid) / mid * 10000) if mid and best_bid and best_ask else None
        )
        return {
            "buy_depth": buy,
            "sell_depth": sell,
            "ratio": ratio,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
            "spread_bps": spread_bps,
        }
    except Exception as e:
        LOGGER.warning(f"[HF-OB] 计算OBI失败: {e}")
        return None


def generate_obi_signal(
    symbol: str,
    depth: int = 5,
    levels: int = 3,
    upper: float = 1.3,
    lower: float = 0.77,
    min_depth_total: float = 1.0,
    max_spread_bps: float = 2.0,
) -> Optional[Dict]:
    """
    基于盘口失衡生成信号：
      ratio > upper → LONG
      ratio < lower → SHORT
      否则 HOLD
    过滤条件：
      - buy_depth + sell_depth >= min_depth_total
      - spread_bps <= max_spread_bps
    """
    ob = fetch_orderbook(symbol, depth)
    if not ob:
        return None
    obi = compute_obi(ob, levels)
    if not obi or obi.get("ratio") is None:
        return None

    total_depth = (obi.get("buy_depth") or 0) + (obi.get("sell_depth") or 0)
    if total_depth < min_depth_total:
        return None

    spread_bps = obi.get("spread_bps")
    if spread_bps is None or spread_bps > max_spread_bps:
        return None

    ratio = obi["ratio"]
    signal = "HOLD"
    if ratio > upper:
        signal = "BUY"
    elif ratio < lower:
        signal = "SELL"

    return {
        "signal": signal,
        "ratio": ratio,
        "buy_depth": obi["buy_depth"],
        "sell_depth": obi["sell_depth"],
        "best_bid": obi["best_bid"],
        "best_ask": obi["best_ask"],
        "mid": obi["mid"],
        "spread_bps": spread_bps,
        "params": {
            "upper": upper,
            "lower": lower,
            "levels": levels,
            "depth": depth,
            "min_depth_total": min_depth_total,
            "max_spread_bps": max_spread_bps,
        },
    }


def record_orderbook_series(
    symbol: str = "BTC/USDT",
    samples: int = 120,
    interval_sec: float = 2.0,  # 默认改为2秒，避免限流
    depth: int = 5,
    levels: int = 3,
) -> pd.DataFrame:
    """
    简易录制近期盘口序列，用于快速"伪回测"。
    注意：这不是历史回测，只是短时采样。
    
    注意：OKX API限流较严格，建议interval_sec >= 2.0秒以避免限流。
    """
    rows: List[Dict] = []
    consecutive_failures = 0
    max_consecutive_failures = 5
    current_interval = interval_sec
    
    for i in range(samples):
        ob = fetch_orderbook(symbol, depth, max_retries=3, retry_delay=3.0)
        if ob:
            consecutive_failures = 0  # 重置连续失败计数
            current_interval = interval_sec  # 重置间隔
            obi = compute_obi(ob, levels)
            ts = ob.get("timestamp") or int(time.time() * 1000)
            if obi:
                rows.append(
                    {
                        "timestamp": ts,
                        "ratio": obi["ratio"],
                        "buy_depth": obi["buy_depth"],
                        "sell_depth": obi["sell_depth"],
                        "best_bid": obi["best_bid"],
                        "best_ask": obi["best_ask"],
                        "mid": obi["mid"],
                    }
                )
        else:
            consecutive_failures += 1
            # 遇到限流时，逐步增加等待时间（指数退避）
            if consecutive_failures > 0:
                current_interval = min(interval_sec * (2 ** consecutive_failures), 30.0)  # 最多30秒
                LOGGER.warning(f"[HF-OB] 连续失败{consecutive_failures}次，将等待间隔调整为 {current_interval:.1f}秒")
            if consecutive_failures >= max_consecutive_failures:
                LOGGER.warning(f"[HF-OB] 连续失败{consecutive_failures}次，可能遇到严重限流，等待10秒后继续...")
                time.sleep(10.0)
                consecutive_failures = 0  # 重置计数
        
        # 等待指定间隔（限流时会自动增加）
        time.sleep(current_interval)
        
        if (i + 1) % 50 == 0:
            LOGGER.info(f"[HF-OB] 录制进度: {i+1}/{samples}, 已收集 {len(rows)} 条有效数据, 当前间隔 {current_interval:.1f}秒")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def backtest_obi(
    df: pd.DataFrame,
    upper: float = 1.3,
    lower: float = 0.77,
    tp_pct: float = 0.0003,
    sl_pct: float = 0.0006,
    maker_rebate: float = 0.0,
    taker_fee: float = 0.0005,
    use_taker: bool = False,
    min_depth_total: float = 1.0,
    max_spread_bps: float = 2.0,
    verbose: bool = False,
    log_every: int = 50,
    slippage_bps: float = 0.5,
) -> Optional[Dict[str, float]]:
    """
    在已录制的盘口序列上做简化回测（近似）：
      - 信号基于 ratio
      - 开仓价：best_ask（BUY）/ best_bid（SELL）假设挂单成交
      - 平仓：达到止盈tp_pct或止损sl_pct；无则在下次反向信号平
      - 费用：开平各收一次。use_taker=True 时按 taker_fee 扣费；否则用 maker_rebate（可为负表示返佣）

    局限：无真实撮合，假设挂单立即成交；仅用于快速验证思路。
    """
    if df is None or df.empty or "ratio" not in df:
        LOGGER.warning("[HF-OB] 回测数据不足")
        return None

    df = df.dropna(subset=["best_bid", "best_ask", "ratio"]).copy()
    if df.empty:
        LOGGER.warning("[HF-OB] 回测数据缺少盘口价格")
        return None

    cash = 1.0
    pos = 0  # 1=long, -1=short
    entry = 0.0
    trades = []
    wins = 0

    for _, row in df.iterrows():
        bid = float(row.best_bid)
        ask = float(row.best_ask)
        mid = float(row.mid) if row.mid else (bid + ask) / 2
        ratio = float(row.ratio)
        spread_bps = float(row.spread_bps) if "spread_bps" in row and row.spread_bps is not None else None
        total_depth = float(row.buy_depth + row.sell_depth)

        # 基本过滤：深度和点差
        if total_depth < min_depth_total:
            sig = "HOLD"
        elif spread_bps is not None and spread_bps > max_spread_bps:
            sig = "HOLD"
        else:
            sig = "HOLD"
            if ratio > upper:
                sig = "BUY"
            elif ratio < lower:
                sig = "SELL"


        # 持仓风控（tp/sl）
        if pos != 0 and entry > 0:
            move = (mid - entry) / entry if pos > 0 else (entry - mid) / entry
            if move >= tp_pct or move <= -sl_pct:
                fee = -taker_fee if use_taker else maker_rebate
                slip = abs(slippage_bps) / 10000 * 2  # 进出各一次
                move_real = move - slip
                cash *= 1 + move_real + fee
                trades.append(move_real)
                if move > 0:
                    wins += 1
                if verbose and len(trades) % max(log_every, 1) == 0:
                    LOGGER.info(
                        f"[HF-OB] trade#{len(trades)} tp/sl exit move={move_real:.5f} cash={cash:.5f} win_rate={wins/len(trades):.3f}"
                    )
                pos = 0
                entry = 0.0
                continue

        # 反向/离场
        if pos != 0:
            if (pos > 0 and sig == "SELL") or (pos < 0 and sig == "BUY"):
                move = (mid - entry) / entry if pos > 0 else (entry - mid) / entry
                fee = -taker_fee if use_taker else maker_rebate
                slip = abs(slippage_bps) / 10000 * 2
                move_real = move - slip
                cash *= 1 + move_real + fee
                trades.append(move_real)
                if move > 0:
                    wins += 1
                if verbose and len(trades) % max(log_every, 1) == 0:
                    LOGGER.info(
                        f"[HF-OB] trade#{len(trades)} reverse exit move={move_real:.5f} cash={cash:.5f} win_rate={wins/len(trades):.3f}"
                    )
                pos = 0
                entry = 0.0

        # 开仓
        if pos == 0 and sig in ("BUY", "SELL"):
            pos = 1 if sig == "BUY" else -1
            entry = ask if sig == "BUY" else bid  # 假设挂单在当前价成交
            fee = -taker_fee if use_taker else maker_rebate
            cash *= 1 + fee

    # 收尾：强制平仓
    if pos != 0 and entry > 0:
        last_mid = float(df.iloc[-1].mid)
        move = (last_mid - entry) / entry if pos > 0 else (entry - last_mid) / entry
        fee = -taker_fee if use_taker else maker_rebate
        slip = abs(slippage_bps) / 10000 * 2
        move_real = move - slip
        cash *= 1 + move_real + fee
        trades.append(move_real)
        if move > 0:
            wins += 1

    if not trades:
        return {"trades": 0, "pnl_pct": 0.0, "win_rate": 0.0, "final_equity": cash}

    wins = wins if wins else sum(1 for t in trades if t > 0)
    result = {
        "trades": len(trades),
        "win_rate": wins / len(trades),
        "pnl_pct": (cash - 1.0) * 100,
        "final_equity": cash,
    }
    if verbose:
        LOGGER.info(
            f"[HF-OB] final trades={result['trades']} win_rate={result['win_rate']:.3f} pnl_pct={result['pnl_pct']:.4f}% equity={result['final_equity']:.5f}"
        )
    return result


def stream_record_and_backtest(
    symbol: str = "BTC/USDT",
    total_samples: int = 7200,
    interval_sec: float = 1.0,
    chunk_size: int = 300,
    depth: int = 5,
    levels: int = 3,
    upper: float = 1.3,
    lower: float = 0.77,
    tp_pct: float = 0.0003,
    sl_pct: float = 0.0006,
    maker_rebate: float = 0.0,
    taker_fee: float = 0.0005,
    use_taker: bool = False,
    min_depth_total: float = 1.0,
    max_spread_bps: float = 2.0,
    log_every: int = 50,
) -> pd.DataFrame:
    """
    分段录制 + 分段回测（累积数据后每 chunk 回溯一次），实时打印盈利/胜率。
    用于长时间录制时观察策略表现。
    返回完整录制的DataFrame。
    """
    rows: List[Dict] = []
    for i in range(total_samples):
        ob = fetch_orderbook(symbol, depth)
        if ob:
            obi = compute_obi(ob, levels)
            ts = ob.get("timestamp") or int(time.time() * 1000)
            if obi:
                rows.append(
                    {
                        "timestamp": ts,
                        "ratio": obi["ratio"],
                        "buy_depth": obi["buy_depth"],
                        "sell_depth": obi["sell_depth"],
                        "best_bid": obi["best_bid"],
                        "best_ask": obi["best_ask"],
                        "mid": obi["mid"],
                        "spread_bps": obi.get("spread_bps"),
                    }
                )
        # 分段回测
        if rows and ((i + 1) % chunk_size == 0):
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            res = backtest_obi(
                df,
                upper=upper,
                lower=lower,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                maker_rebate=maker_rebate,
                taker_fee=taker_fee,
                use_taker=use_taker,
                min_depth_total=min_depth_total,
                max_spread_bps=max_spread_bps,
                verbose=False,
                log_every=log_every,
            )
            LOGGER.info(
                f"[HF-OB][stream] samples={i+1}/{total_samples} trades={res['trades']} "
                f"win_rate={res['win_rate']:.3f} pnl_pct={res['pnl_pct']:.4f}% equity={res['final_equity']:.5f}"
            )
        time.sleep(interval_sec)

    if not rows:
        return pd.DataFrame()
    df_final = pd.DataFrame(rows)
    df_final["timestamp"] = pd.to_datetime(df_final["timestamp"], unit="ms", utc=True)
    df_final.set_index("timestamp", inplace=True)
    LOGGER.info(f"[HF-OB][stream] 完成录制，样本 {len(df_final)}")
    return df_final


# ============ 高级特征 + 策略 ===============

def _enrich_features(df: pd.DataFrame, mid_ret_lookback: int = 5, ratio_ewm_span: int = 8) -> pd.DataFrame:
    """为盘口序列添加简单的趋势/动量特征。"""
    df = df.copy()
    df["mid_ret"] = df["mid"].pct_change(mid_ret_lookback)
    df["ratio_ema"] = df["ratio"].ewm(span=ratio_ewm_span, adjust=False).mean()
    df["ratio_slope"] = df["ratio_ema"].diff()
    return df


def backtest_obi_advanced(
    df: pd.DataFrame,
    upper: float = 2.0,
    lower: float = 0.5,
    tp_pct: float = 0.0010,
    sl_pct: float = 0.0012,
    maker_rebate: float = 0.0,
    taker_fee: float = 0.00028,
    use_taker: bool = False,
    min_depth_total: float = 3.0,
    max_spread_bps: float = 1.0,
    mid_ret_lookback: int = 5,
    ratio_ewm_span: int = 8,
    mid_ret_floor: float = -0.0002,
    cooldown_bars: int = 10,
    time_stop_bars: int = 30,
    slippage_bps: float = 0.2,
    verbose: bool = False,
    log_every: int = 50,
) -> Optional[Dict[str, float]]:
    """
    高级策略回测：
      - 信号：ratio>upper 且 ratio_slope>0 且 mid_ret>mid_ret_floor -> BUY
              ratio<lower 且 ratio_slope<0 且 mid_ret< -mid_ret_floor -> SELL
      - 过滤：深度、点差
      - 冷却：cooldown_bars 内不再开新仓
      - 时间止盈/止损：持仓超过 time_stop_bars，按当前价强平
    """
    if df is None or df.empty or "ratio" not in df:
        LOGGER.warning("[HF-OB-ADV] 回测数据不足")
        return None

    df = df.dropna(subset=["best_bid", "best_ask", "ratio"]).copy()
    if df.empty:
        LOGGER.warning("[HF-OB-ADV] 回测数据缺少盘口价格")
        return None

    df = _enrich_features(df, mid_ret_lookback, ratio_ewm_span)

    cash = 1.0
    pos = 0  # 1=long, -1=short
    entry = 0.0
    trades: list[float] = []
    wins = 0
    hold_bars = 0
    cooldown = 0

    for _, row in df.iterrows():
        bid = float(row.best_bid)
        ask = float(row.best_ask)
        mid = float(row.mid) if row.mid else (bid + ask) / 2
        ratio = float(row.ratio)
        spread_bps = float(row.spread_bps) if "spread_bps" in row and row.spread_bps is not None else None
        total_depth = float(row.buy_depth + row.sell_depth)
        ratio_slope = row.ratio_slope if "ratio_slope" in row else 0.0
        mid_ret = row.mid_ret if "mid_ret" in row else 0.0

        if cooldown > 0:
            cooldown -= 1

        # 过滤
        if total_depth < min_depth_total or (spread_bps is not None and spread_bps > max_spread_bps):
            sig = "HOLD"
        else:
            sig = "HOLD"
            if ratio > upper and ratio_slope > 0 and mid_ret > mid_ret_floor:
                sig = "BUY"
            elif ratio < lower and ratio_slope < 0 and mid_ret < -mid_ret_floor:
                sig = "SELL"

        # 持仓风控（tp/sl/time-stop）
        if pos != 0 and entry > 0:
            hold_bars += 1
            move = (mid - entry) / entry if pos > 0 else (entry - mid) / entry
            exit_now = False
            if move >= tp_pct or move <= -sl_pct:
                exit_now = True
            elif hold_bars >= time_stop_bars:
                exit_now = True
            if exit_now:
                fee = -taker_fee if use_taker else maker_rebate
                slip = abs(slippage_bps) / 10000 * 2
                move_real = move - slip
                cash *= 1 + move_real + fee
                trades.append(move_real)
                if move_real > 0:
                    wins += 1
                if verbose and len(trades) % max(log_every, 1) == 0:
                    LOGGER.info(
                        f"[HF-OB-ADV] trade#{len(trades)} exit move={move_real:.5f} cash={cash:.5f} win_rate={wins/len(trades):.3f}"
                    )
                pos = 0
                entry = 0.0
                hold_bars = 0
                cooldown = cooldown_bars
                continue

        # 反向/离场
        if pos != 0:
            if (pos > 0 and sig == "SELL") or (pos < 0 and sig == "BUY"):
                move = (mid - entry) / entry if pos > 0 else (entry - mid) / entry
                fee = -taker_fee if use_taker else maker_rebate
                slip = abs(slippage_bps) / 10000 * 2
                move_real = move - slip
                cash *= 1 + move_real + fee
                trades.append(move_real)
                if move_real > 0:
                    wins += 1
                if verbose and len(trades) % max(log_every, 1) == 0:
                    LOGGER.info(
                        f"[HF-OB-ADV] trade#{len(trades)} reverse exit move={move_real:.5f} cash={cash:.5f} win_rate={wins/len(trades):.3f}"
                    )
                pos = 0
                entry = 0.0
                hold_bars = 0
                cooldown = cooldown_bars

        # 开仓
        if pos == 0 and sig in ("BUY", "SELL") and cooldown == 0:
            pos = 1 if sig == "BUY" else -1
            entry = ask if sig == "BUY" else bid
            fee = -taker_fee if use_taker else maker_rebate
            cash *= 1 + fee
            hold_bars = 0

    # 收尾
    if pos != 0 and entry > 0:
        last_mid = float(df.iloc[-1].mid)
        move = (last_mid - entry) / entry if pos > 0 else (entry - last_mid) / entry
        fee = -taker_fee if use_taker else maker_rebate
        slip = abs(slippage_bps) / 10000 * 2
        move_real = move - slip
        cash *= 1 + move_real + fee
        trades.append(move_real)
        if move_real > 0:
            wins += 1

    if not trades:
        return {"trades": 0, "pnl_pct": 0.0, "win_rate": 0.0, "final_equity": cash}

    result = {
        "trades": len(trades),
        "win_rate": wins / len(trades),
        "pnl_pct": (cash - 1.0) * 100,
        "final_equity": cash,
    }
    if verbose:
        LOGGER.info(
            f"[HF-OB-ADV] final trades={result['trades']} win_rate={result['win_rate']:.3f} pnl_pct={result['pnl_pct']:.4f}% equity={result['final_equity']:.5f}"
        )
    return result


def stream_record_and_backtest_adv(
    symbol: str = "BTC/USDT",
    total_samples: int = 3600,
    interval_sec: float = 1.0,
    chunk_size: int = 300,
    depth: int = 5,
    levels: int = 3,
    upper: float = 2.0,
    lower: float = 0.5,
    tp_pct: float = 0.0010,
    sl_pct: float = 0.0012,
    maker_rebate: float = 0.0,
    taker_fee: float = 0.00028,
    use_taker: bool = False,
    min_depth_total: float = 3.0,
    max_spread_bps: float = 1.0,
    mid_ret_lookback: int = 5,
    ratio_ewm_span: int = 8,
    mid_ret_floor: float = -0.0002,
    cooldown_bars: int = 10,
    time_stop_bars: int = 30,
    slippage_bps: float = 0.2,
    log_every: int = 50,
) -> pd.DataFrame:
    """
    分段录制 + 高级策略回测（实时打印）。
    """
    rows: List[Dict] = []
    for i in range(total_samples):
        ob = fetch_orderbook(symbol, depth)
        if ob:
            obi = compute_obi(ob, levels)
            ts = ob.get("timestamp") or int(time.time() * 1000)
            if obi:
                rows.append(
                    {
                        "timestamp": ts,
                        "ratio": obi["ratio"],
                        "buy_depth": obi["buy_depth"],
                        "sell_depth": obi["sell_depth"],
                        "best_bid": obi["best_bid"],
                        "best_ask": obi["best_ask"],
                        "mid": obi["mid"],
                        "spread_bps": obi.get("spread_bps"),
                    }
                )
        if rows and ((i + 1) % chunk_size == 0):
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            res = backtest_obi_advanced(
                df,
                upper=upper,
                lower=lower,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                maker_rebate=maker_rebate,
                taker_fee=taker_fee,
                use_taker=use_taker,
                min_depth_total=min_depth_total,
                max_spread_bps=max_spread_bps,
                mid_ret_lookback=mid_ret_lookback,
                ratio_ewm_span=ratio_ewm_span,
                mid_ret_floor=mid_ret_floor,
                cooldown_bars=cooldown_bars,
                time_stop_bars=time_stop_bars,
                slippage_bps=slippage_bps,
                verbose=False,
                log_every=log_every,
            )
            LOGGER.info(
                f"[HF-OB-ADV][stream] samples={i+1}/{total_samples} trades={res['trades']} "
                f"win_rate={res['win_rate']:.3f} pnl_pct={res['pnl_pct']:.4f}% equity={res['final_equity']:.5f}"
            )
        time.sleep(interval_sec)

    if not rows:
        return pd.DataFrame()
    df_final = pd.DataFrame(rows)
    df_final["timestamp"] = pd.to_datetime(df_final["timestamp"], unit="ms", utc=True)
    df_final.set_index("timestamp", inplace=True)
    LOGGER.info(f"[HF-OB-ADV][stream] 完成录制，样本 {len(df_final)}")
    return df_final


# ============ microprice + OFI 策略（更高质量信号） ===============

def _enrich_micro_ofi_features(df: pd.DataFrame, ofi_span: int = 8) -> pd.DataFrame:
    """
    为盘口序列添加 microprice 与 OFI 近似特征：
      - microprice = (ask*bid_depth + bid*ask_depth) / (bid_depth+ask_depth)
      - micro_bias = (microprice - mid) / mid
      - ofi_proxy = Δbuy_depth - Δsell_depth，之后做 EMA 平滑
    """
    df = df.copy()
    micro_num = df["best_ask"] * df["buy_depth"] + df["best_bid"] * df["sell_depth"]
    micro_den = df["buy_depth"] + df["sell_depth"]
    df["microprice"] = micro_num / micro_den.replace(0, pd.NA)
    df["micro_bias"] = (df["microprice"] - df["mid"]) / df["mid"]
    df["ofi_raw"] = df["buy_depth"].diff().fillna(0) - df["sell_depth"].diff().fillna(0)
    df["ofi_ema"] = df["ofi_raw"].ewm(span=ofi_span, adjust=False).mean()
    return df


def backtest_micro_ofi(
    df: pd.DataFrame,
    bias_long: float = 2e-5,
    bias_short: float = -2e-5,
    ofi_long: float = 0.0,
    ofi_short: float = 0.0,
    tp_pct: float = 0.0010,
    sl_pct: float = 0.0012,
    maker_rebate: float = 0.0,
    taker_fee: float = 0.00028,
    use_taker: bool = False,
    min_depth_total: float = 3.0,
    max_spread_bps: float = 1.0,
    cooldown_bars: int = 10,
    time_stop_bars: int = 30,
    slippage_bps: float = 0.2,
    log_every: int = 50,
    verbose: bool = False,
) -> Optional[Dict[str, float]]:
    """
    基于 microprice 偏移 + OFI 方向的信号：
      BUY: micro_bias > bias_long 且 ofi_ema > ofi_long
      SELL: micro_bias < bias_short 且 ofi_ema < ofi_short
    其余 HOLD。包含深度/点差过滤、冷却、时间止盈、滑点与费用。
    """
    if df is None or df.empty:
        return None
    df = df.dropna(subset=["best_bid", "best_ask", "ratio"]).copy()
    df = _enrich_micro_ofi_features(df)
    df = df.dropna(subset=["micro_bias"])
    if df.empty:
        return None

    cash = 1.0
    pos = 0
    entry = 0.0
    trades: list[float] = []
    wins = 0
    cooldown = 0
    hold_bars = 0

    for _, row in df.iterrows():
        bid = float(row.best_bid)
        ask = float(row.best_ask)
        mid = float(row.mid) if row.mid else (bid + ask) / 2
        spread_bps = float(row.spread_bps) if "spread_bps" in row and row.spread_bps is not None else None
        total_depth = float(row.buy_depth + row.sell_depth)
        bias = float(row.micro_bias)
        ofi = float(row.ofi_ema)

        if cooldown > 0:
            cooldown -= 1

        # 过滤
        if total_depth < min_depth_total or (spread_bps is not None and spread_bps > max_spread_bps):
            sig = "HOLD"
        else:
            # OFI 条件放宽：只要不是强烈反向即可（ofi_long/ofi_short 为负/正时表示允许 OFI 在阈值范围内）
            # 如果 ofi_long=0 且 ofi_short=0，则只用 micro_bias 判断
            if bias > bias_long and (ofi_long == 0.0 or ofi > ofi_long):
                sig = "BUY"
            elif bias < bias_short and (ofi_short == 0.0 or ofi < ofi_short):
                sig = "SELL"
            else:
                sig = "HOLD"

        # 持仓风控（tp/sl/time-stop）
        if pos != 0 and entry > 0:
            hold_bars += 1
            move = (mid - entry) / entry if pos > 0 else (entry - mid) / entry
            exit_now = move >= tp_pct or move <= -sl_pct or hold_bars >= time_stop_bars
            if exit_now:
                fee = -taker_fee if use_taker else maker_rebate
                slip = abs(slippage_bps) / 10000 * 2
                move_real = move - slip
                cash *= 1 + move_real + fee
                trades.append(move_real)
                if move_real > 0:
                    wins += 1
                if verbose and len(trades) % max(log_every, 1) == 0:
                    LOGGER.info(
                        f"[HF-OB-MICRO] trade#{len(trades)} exit move={move_real:.5f} cash={cash:.5f} win_rate={wins/len(trades):.3f}"
                    )
                pos = 0
                entry = 0.0
                hold_bars = 0
                cooldown = cooldown_bars
                continue

        # 反向/离场
        if pos != 0:
            if (pos > 0 and sig == "SELL") or (pos < 0 and sig == "BUY"):
                move = (mid - entry) / entry if pos > 0 else (entry - mid) / entry
                fee = -taker_fee if use_taker else maker_rebate
                slip = abs(slippage_bps) / 10000 * 2
                move_real = move - slip
                cash *= 1 + move_real + fee
                trades.append(move_real)
                if move_real > 0:
                    wins += 1
                if verbose and len(trades) % max(log_every, 1) == 0:
                    LOGGER.info(
                        f"[HF-OB-MICRO] trade#{len(trades)} reverse exit move={move_real:.5f} cash={cash:.5f} win_rate={wins/len(trades):.3f}"
                    )
                pos = 0
                entry = 0.0
                hold_bars = 0
                cooldown = cooldown_bars

        # 开仓
        if pos == 0 and sig in ("BUY", "SELL") and cooldown == 0:
            pos = 1 if sig == "BUY" else -1
            entry = ask if sig == "BUY" else bid
            fee = -taker_fee if use_taker else maker_rebate
            cash *= 1 + fee
            hold_bars = 0

    # 收尾
    if pos != 0 and entry > 0:
        last_mid = float(df.iloc[-1].mid)
        move = (last_mid - entry) / entry if pos > 0 else (entry - last_mid) / entry
        fee = -taker_fee if use_taker else maker_rebate
        slip = abs(slippage_bps) / 10000 * 2
        move_real = move - slip
        cash *= 1 + move_real + fee
        trades.append(move_real)
        if move_real > 0:
            wins += 1

    if not trades:
        return {"trades": 0, "win_rate": 0.0, "pnl_pct": 0.0, "final_equity": cash}

    result = {
        "trades": len(trades),
        "win_rate": wins / len(trades),
        "pnl_pct": (cash - 1.0) * 100,
        "final_equity": cash,
    }
    if verbose:
        LOGGER.info(
            f"[HF-OB-MICRO] final trades={result['trades']} win_rate={result['win_rate']:.3f} pnl_pct={result['pnl_pct']:.4f}% equity={result['final_equity']:.5f}"
        )
    return result


def stream_record_and_backtest_micro(
    symbol: str = "BTC/USDT",
    total_samples: int = 1800,
    interval_sec: float = 1.0,
    chunk_size: int = 300,
    depth: int = 5,
    levels: int = 3,
    bias_long: float = 2e-5,
    bias_short: float = -2e-5,
    ofi_long: float = 0.0,
    ofi_short: float = 0.0,
    tp_pct: float = 0.0010,
    sl_pct: float = 0.0012,
    maker_rebate: float = 0.0,
    taker_fee: float = 0.00028,
    use_taker: bool = False,
    min_depth_total: float = 3.0,
    max_spread_bps: float = 1.0,
    cooldown_bars: int = 10,
    time_stop_bars: int = 30,
    slippage_bps: float = 0.2,
    ofi_span: int = 8,
    log_every: int = 50,
) -> pd.DataFrame:
    """
    分段录制 + microprice/OFI 策略回测（实时打印）。
    """
    rows: List[Dict] = []
    for i in range(total_samples):
        ob = fetch_orderbook(symbol, depth)
        if ob:
            obi = compute_obi(ob, levels)
            ts = ob.get("timestamp") or int(time.time() * 1000)
            if obi:
                rows.append(
                    {
                        "timestamp": ts,
                        "ratio": obi["ratio"],
                        "buy_depth": obi["buy_depth"],
                        "sell_depth": obi["sell_depth"],
                        "best_bid": obi["best_bid"],
                        "best_ask": obi["best_ask"],
                        "mid": obi["mid"],
                        "spread_bps": obi.get("spread_bps"),
                    }
                )
        if rows and ((i + 1) % chunk_size == 0):
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df_feat = _enrich_micro_ofi_features(df, ofi_span=ofi_span)
            res = backtest_micro_ofi(
                df_feat,
                bias_long=bias_long,
                bias_short=bias_short,
                ofi_long=ofi_long,
                ofi_short=ofi_short,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                maker_rebate=maker_rebate,
                taker_fee=taker_fee,
                use_taker=use_taker,
                min_depth_total=min_depth_total,
                max_spread_bps=max_spread_bps,
                cooldown_bars=cooldown_bars,
                time_stop_bars=time_stop_bars,
                slippage_bps=slippage_bps,
                verbose=False,
                log_every=log_every,
            )
            LOGGER.info(
                f"[HF-OB-MICRO][stream] samples={i+1}/{total_samples} trades={res['trades']} "
                f"win_rate={res['win_rate']:.3f} pnl_pct={res['pnl_pct']:.4f}% equity={res['final_equity']:.5f}"
            )
        time.sleep(interval_sec)

    if not rows:
        return pd.DataFrame()
    df_final = pd.DataFrame(rows)
    df_final["timestamp"] = pd.to_datetime(df_final["timestamp"], unit="ms", utc=True)
    df_final.set_index("timestamp", inplace=True)
    LOGGER.info(f"[HF-OB-MICRO][stream] 完成录制，样本 {len(df_final)}")
    return df_final
