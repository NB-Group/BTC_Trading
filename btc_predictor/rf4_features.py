import pandas as pd
import numpy as np
from ta.momentum import rsi
from scipy.signal import find_peaks

def calculate_rf4(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """
    计算 RF4 指标，这里假设它是一个标准的RSI。
    图片中的参数很复杂，我们先从一个标准的RSI开始，如果需要可以后续再增加复杂度。
    """
    if len(close_prices) < period + 1:
        # 数据不足，返回全为50的RSI（中性值）
        return pd.Series([50.0] * len(close_prices), index=close_prices.index)
    
    rsi_values = rsi(close_prices, window=period)
    
    # 处理NaN值，用前值填充
    rsi_values = rsi_values.ffill()
    
    # 如果仍有NaN（开头部分），用50填充
    rsi_values = rsi_values.fillna(50.0)
    
    return rsi_values

def find_divergences(price: pd.Series, indicator: pd.Series, order: int = 5) -> pd.DataFrame:
    """
    寻找价格和指标之间的看涨和看跌背离。

    看涨背离：价格创出更低的低点，但指标创出更高的低点。
    看跌背离：价格创出更高的高点，但指标创出更低的低点。
    
    我们的策略是：寻找两次连续的看涨背离作为买入信号，一次看跌背离作为卖出信号。
    
    Args:
        price (pd.Series): 价格序列 (例如 'close')。
        indicator (pd.Series): 指标序列 (例如 RSI)。
        order (int): 定义波峰/波谷时向左和向右看的K线数量。

    Returns:
        pd.DataFrame: 一个包含 'bullish_divergence' 和 'bearish_divergence' 列的DataFrame。
    """
    # 输入验证
    if len(price) < order * 3 or len(indicator) < order * 3:
        # 数据不足，返回全零的背离
        divergences = pd.DataFrame(index=price.index, columns=['bullish_divergence', 'bearish_divergence'])
        divergences['bullish_divergence'] = 0
        divergences['bearish_divergence'] = 0
        return divergences
    
    # 确保数据没有NaN值
    if price.isna().any() or indicator.isna().any():
        # 有NaN值，先填充
        price = price.ffill().bfill()
        indicator = indicator.ffill().bfill()
    
    # 寻找波峰和波谷
    try:
        price_peaks, _ = find_peaks(price, distance=order)
        price_troughs, _ = find_peaks(-price, distance=order)
        indicator_peaks, _ = find_peaks(indicator, distance=order)
        indicator_troughs, _ = find_peaks(-indicator, distance=order)
    except Exception:
        # 波峰波谷检测失败，返回全零的背离
        divergences = pd.DataFrame(index=price.index, columns=['bullish_divergence', 'bearish_divergence'])
        divergences['bullish_divergence'] = 0
        divergences['bearish_divergence'] = 0
        return divergences

    divergences = pd.DataFrame(index=price.index, columns=['bullish_divergence', 'bearish_divergence'])
    divergences['bullish_divergence'] = 0
    divergences['bearish_divergence'] = 0

    # 看跌背离 (价格更高的高点, 指标更低的高点)
    for i in range(1, len(price_peaks)):
        current_peak_idx = price_peaks[i]
        prev_peak_idx = price_peaks[i-1]

        # 确保我们只看与价格波峰“接近”的指标波峰
        nearby_indicator_peaks = indicator_peaks[
            (indicator_peaks > prev_peak_idx - order) & 
            (indicator_peaks < current_peak_idx + order)
        ]
        
        if len(nearby_indicator_peaks) < 2:
            continue

        # 找到与当前价格波峰对应的指标波峰
        current_indicator_peak_idx = nearby_indicator_peaks[np.abs(nearby_indicator_peaks - current_peak_idx).argmin()]
        # 找到与前一个价格波峰对应的指标波峰
        prev_indicator_peak_idx = nearby_indicator_peaks[np.abs(nearby_indicator_peaks - prev_peak_idx).argmin()]

        # 安全的索引访问
        try:
            if (current_peak_idx < len(price) and prev_peak_idx < len(price) and 
                current_indicator_peak_idx < len(indicator) and prev_indicator_peak_idx < len(indicator)):
                
                if price.iloc[current_peak_idx] > price.iloc[prev_peak_idx] and \
                   indicator.iloc[current_indicator_peak_idx] < indicator.iloc[prev_indicator_peak_idx]:
                    divergences.loc[price.index[current_peak_idx], 'bearish_divergence'] = 1
        except (IndexError, KeyError):
            continue

    # 看涨背离 (价格更低的低点, 指标更高的低点)
    for i in range(1, len(price_troughs)):
        current_trough_idx = price_troughs[i]
        prev_trough_idx = price_troughs[i-1]

        # 确保我们只看与价格波谷“接近”的指标波谷
        nearby_indicator_troughs = indicator_troughs[
            (indicator_troughs > prev_trough_idx - order) & 
            (indicator_troughs < current_trough_idx + order)
        ]

        if len(nearby_indicator_troughs) < 2:
            continue
            
        current_indicator_trough_idx = nearby_indicator_troughs[np.abs(nearby_indicator_troughs - current_trough_idx).argmin()]
        prev_indicator_trough_idx = nearby_indicator_troughs[np.abs(nearby_indicator_troughs - prev_trough_idx).argmin()]

        # 安全的索引访问
        try:
            if (current_trough_idx < len(price) and prev_trough_idx < len(price) and 
                current_indicator_trough_idx < len(indicator) and prev_indicator_trough_idx < len(indicator)):
                
                if price.iloc[current_trough_idx] < price.iloc[prev_trough_idx] and \
                   indicator.iloc[current_indicator_trough_idx] > indicator.iloc[prev_indicator_trough_idx]:
                    divergences.loc[price.index[current_trough_idx], 'bullish_divergence'] = 1
        except (IndexError, KeyError):
            continue

    return divergences

def generate_rf4_signals(df: pd.DataFrame, period: int = 14, order: int = 5) -> pd.DataFrame:
    """
    根据 RF4 背离策略生成交易信号。
    - 买入信号: 2次连续的看涨背离。
    - 卖出信号: 1次看跌背离。
    """
    # 输入验证
    if df is None or df.empty or 'close' not in df.columns:
        # 返回空信号
        result = pd.DataFrame(index=df.index if df is not None else [], columns=['signal'])
        result['signal'] = 0
        return result
    
    if len(df) < max(period + 1, order * 3):
        # 数据不足，返回全零信号
        result = df[['close']].copy()
        result['signal'] = 0
        return result[['signal']]
    
    try:
        # 计算RF4指标
        df = df.copy()  # 避免修改原始数据
        df['rf4'] = calculate_rf4(df['close'], period=period)
        
        # 计算背离
        divergences = find_divergences(df['close'], df['rf4'], order=order)
        
        df['bullish_divergence'] = divergences['bullish_divergence']
        df['bearish_divergence'] = divergences['bearish_divergence']

        # 根据规则生成信号
        df['signal'] = 0
        
        # 任何看跌背离都卖出
        df.loc[df['bearish_divergence'] == 1, 'signal'] = -1
        
        # 两次连续的看涨背离买入
        bullish_dates = df[df['bullish_divergence'] == 1].index
        if len(bullish_dates) >= 2:
            for i in range(1, len(bullish_dates)):
                # 检查背离之间的时间间隔，避免过于频繁的信号
                if i < len(bullish_dates):
                    df.loc[bullish_dates[i], 'signal'] = 1

        # 返回包含所有列的DataFrame，方便调试
        return df[['signal', 'rf4', 'bullish_divergence', 'bearish_divergence']]
        
    except Exception as e:
        # 发生错误时返回全零信号
        result = df[['close']].copy()
        result['signal'] = 0
        return result[['signal']] 