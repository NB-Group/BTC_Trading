
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands, average_true_range
from ta.trend import MACD, adx, AroonIndicator
from ta.momentum import rsi
import logging
from .config import get_model_config
from .utils import LOGGER # 导入我们自定义的LOGGER

def create_features(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    根据给定的DataFrame和模型配置创建技术分析特征。
    """
    model_config = get_model_config(model_name)
    feature_eng_config = model_config.get('feature_engineering', {})
    lags = feature_eng_config.get('lags', [1, 2, 3, 6, 12, 24])
    rolling_windows = feature_eng_config.get('rolling_windows', [6, 12, 24, 48, 72])
        
    df_copy = df.copy()

    # 基本回报率
    df_copy['returns'] = df_copy['close'].pct_change()
    
    # 波动率
    df_copy['volatility'] = df_copy['returns'].rolling(window=24).std()

    # 动量
    df_copy['momentum'] = df_copy['close'].pct_change(periods=5)

    # RSI
    df_copy['rsi'] = rsi(df_copy['close'])

    # 保证high/low/close都是Series
    high = df_copy['high'] if isinstance(df_copy['high'], pd.Series) else pd.Series(df_copy['high'])
    low = df_copy['low'] if isinstance(df_copy['low'], pd.Series) else pd.Series(df_copy['low'])
    close = df_copy['close'] if isinstance(df_copy['close'], pd.Series) else pd.Series(df_copy['close'])
    # ATR (标准14周期)
    df_copy['atr'] = average_true_range(high=high.astype(float), low=low.astype(float), close=close.astype(float), window=14)

    # --- NEW: Advanced Features ---
    # 1. Volatility Ratio
    atr100 = average_true_range(high=high.astype(float), low=low.astype(float), close=close.astype(float), window=100)
    df_copy['atr_ratio'] = df_copy['atr'] / atr100

    # 2. 循环时间特征
    # 兼容linter：用numpy的datetime64提取weekday和hour，确保为numpy数组
    index_as_dt = np.array(df_copy.index).astype('datetime64[ns]')
    try:
        dayofweek = pd.Series(index_as_dt).dt.dayofweek.values
        hour = pd.Series(index_as_dt).dt.hour.values
    except Exception:
        # 兜底方案：用datetime对象提取
        dt_objs = pd.to_datetime(index_as_dt)
        dayofweek = np.array([d.weekday() for d in dt_objs])
        hour = np.array([d.hour for d in dt_objs])
    df_copy['day_sin'] = np.sin(2 * np.pi * dayofweek / 7)
    df_copy['day_cos'] = np.cos(2 * np.pi * dayofweek / 7)
    df_copy['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df_copy['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    # --- END: Advanced Features ---

    # 移动平均线
    # 确保'ma60'总是被计算，因为它是交叉策略的核心
    if 60 not in rolling_windows:
        rolling_windows.append(60)
        
    for window in rolling_windows:
        df_copy[f'ma{window}'] = df_copy['close'].rolling(window=window).mean()
    
    # 保证close一定是Series且不是DataFrame
    close = df_copy['close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if not isinstance(close, pd.Series):
        close = pd.Series(close)
    # MACD
    macd = MACD(close=close.astype(float))
    df_copy['macd'] = macd.macd()
    df_copy['macd_signal'] = macd.macd_signal()
    df_copy['macd_hist'] = macd.macd_diff()

    # 布林带
    bollinger = BollingerBands(close=close.astype(float))
    df_copy['bollinger_upper'] = bollinger.bollinger_hband()
    df_copy['bollinger_lower'] = bollinger.bollinger_lband()
    df_copy['bollinger_mid'] = bollinger.bollinger_mavg()

    # 滞后特征
    for lag in lags:
        df_copy[f'lag_{lag}'] = df_copy['close'].shift(lag)

    # --- 新增：历史回报率/动量特征 ---
    lags = [1, 3, 6, 12, 24, 48, 72]
    for lag in lags:
        # 使用 pct_change 计算回报率
        df_copy[f'return_{lag}h'] = df_copy['close'].pct_change(periods=lag)

    # 由于增加了大量使用过去数据的特征，需要丢弃前面包含NaN的行
    df_copy.dropna(inplace=True)
    
    LOGGER.info(f"特征创建完成。DataFrame 形状: {df_copy.shape}")
    return df_copy

def generate_regression_targets(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """
    为传统的回归任务生成目标变量 y (未来收益率)。
    """
    y = df['close'].pct_change(periods=window).shift(-window)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    return y.rename('target')

def generate_classification_targets(df: pd.DataFrame, window: int = 24, threshold: float = 0.01) -> pd.Series:
    """
    为分类任务生成目标变量 (涨/跌)。
    """
    future_max = df['high'].rolling(window=window).max().shift(-window)
    buy_signal = future_max > (df['close'] * (1 + threshold))
    y = buy_signal.astype(int)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    return y.rename('target')

def generate_crossover_regression_targets(df: pd.DataFrame, main_ma: str = 'ma60') -> pd.DataFrame:
    """
    为事件驱动的交叉回归策略生成基于事件的特征和目标。
    """
    LOGGER.info(f"开始为交叉回归策略生成目标，使用均线: {main_ma}")
    if main_ma not in df.columns:
        raise ValueError(f"DataFrame中缺少主要均线列: {main_ma}")

    price_over_ma = df['close'] - df[main_ma]
    crossovers = np.where((price_over_ma.iloc[1:].values * price_over_ma.iloc[:-1].values) < 0)[0] + 1
    
    if len(crossovers) < 2:
        LOGGER.warning("交叉事件点少于2个，无法生成任何训练样本。")
        return pd.DataFrame()

    LOGGER.info(f"找到 {len(crossovers)} 个交叉事件点。")
    
    events = []
    for i in range(len(crossovers) - 1):
        start_idx = crossovers[i]
        end_idx = crossovers[i+1]
        
        start_price = df.iloc[start_idx]['close']
        period_data = df.iloc[start_idx:end_idx]
        
        is_golden_cross = price_over_ma.iloc[start_idx] > 0

        if is_golden_cross:
            future_peak_price = period_data['high'].max()
            target_return = (future_peak_price - start_price) / start_price if start_price != 0 else 0
        else:
            future_trough_price = period_data['low'].min()
            target_return = (future_trough_price - start_price) / start_price if start_price != 0 else 0
            
        event = df.iloc[start_idx].to_dict()
        event['y'] = target_return
        event['crossover_type'] = 'golden' if is_golden_cross else 'death'
        events.append(event)
        
    if not events:
        LOGGER.warning("未能从数据中生成任何交叉事件。")
        return pd.DataFrame()

    event_df = pd.DataFrame(events)
    event_df.set_index(df.index[crossovers[:len(events)]], inplace=True)
    
    LOGGER.success(f"成功生成 {len(event_df)} 个交叉回归事件。")
    return event_df

def prepare_training_data(df: pd.DataFrame, model_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    根据模型配置，准备最终的特征(X)和目标(y)以供训练。
    """
    model_config = get_model_config(model_name)
    task = model_config.get('task')
    feature_names = model_config.get('features', [])

    LOGGER.info(f"正在为模型 '{model_name}' (任务: {task}) 准备训练数据...")

    if model_name == 'btc-crossover-regression-v1':
        event_df = generate_crossover_regression_targets(df)
        if event_df.empty:
            return pd.DataFrame(), pd.Series(dtype=float)
        y = event_df['y']
        X = df.loc[event_df.index][feature_names]
        # 类型强制
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        return X, y

    # 通用逻辑
    if task == 'regression':
        y = generate_regression_targets(df)
    elif task == 'classification':
        y = generate_classification_targets(df)
    else:
        raise ValueError(f"未知任务类型: {task}")

    X = df[feature_names].copy()
    aligned_X, aligned_y = X.align(y, join='inner', axis=0)
    if not isinstance(aligned_X, pd.DataFrame):
        aligned_X = pd.DataFrame(aligned_X)
    if not isinstance(aligned_y, pd.Series):
        aligned_y = pd.Series(aligned_y)
    if aligned_X.empty:
        LOGGER.warning("对齐后数据为空。")
        return pd.DataFrame(), pd.Series(dtype=float)
    LOGGER.info(f"数据准备完成。特征(X)形状: {aligned_X.shape}, 目标(y)形状: {aligned_y.shape}")
    return aligned_X, aligned_y 