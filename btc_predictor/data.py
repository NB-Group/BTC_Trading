# btc_predictor/data.py
import os
import time
import ccxt
import pandas as pd
import requests
import pickle
from datetime import datetime, timezone
from typing import Optional, Tuple # 添加导入
from tqdm import tqdm
# 从模块内config导入其特有的配置
from .config import DATA_CONFIG, PATHS, FEATURE_CONFIG, get_model_config
from .utils import LOGGER
from .features import create_features, prepare_training_data

import config as root_config # 导入根配置

# --- Global State ---
# --- 全局状态 ---
_exchange = None

def get_exchange() -> Optional[ccxt.Exchange]:
    """
    初始化并返回带有代理支持的ccxt交易所实例。
    """
    global _exchange
    if _exchange:
        return _exchange

    try:
        exchange_class = getattr(ccxt, DATA_CONFIG['exchange'])
        
        # 统一从根配置获取代理
        proxy_url = getattr(root_config, 'DEFAULTS', {}).get('proxy_url')
        
        exchange_config = {'options': {'defaultType': 'spot'}} # 默认使用现货市场
        
        if proxy_url:
            LOGGER.info(f"正在为交易所配置代理: {proxy_url}")
            exchange_config['proxies'] = {
                'http': proxy_url,
                'https': proxy_url,
            }
        
        _exchange = exchange_class(exchange_config)
        return _exchange
    except AttributeError:
        LOGGER.error(f"不支持的交易所: {DATA_CONFIG['exchange']}")
        return None
    except Exception as e:
        LOGGER.error(f"初始化ccxt交易所时发生错误: {e}")
        return None

def get_data(symbol: str, timeframe: str, since: Optional[str] = None, limit: int = 72) -> pd.DataFrame:
    """
    获取并缓存历史OHLCV数据。
    
    这是主要的获取数据函数。
    - 如果'since'未提供，它将只获取最新的'limit'条数据（默认为72）。
    - 如果'since'提供了，它将获取从该时间点开始的所有可用数据，并更新缓存。
    
    参数：
        symbol (str): 交易符号 (例如: 'BTC/USDT')。
        timeframe (str): 时间框架 (例如: '1h')。
        since (str, optional): 开始日期字符串，格式为ISO 8601。 
                               如果未提供，将只获取最新数据。
        limit (int, optional): 要获取的K线条数。默认为 72。

    返回：
        pd.DataFrame: 包含OHLCV数据的DataFrame，按时间戳索引。
                      失败时返回空DataFrame。
    """
    # 统一在这里定义cache_path
    cache_path = os.path.join(PATHS['cache'], f"v2_ohlcv_{symbol.replace('/', '')}_{timeframe}.pkl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    exchange = None
    try:
        exchange = get_exchange()
    except ConnectionError as e:
        LOGGER.warning(f"无法初始化交易所: {e}. 将尝试仅从本地缓存加载数据。")

    # 如果交易所初始化失败，直接尝试从缓存加载
    if exchange is None:
        if os.path.exists(cache_path):
            try:
                df = pd.read_pickle(cache_path)
                LOGGER.success(f"成功从缓存加载了 {len(df)} 条数据。回测将仅使用缓存数据。")
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                return df.copy()
            except Exception as read_e:
                LOGGER.error(f"无法读取缓存文件 {cache_path}: {read_e}。且网络连接失败，无法继续。")
                return pd.DataFrame()
        else:
            LOGGER.error(f"网络连接失败，且找不到本地缓存文件 {cache_path}。无法获取数据。")
            return pd.DataFrame()
            
    # --- 如果交易所初始化成功，则执行正常的在线数据获取流程 ---
    
    # --- 新逻辑：根据 'since' 参数决定行为 ---
    if since is None:
        # --- 行为1: 'since' 未提供 - 只获取最新的 'limit' 条数据 ---
        LOGGER.info(f"正在获取最新的 {limit} 条数据...")
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                LOGGER.error("无法从交易所获取最新数据。")
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            LOGGER.success(f"成功获取了 {len(df)} 条最新数据。")
            return df

        except Exception as e:
            LOGGER.error(f"获取最新数据时发生错误: {e}", exc_info=True)
            return pd.DataFrame()

    # --- 行为2: 'since' 已提供 - 执行完整的历史数据获取和缓存更新 ---
    LOGGER.info(f"参数 'since' 已提供。将执行完整的历史数据获取与缓存更新流程...")
    
    # 使用提供的 'since' 或 fallback 到配置
    since_str = since
    start_timestamp = exchange.parse8601(since_str)
    
    df = pd.DataFrame()
    
    # --- 1. Load from Cache ---
    # --- 1. 从缓存加载 ---
    if os.path.exists(cache_path):
        try:
            df = pd.read_pickle(cache_path)
            LOGGER.info(f"Loaded {len(df)} records from cache.")
            # 如果缓存不为空，从最后一个已知记录开始获取
            if not df.empty:
                last_index = df.index[-1]
                # 只处理不是Index等类型的情况
                if hasattr(last_index, 'value') and last_index is not None:
                    start_timestamp = int(getattr(last_index, 'value')) // 10**6
                elif last_index is not None and not isinstance(last_index, (pd.Index, pd.MultiIndex, pd.RangeIndex, tuple)):
                    try:
                        ts = pd.Timestamp(last_index) if last_index is not None else pd.Timestamp(int(time.time() * 1000))
                        start_timestamp = int(ts.value) // 10**6
                    except Exception:
                        start_timestamp = int(last_index) if last_index is not None else int(time.time() * 1000)
                else:
                    start_timestamp = int(time.time() * 1000)
        except Exception as e:
            LOGGER.warning(f"Cache file '{cache_path}' might be corrupted ({e}). Re-fetching all data.")
            df = pd.DataFrame()

    # --- 2. Fetch New Data ---
    # --- 2. 获取新数据 ---
    all_ohlcv = []
    LOGGER.info(f"Fetching all available data since {exchange.iso8601(start_timestamp)}...")
    
    # 我们在start_timestamp上加1毫秒以避免重复获取最后一条记录
    if start_timestamp is None:
        start_timestamp = int(time.time() * 1000)
    current_timestamp = int(start_timestamp) + 1 

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_timestamp, limit=1000)
            if not ohlcv:
                LOGGER.info("交易所没有更多新数据。")
                break
            
            # 如果交易所返回相同的最后一根K线，防止无限循环
            if all_ohlcv and ohlcv[-1][0] == all_ohlcv[-1][0]:
                LOGGER.warning("收到重复的数据块，停止获取。")
                break
            
            all_ohlcv.extend(ohlcv)
            # 下一次请求的'since'是我们收到的最后一根K线的时间戳
            current_timestamp = ohlcv[-1][0]
            
            LOGGER.info(f"本次获取了 {len(ohlcv)} 条新数据，累计已获取 {len(all_ohlcv)} 条。")
            time.sleep(exchange.rateLimit / 1000) # 遵守速率限制
        
        except Exception as e:
            LOGGER.error(f"获取数据时发生错误: {e}，5秒后重试...", exc_info=True)
            time.sleep(5)
            continue

    # --- 3. Combine, Save, and Return Data ---
    # --- 3. 合并、保存并返回数据 ---
    if all_ohlcv:
        new_df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
        new_df.set_index('timestamp', inplace=True)
        
        # 合并新旧数据，删除重复项并排序
        if not df.empty:
            df = pd.concat([df, new_df])
        else:
            df = new_df
        
        # 对于重复的索引保留最后一条记录
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        
        # 将合并后的数据保存回缓存
        df.to_pickle(cache_path)
        LOGGER.success(f"数据获取完成，缓存总记录数: {len(df)}，已保存到 {cache_path}")

    # 保证df始终为DataFrame
    if not isinstance(df, pd.DataFrame):
        LOGGER.error("get_data返回类型异常，已强制返回空DataFrame。")
        return pd.DataFrame()
    if df.empty:
        LOGGER.error("No data could be fetched or loaded.")
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df.copy()

def get_prepared_data_for_training(model_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    获取并准备用于指定模型训练的最终数据 (X, y)。
    """
    LOGGER.info(f"正在为模型 '{model_name}' 准备完整的训练数据...")
    
    # 1. 获取原始价格数据
    price_data = get_data(symbol=DATA_CONFIG['symbol'], timeframe=DATA_CONFIG['timeframe'])
    if price_data.empty:
        LOGGER.error("无法加载价格数据，数据准备中止。")
        return pd.DataFrame(), pd.Series(dtype=float)

    # 2. 创建基础技术特征
    # 直接传递model_name，让create_features内部处理配置
    feature_df = create_features(price_data, model_name)

    # 3. 根据模型类型生成特定的目标变量并选择最终特征
    X, y = prepare_training_data(feature_df, model_name)
    
    # 保证X始终为DataFrame，y始终为Series
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    return X, y

def save_data(df: pd.DataFrame, file_path: str):
    """
    将DataFrame保存到指定路径。
    """
    if df is None:
        df = pd.DataFrame()
    if file_path is None:
        LOGGER.error("file_path不能为None")
        return
    try:
        df.to_pickle(file_path)
        LOGGER.success(f"数据已成功保存到 {file_path}")
    except Exception as e:
        LOGGER.error(f"保存数据到 {file_path} 失败: {e}") 