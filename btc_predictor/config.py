import os
import torch
import logging
from typing import Dict, Any
from .model import BTCPriceTransformer

# ----------------- Data Config -----------------
DATA_CONFIG = {
    'symbol': 'BTC/USDT',
    'timeframe': '1h',
    'exchange': 'binance',
    'since': '2020-01-01T00:00:00Z', # Fetch data from the beginning of 2020
    'limit': 100000, # Set a high limit to ensure all data since 'since' is fetched
    'cache_path': 'cache/v2_ohlcv_{symbol}_{timeframe}.pkl'
}


# ----------------- Feature Engineering Config -----------------
FEATURE_CONFIG = {
    # Indicator periods
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'rsi_period': 14,
    'atr_period': 14,
    'bbands_period': 20,
    'bbands_std_dev': 2,
    'stoch_k': 14,
    'stoch_d': 3,
    'ma_short_period': 7,
    'ma_long_period': 25,
    'ma60_period': 60, # 新增 MA60 周期

    # 要在模型中使用的特征列
    # 注意: 'ma60' 本身通常不作为特征，而是用于生成信号，但也可以包含进来测试效果
    'feature_names': [
        'ma_short', 'ma_long', 'ma60', 'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_width', 'atr', 'stoch', 'stoch_signal', 'volume', 'high', 'low', 'close'
    ]
}

# ----------------- Model Configuration -----------------
MODEL_CONFIG = {
    'model_type': 'transformer', # 'lstm' 或 'transformer'
    'seq_length': 60,            # 输入序列长度
    'output_dim': 1,             # 回归任务输出维度为1
    'dropout': 0.2,              # 通用 Dropout 概率

    # LSTM 模型特定参数
    'lstm_hidden_dim': 128,          
    'lstm_n_layers': 3,              

    # Transformer 模型特定参数
    'transformer_d_model': 64,   # Transformer 内部工作维度
    'transformer_ff_dim': 256,   # 前馈网络维度
    'transformer_n_head': 4,     # 注意力头数量
    'transformer_n_layers': 2,   # Transformer Encoder 层数
}


# ----------------- Optuna Configuration -----------------
OPTUNA_CONFIG = {
    "n_trials": 50,
    "timeout": None,  # In seconds
    "storage_name": "optuna_studies.db",
    "study_name": "btc_prediction_study"
}

# ----------------- Training Configuration -----------------
TRAINING_CONFIG = {
    'batch_size': 256,
    'epochs': 50,
    'patience': 10, # 稍微增加耐心
    'learning_rate': 1e-4, # 降低学习率以适应更复杂的任务
    'num_workers': 0,
    'data_limit': 50000, # 增加数据限制以使用更多历史数据
}

# 回测配置
BACKTEST_CONFIG = {
    'symbol': 'BTC/USDT',
    'timeframe': '1h',
    'initial_capital': 100000,
    'commission': 0.0005 # 币安手续费 (0.05%)
}


# ----------------- File Paths -----------------
PATHS = {
    'logs': 'logs',
    'results': 'results',
    'cache': 'cache',
    'models': 'models'
}
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# --- Logger Configuration ---
LOG_FILE = os.path.join(PATHS['logs'], 'btc_predictor_run.log')

# ----------------- GPU Config -----------------
GPU_CONFIG = {
    'benchmark': True, # Set to True if input sizes don't vary, for performance
    'num_workers': 4,  # Adjust based on your machine's core count
    'pin_memory': True, # Helps speed up data transfer to GPU
    'non_blocking': True # Use with pin_memory for asynchronous GPU copies
}


# ----------------- System Config -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Deprecated Proxy Setting -----------------
# The PROXY setting is deprecated. Use DEFAULTS['proxy_url'] in the root config instead.
PROXY = None 

def get_all_models():
    """返回所有模型配置的字典。"""
    return {
        'btc-transformer-v1': {
            'task': 'regression',
            'model_type': 'transformer',
            'target_col': 'predicted_return',
            'scaler_name': 'final_model_btc-transformer-v1',
            'model_class': BTCPriceTransformer,
            'best_params': {
                "seq_length": 60,
                "d_model": 128,
                "n_head": 8,
                "n_layers": 4,
                "dim_feedforward": 256,
                "dropout": 0.3,
                "learning_rate": 0.0001,
                "batch_size": 128
            }
        },
        'btc-classification-v1': {
            'task': 'classification', # 明确指定任务类型
            'model_type': 'transformer',
            'target_col': 'target', # 目标列现在是 'target' (0或1)
            'scaler_name': 'final_model_btc-classification-v1',
            'model_class': BTCPriceTransformer, 
            'best_params': { 
                "seq_length": 60,
                "d_model": 128,
                "n_head": 8,
                "n_layers": 4,
                "dim_feedforward": 256,
                "dropout": 0.3,
                "learning_rate": 0.0001,
                "batch_size": 128
            }
        },
        'btc-crossover-regression-v1': {
            'task': 'regression',
            'model_type': 'transformer',
            'description': "事件驱动模型，在MA60与K线交叉时触发，预测至下一次反向交叉期间的最大涨/跌幅。",
            'scaler_name': 'final_model_btc-crossover-regression-v1',
            # 更新后的特征列表 (修正大小写)
            'features': [
                # 核心价格/MA 特征
                'close', 'ma60', 
                # 技术指标
                'rsi', 'macd', 'macd_signal', 'macd_hist', 'atr',
                # 新增的历史回报率 (动量) 特征
                'return_1h', 'return_3h', 'return_6h', 'return_12h',
                'return_24h', 'return_48h', 'return_72h',
                # --- 新增的高级特征 ---
                'atr_ratio', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos'
            ],
            # 特征工程中使用的参数 (这部分当前代码并未使用，但可保留作为参考)
            'feature_engineering': {
                'lags': [1, 2, 3, 6, 12, 24],
                'rolling_windows': [6, 12, 24, 48, 72]
            },
        }
    }

# --- 模型配置 ---
# 全局模型注册表
# MODELS = { ... } # <--- 这部分被移入 get_all_models()

def get_model_config(model_name: str) -> dict:
    """
    从模型注册表中获取指定模型的配置。
    """
    all_models = get_all_models()
    if model_name not in all_models:
        raise ValueError(f"模型 '{model_name}' 未在 config.py 中定义。")
    return all_models[model_name] 