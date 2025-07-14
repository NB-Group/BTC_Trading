# btc_predictor/utils.py

import logging
import random
import numpy as np
import torch
import os
import sys
import json
from loguru import logger
from pathlib import Path
import joblib
from typing import Dict, Any
from torch.utils.data import Dataset

# 从btc_predictor自己的配置导入
from .config import LOG_FILE, PATHS, DEVICE, MODEL_CONFIG, FEATURE_CONFIG, get_model_config, LOG_LANG
from .model import create_model

# --- 日志记录 ---

def setup_logger():
    """
    设置loguru日志记录器，支持中英文切换。
    """
    logger.remove()
    if LOG_LANG == 'zh':
        fmt_console = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        fmt_file = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    else:
        fmt_console = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        fmt_file = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    logger.add(
        sys.stderr,
        level="INFO",
        format=fmt_console
    )
    logger.add(
        LOG_FILE,
        level="DEBUG",
        rotation="10 MB",
        retention="10 days",
        format=fmt_file
    )
    return logger

# Globally accessible logger
LOGGER = setup_logger() 

# --- 新增：模型加载工具 ---
def load_model_artifacts(model_name: str, verbose: bool = True) -> Dict[str, Any]:
    """
    加载指定模型的所有相关工件：模型权重、缩放器和最优参数。

    参数：
        model_name (str): 模型名称，应与config.py中定义的键匹配。
        verbose (bool): 是否打印加载日志。

    返回：
        dict: 包含所有加载工件的字典。
              例如: {'model': model_object, 'scaler_X': scaler_X_object, ...}
    
    异常：
        FileNotFoundError: 如果任何必需的文件（模型、参数、缩放器）不存在。
        Exception: 捕获其他可能的加载错误。
    """
    if verbose:
        LOGGER.info(f"正在加载模型 '{model_name}' 的所有工件...")
    
    try:
        model_config = get_model_config(model_name)
        task = model_config.get('task', 'regression')
        
        model_path = os.path.join(PATHS['models'], f"final_model_{model_name}.pth")
        scaler_x_path = os.path.join(PATHS['models'], f"{model_config['scaler_name']}_X.joblib")
        params_path = os.path.join(PATHS['results'], f"best_params_{model_name}.json")

        # 检查核心文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        if not os.path.exists(scaler_x_path):
            raise FileNotFoundError(f"X缩放器文件未找到: {scaler_x_path}")
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"参数文件未找到: {params_path}")

        # 加载模型
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        model_params = {
            'input_dim': 0, # 临时值, 稍后需要确定
            'task': task,
            'model_type': model_config.get('model_type', 'transformer'),
            **params
        }
        # 动态确定input_dim
        temp_scaler_x = joblib.load(scaler_x_path)
        model_params['input_dim'] = temp_scaler_x.n_features_in_
        
        model = create_model(**model_params).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        # 加载X缩放器
        scaler_X = joblib.load(scaler_x_path)
        
        # 条件加载Y缩放器
        scaler_y = None
        if task == 'regression':
            scaler_y_path = os.path.join(PATHS['models'], f"{model_config['scaler_name']}_y.joblib")
            if not os.path.exists(scaler_y_path):
                raise FileNotFoundError(f"Y缩放器文件未找到: {scaler_y_path}")
            scaler_y = joblib.load(scaler_y_path)
            
        artifacts = {
            "model": model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y, # 可能为None
            "params": params
        }
        if verbose:
            LOGGER.success(f"模型 '{model_name}' 的所有工件加载成功。")
        return artifacts

    except Exception as e:
        if verbose:
            LOGGER.error(f"加载模型工件 '{model_name}' 时发生严重错误: {e}")
        raise # 重新抛出异常，让调用者处理


def seed_everything(seed: int):
    """设置随机种子以保证实验可复现性。"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_project_root():
    """获取项目根目录的绝对路径"""
    return Path(__file__).parent.parent.resolve()

# --- 新增：通用的数据处理工具 ---

def create_sequences(X_data, y_data, seq_length):
    """
    将时间序列数据转换为适用于LSTM/Transformer的序列格式。
    """
    X_seq, y_seq = [], []
    for i in range(len(X_data) - seq_length):
        X_seq.append(X_data[i:i+seq_length])
        y_seq.append(y_data[i+seq_length])
    return np.array(X_seq), np.array(y_seq) 

# --- 新增：通用数据集类 ---
class RegressionDataset(Dataset):
    """用于回归任务的自定义PyTorch数据集。"""
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx] 