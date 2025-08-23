# btc_predictor/optimize.py

import torch
import numpy as np
import pandas as pd
import json
import os
import optuna
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, mean_squared_error
import torch.optim as optim

from .config import get_model_config, PATHS
from .data import get_prepared_data_for_training
from .model import create_model
from .utils import LOGGER, setup_logger

# --- 全局常量 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRIALS = 50 # Optuna 优化的试验次数

def objective(trial: optuna.Trial, model_name: str, X_train, y_train, X_val, y_val) -> float:
    """
    Optuna 的目标函数，用于寻找最优超参数。
    """
    model_config = get_model_config(model_name)
    task = model_config.get('task', 'regression')

    # --- 1. 定义超参数搜索空间（扩展版） ---
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True), # 学习率范围更广
        'nhead': trial.suggest_categorical('nhead', [2, 4, 8, 16]), # 增加更多头
        'num_encoder_layers': trial.suggest_int('num_encoder_layers', 1, 6), # 更深的模型
        'dim_feedforward': trial.suggest_int('dim_feedforward', 128, 1024, step=128), # 更宽的网络
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]), # 更多批次大小选项
    }

    # --- 2. 数据准备和加载 ---
    X_train_scaled = torch.tensor(X_train, dtype=torch.float32)
    y_train_scaled = torch.tensor(y_train, dtype=torch.float32)
    X_val_scaled = torch.tensor(X_val, dtype=torch.float32)
    y_val_scaled = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_scaled, y_train_scaled)
    val_dataset = TensorDataset(X_val_scaled, y_val_scaled)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # --- 3. 模型和损失函数 ---
    model_params = {
        'input_dim': X_train.shape[1],
        'task': task,
        'model_type': model_config.get('model_type', 'transformer'),
        **params
    }
    model = create_model(**model_params).to(DEVICE)
    
    if task == 'regression':
        criterion = nn.MSELoss()
        eval_metric = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)) # RMSE
    else: # classification
        criterion = nn.BCELoss()
        eval_metric = f1_score
        
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # --- 4. 训练和验证循环 ---
    for epoch in range(25): # 在优化中增加周期以获得更可靠的结果
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # --- 5. 评估 ---
    model.eval()
    y_preds = []
    y_trues = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_X)
            y_preds.extend(outputs.cpu().numpy())
            y_trues.extend(batch_y.cpu().numpy())

    if task == 'regression':
        # 明确将numpy.float64转换为python float
        return float(eval_metric(y_trues, y_preds))
    else: # classification
        y_preds_binary = (np.array(y_preds) > 0.5).astype(int)
        # 明确将numpy.float64转换为python float
        return float(eval_metric(y_trues, y_preds_binary))


def run_optimization(model_name: str):
    """
    为指定模型运行超参数优化。
    """
    setup_logger()
    LOGGER.info(f"🚀 开始为模型 '{model_name}' 进行超参数优化...")

    # 1. 加载数据
    X, y = get_prepared_data_for_training(model_name)
    if X.empty:
        LOGGER.error("数据为空，优化中止。")
        return

    # 2. 数据拆分和缩放（与 train.py 保持一致）
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, val_index in tscv.split(X):
        pass # 获取最后一个拆分
    
    X_train, X_val = X.iloc[train_index, :], X.iloc[val_index,:]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    task = get_model_config(model_name).get('task', 'regression')
    if task == 'regression':
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))
    else:
        y_train_scaled, y_val_scaled = y_train.values, y_val.values

    # 3. 运行 Optuna 研究
    study_direction = 'minimize' if task == 'regression' else 'maximize'
    study = optuna.create_study(direction=study_direction)
    
    study.optimize(
        lambda trial: objective(trial, model_name, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled),
        n_trials=None, # 移除固定试验次数
        timeout=36000, # 设置10小时运行时间
        show_progress_bar=True
    )

    # 4. 保存最优参数
    best_params = study.best_params
    params_path = os.path.join(PATHS['results'], f"best_params_{model_name}.json")
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
        
    LOGGER.success(f"优化完成！最佳参数已保存至 {params_path}")
    LOGGER.info(f"最佳试验分数（{'RMSE' if task == 'regression' else 'F1分数'}）: {study.best_value}")
    LOGGER.info(f"最佳参数: {best_params}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行模型的超参数优化。")
    parser.add_argument("--model-name", required=True, type=str, help="要优化的模型名称。")
    args = parser.parse_args()
    run_optimization(args.model_name)