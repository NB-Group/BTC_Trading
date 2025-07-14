# btc_predictor/train.py

import os
import torch
import torch.nn as nn
import numpy as np
import joblib
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import torch.optim as optim

from .config import get_model_config, PATHS
from .model import create_model
from .utils import LOGGER, setup_logger
from .data import get_prepared_data_for_training
import json

# --- 全局常量 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_training(model_name: str):
    """
    为指定的模型名称运行完整的训练、验证和保存流程。
    """
    setup_logger()
    LOGGER.info(f"🚀 开始为模型 '{model_name}' 进行训练...")
    
    # --- 1. 加载和准备数据 ---
    try:
        model_config = get_model_config(model_name)
        task = model_config.get('task', 'regression')
        
        X, y = get_prepared_data_for_training(model_name)
        
        if X.empty or y.empty:
            LOGGER.error("没有可供训练的数据。")
            return

    except Exception as e:
        LOGGER.error(f"数据准备失败: {e}")
        return

    # --- 2. 数据拆分和缩放 ---
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 获取最后一个拆分作为训练集和验证集
    for train_index, val_index in tscv.split(X):
        pass
    
    X_train, X_val = X.iloc[train_index, :], X.iloc[val_index,:]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    scaler_y = None
    if task == 'regression':
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))
    else: # 分类任务
        y_train_scaled = y_train.values.reshape(-1, 1)
        y_val_scaled = y_val.values.reshape(-1, 1)

    # --- 3. 创建数据加载器 ---
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val_scaled, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # --- 4. 初始化模型、损失函数和优化器 ---
    try:
        params_path = os.path.join(PATHS['results'], f"best_params_{model_name}.json")
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                hyperparams = json.load(f)
            LOGGER.info(f"从 {params_path} 加载了优化超参数。")
        else:
            hyperparams = {} # 使用默认参数
            LOGGER.warning("未找到超参数文件，将使用默认模型参数。")

        model_params = {
            'input_dim': X_train.shape[1],
            'task': task,
            'model_type': model_config.get('model_type', 'transformer'),
            **hyperparams
        }
        model = create_model(**model_params).to(DEVICE)

        if task == 'regression':
            criterion = nn.MSELoss()
        else: # 分类任务
            criterion = nn.BCELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=hyperparams.get('learning_rate', 0.001))

    except Exception as e:
        LOGGER.error(f"模型初始化失败: {e}")
        return

    # --- 5. 训练循环 ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10

    for epoch in range(50): # 假设最多训练50个周期
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # 验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                val_loss = criterion(outputs, batch_y)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        LOGGER.info(f"Epoch [{epoch+1}/50], 训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(PATHS['models'], f"final_model_{model_name}.pth"))
            joblib.dump(scaler_X, os.path.join(PATHS['models'], f"{model_config['scaler_name']}_X.joblib"))
            if scaler_y:
                joblib.dump(scaler_y, os.path.join(PATHS['models'], f"{model_config['scaler_name']}_y.joblib"))
            LOGGER.info("验证损失改善，模型已保存。")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            LOGGER.info(f"验证损失连续 {patience} 个周期未改善，触发早停。")
            break
            
    LOGGER.success(f"训练完成。最佳验证损失: {best_val_loss:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行BTC预测模型的训练流程。")
    parser.add_argument("--model-name", type=str, required=True, help="要训练的模型的名称（在config.py中定义）。")
    args = parser.parse_args()
    
    run_training(model_name=args.model_name) 