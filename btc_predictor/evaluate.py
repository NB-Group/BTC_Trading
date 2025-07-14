# btc_predictor/evaluate.py

import os
import torch
import joblib
import numpy as np
import pandas as pd
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from .config import PATHS, DEVICE, get_model_config
from .data import get_prepared_data_for_training
from .utils import LOGGER, setup_logger, load_model_artifacts, RegressionDataset
from torch.utils.data import DataLoader

def run_evaluation(model_name: str):
    """
    加载已训练的事件驱动模型，在测试集上进行评估，并生成报告和图表。
    """
    setup_logger()
    LOGGER.info(f"🚀 开始评估模型: {model_name}")

    # --- 1. 加载模型和工件 ---
    try:
        artifacts = load_model_artifacts(model_name)
        model = artifacts['model']
        scaler_X = artifacts['scaler_X']
        scaler_y = artifacts.get('scaler_y') # 如果是回归任务，则应该存在
        params = artifacts['params']
    except Exception as e:
        LOGGER.error(f"加载模型工件失败: {e}。请先运行 train.py。")
        return

    if not scaler_y:
        LOGGER.error("模型是回归任务，但未找到 y-scaler (scaler_y)。无法继续评估。")
        return
        
    LOGGER.info("模型和工件加载成功。")

    # --- 2. 准备评估数据 (测试集) ---
    LOGGER.info("正在准备测试数据集...")
    X, y = get_prepared_data_for_training(model_name)
    if X.empty:
        LOGGER.error("数据准备失败，无法进行评估。")
        return

    # 使用与训练时相同的方式分割数据以获取测试集
    n_splits = get_model_config(model_name).get('n_splits', 5)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # 获取最后一个 (train_index, test_index) 分割
    all_splits = list(tscv.split(X))
    train_index, test_index = all_splits[-1]
    
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]
    
    # 对测试集进行缩放
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    # 创建DataLoader
    test_dataset = RegressionDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32), 
        torch.tensor(y_test_scaled, dtype=torch.float32).squeeze()
    )
    test_loader = DataLoader(test_dataset, batch_size=params.get('batch_size', 32), shuffle=False)
    
    LOGGER.info(f"已准备 {len(test_dataset)} 个样本用于最终评估。")

    # --- 3. 执行预测 ---
    model.eval()
    all_preds_scaled = []
    all_true_scaled = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            all_preds_scaled.extend(outputs.cpu().numpy())
            all_true_scaled.extend(y_batch.cpu().numpy().reshape(-1, 1))

    # --- 4. 逆缩放以获得真实值 ---
    predictions = scaler_y.inverse_transform(np.array(all_preds_scaled)).flatten()
    true_values = scaler_y.inverse_transform(np.array(all_true_scaled)).flatten()
    
    # --- 5. 计算回归指标 ---
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'test_set_size': len(true_values)
    }
    LOGGER.info(f"评估指标:\n{json.dumps(metrics, indent=4)}")

    # --- 6. 保存报告 ---
    report_path = os.path.join(PATHS['results'], f"final_evaluation_report_{model_name}.json")
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    LOGGER.success(f"评估报告已保存至 {report_path}")

    # --- 7. 可视化：预测值 vs 真实值 ---
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
    except Exception as e:
        LOGGER.warning(f"设置中文字体失败，图表可能显示乱码: {e}")

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x=true_values, y=predictions, ax=ax, alpha=0.6, edgecolor='w')
    
    lims = [
        min(min(true_values), min(predictions)),
        max(max(true_values), max(predictions))
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='理想情况 (y=x)')
    
    ax.set_xlabel("真实值 (True Values)", fontsize=12)
    ax.set_ylabel("预测值 (Predicted Values)", fontsize=12)
    ax.set_title(f"模型评估: 预测值 vs. 真实值 ({model_name}) | R² = {r2:.3f}", fontsize=16)
    ax.legend()
    ax.grid(True)
    
    plot_path = os.path.join(PATHS['results'], f"prediction_vs_true_{model_name}.png")
    plt.savefig(plot_path)
    LOGGER.success(f"评估图表已保存至 {plot_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估事件驱动回归模型。")
    parser.add_argument(
        '--model-name', 
        type=str, 
        default='btc-crossover-regression-v1', 
        help='要评估的模型名称。'
    )
    args = parser.parse_args()
    run_evaluation(model_name=args.model_name) 