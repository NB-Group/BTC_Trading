
import pandas as pd
import numpy as np
import torch
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from .utils import LOGGER, setup_logger, load_model_artifacts, DEVICE
from .config import get_model_config
from .features import create_features

def predict_for_event(model: torch.nn.Module, scaler_X, feature_names: list, event_data: pd.DataFrame) -> Optional[float]:
    """
    为单个交叉事件实时生成预测。
    这个版本接收已加载的模型和scaler，以提高性能。
    
    Args:
        model (torch.nn.Module): 预先加载的PyTorch模型。
        scaler_X: 预先加载的特征缩放器。
        feature_names (list): 模型使用的特征名称列表。
        event_data (pd.DataFrame): 包含单行事件数据的DataFrame。

    Returns:
        Optional[float]: 对未来最大回报率的单个浮点数预测，如果出错则返回None。
    """
    try:
        # --- FIX: Ensure we only use features the scaler was fitted on ---
        # Get the feature names the scaler expects
        scaler_feature_names = scaler_X.feature_names_in_
        
        # Filter the event_data to only include these features
        if hasattr(event_data, 'columns') and hasattr(scaler_feature_names, 'tolist'):
            input_features = event_data[scaler_feature_names.tolist()]
        else:
            input_features = event_data[scaler_feature_names]
        
        # Now, the columns in input_features perfectly match what the scaler expects.
        input_scaled = scaler_X.transform(input_features)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(DEVICE)
        
        model.eval()
        with torch.no_grad():
            prediction = model(input_tensor)
            
        return prediction.item()

    except Exception as e:
        LOGGER.error(f"为事件生成预测时出错: {e}")
        return None

def get_all_predictions(model_name: str, price_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    为整个数据集批量生成预测。主要用于分析和可视化。
    """
    setup_logger()
    LOGGER.info(f"开始为模型 '{model_name}' 批量生成所有预测...")

    try:
        feature_df = create_features(price_data, model_name)
        
        artifacts = load_model_artifacts(model_name)
        model = artifacts['model']
        scaler_X = artifacts['scaler_X']
        scaler_y = artifacts.get('scaler_y') # 可能为None
        model_config = get_model_config(model_name)
        feature_names = model_config.get('features', [])

        X = feature_df[feature_names]
        if hasattr(X, 'columns') and hasattr(X.columns, 'tolist'):
            X = X[X.columns.tolist()]
        X_scaled = scaler_X.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

        model.eval()
        with torch.no_grad():
            predictions_scaled = model(X_tensor).cpu().numpy()

        # 如果是回归任务且有y缩放器，则进行逆转换
        if model_config.get('task') == 'regression' and scaler_y:
            predictions = scaler_y.inverse_transform(predictions_scaled)
        else:
            predictions = predictions_scaled
        
        prediction_col_name = 'prediction_probability' if model_config.get('task') == 'classification' else 'prediction'
        # 保证columns为Index类型
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        results_df = pd.DataFrame(predictions, index=feature_df.index, columns=np.array([prediction_col_name]))
        
        LOGGER.success(f"为 {len(results_df)} 个时间点成功生成批量预测。")
        return results_df

    except Exception as e:
        LOGGER.error(f"批量生成预测时发生严重错误: {e}")
        return None

def get_live_trade_signal(model_name: str) -> Optional[Dict[str, Any]]:
    """
    获取最新的实时交易信号。
    这是主循环调用的核心函数，用于实盘交易决策。
    """
    from .data import get_data # 局部导入
    from .config import DATA_CONFIG # 导入数据配置
    
    LOGGER.info(f"正在为模型 '{model_name}' 获取实时交易信号...")
    
    try:
        # 1. 加载模型
        artifacts = load_model_artifacts(model_name)
        model = artifacts['model']
        scaler_X = artifacts['scaler_X']
        model_config = get_model_config(model_name)
        feature_names = model_config.get('features', [])
        ma_window = model_config.get('ma_window', 60)

        # 2. 获取最新数据 (获取稍多一些数据以计算指标)
        price_data = get_data(
            symbol=DATA_CONFIG['symbol'], 
            timeframe=DATA_CONFIG['timeframe'], 
            limit=ma_window + 150 # 增加获取量以确保有足够数据
        )
        if price_data is None or len(price_data) < ma_window:
            LOGGER.warning("获取的数据不足以计算指标，无法生成信号。")
            return None

        # 3. 计算特征和信号
        features_df = create_features(price_data.copy(), model_name)
        features_df[f'ma{ma_window}'] = features_df['close'].rolling(window=ma_window).mean()
        
        # 获取最新的两个数据点以探测交叉
        latest = features_df.iloc[-1]
        previous = features_df.iloc[-2]

        # 4. 探测交叉信号
        signal = "HOLD"
        is_golden_cross = previous['close'] < previous[f'ma{ma_window}'] and latest['close'] > latest[f'ma{ma_window}']
        is_death_cross = previous['close'] > previous[f'ma{ma_window}'] and latest['close'] < latest[f'ma{ma_window}']

        if is_golden_cross:
            signal = "BUY"
        elif is_death_cross:
            signal = "SELL"
            
        # 5. 如果有信号，则获取模型预测
        prediction = 0.0
        if signal != "HOLD":
            event_data = features_df.iloc[[-1]] # 获取最后一行的DataFrame
            prediction = predict_for_event(
                model=model,
                scaler_X=scaler_X,
                feature_names=feature_names,
                event_data=event_data
            )
            if prediction is None:
                LOGGER.error("模型预测失败，信号被忽略。")
                signal = "HOLD" # 预测失败则不交易
                prediction = 0.0

        result = {
            "signal": signal,
            "predicted_return": prediction,
            "timestamp": latest.name.isoformat(),
            "current_price": latest['close'],
            "info": "Signal processed successfully." # 明确的成功信息
        }
        LOGGER.info(f"实时信号获取成功: {result}")
        return result

    except Exception as e:
        LOGGER.error(f"获取实时交易信号时发生严重错误: {e}")
        return {
            "signal": "HOLD",
            "predicted_return": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "info": f"获取信号时出错: {e}"
        } 