# btc_predictor/backtest.py

import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from loguru import logger as LOGGER
from tqdm.auto import tqdm # 引入tqdm

# --- Start: Path Fix ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End: Path Fix ---

from btc_predictor.data import get_data
from btc_predictor.predict import predict_for_event
from btc_predictor.config import BACKTEST_CONFIG, get_model_config
from btc_predictor.features import create_features
from btc_predictor.utils import load_model_artifacts # 引入load_model_artifacts

def run_event_driven_backtest(model_name: str, prediction_threshold: float, ma_window: int, ohlcv_df: pd.DataFrame, artifacts: dict, show_progress: bool = True) -> dict:
    """
    运行一个完整的事件驱动回测。
    此版本经过重构，以严格模拟逐根K线的时间流逝，防止前视偏差。
    
    Returns:
        dict: 包含回测结果的字典。
    """
    model = artifacts['model']
    scaler_X = artifacts['scaler_X']
    model_config = get_model_config(model_name)
    feature_names = model_config.get('features', [])

    initial_capital = BACKTEST_CONFIG['initial_capital']
    cash = initial_capital
    position = 0.0
    commission = BACKTEST_CONFIG['commission']
    trades = []
    
    in_position = False
    current_trade = {}
    
    # 预计算信号检测所需的特定MA，避免在循环中重新计算整个特征集
    ohlcv_df[f'ma{ma_window}'] = ohlcv_df['close'].rolling(window=ma_window).mean()
    # 删除滚动平均产生的初始NaN值
    ohlcv_df.dropna(inplace=True)
    
    # 为了让所有技术指标（尤其是长周期的）都能被正确计算，需要一个“热身”期。
    # 从features.py可知，最长的周期是72 (return_72h)，所以设置一个安全的最小历史数据长度。
    min_history_size = 100

    iterable = ohlcv_df.iterrows()
    if show_progress:
        progress_bar = tqdm(iterable, total=len(ohlcv_df), desc=f"模拟交易", leave=False)
        iterable = progress_bar

    # --- 主循环：逐根K线模拟 ---
    # 我们从ma_window开始，确保有足够的历史数据来计算初始MA
    for i in range(1, len(ohlcv_df)):
        # 使用 .iloc 来访问，确保是基于整数位置的切片
        current_data_point = ohlcv_df.iloc[i]
        prev_data_point = ohlcv_df.iloc[i-1]
        
        current_price = current_data_point['close']
        
        # 1. 探测交叉信号
        is_golden_cross = prev_data_point['close'] < prev_data_point[f'ma{ma_window}'] and \
                          current_data_point['close'] > current_data_point[f'ma{ma_window}']
        is_death_cross = prev_data_point['close'] > prev_data_point[f'ma{ma_window}'] and \
                         current_data_point['close'] < current_data_point[f'ma{ma_window}']
        
        signal = 0
        if is_golden_cross:
            signal = 1
        elif is_death_cross:
            signal = -1
        
        # 2. 检查是否应该退出当前持仓 (基于反向信号)
        if in_position:
            is_long = current_trade['direction'] == 'long'
            exit_signal = (is_long and signal == -1) or \
                          (not is_long and signal == 1)
            
            if exit_signal:
                pnl = (current_price - current_trade['entry_price']) * current_trade['size'] if is_long else (current_trade['entry_price'] - current_price) * current_trade['size']
                trade_value_at_exit = current_trade['size'] * current_price
                cash += trade_value_at_exit * (1 - commission)
                
                current_trade['exit_price'] = current_price
                current_trade['exit_date'] = current_data_point.name
                current_trade['pnl'] = pnl
                current_trade['exit_reason'] = 'reverse_signal'
                trades.append(current_trade)
                
                position = 0.0
                in_position = False
                current_trade = {}

        # 3. 如果没有持仓，并且探测到信号，则考虑开仓
        if not in_position and signal != 0:
            # 增加热身期检查，确保有足够数据计算特征
            if i + 1 < min_history_size:
                continue

            # 截取到当前时间点为止的所有历史数据
            history_to_date = ohlcv_df.iloc[:i+1]
            
            # 为这段历史动态生成特征
            # 这是关键修复：特征是在事件发生时，仅基于可用历史生成的
            features_for_event = create_features(history_to_date, model_name)
            
            # --- 新增的修复 ---
            # 如果历史数据不足以生成任何有效的特征行，则跳过
            if features_for_event.empty:
                continue

            # 获取最后一行的特征（即当前事件的特征）
            event_data = features_for_event.iloc[[-1]]

            # 调用预测函数
            prediction = predict_for_event(
                model=model,
                scaler_X=scaler_X,
                feature_names=feature_names,
                event_data=event_data
            )
            
            if prediction is None:
                continue

            take_trade = (signal == 1 and prediction > prediction_threshold) or \
                         (signal == -1 and prediction < -prediction_threshold)

            if take_trade:
                trade_size_in_btc = (cash * 0.95) / current_price
                entry_value = trade_size_in_btc * current_price
                cash -= entry_value * (1 + commission)
                position = trade_size_in_btc
                in_position = True
                
                current_trade = {
                    'entry_date': current_data_point.name,
                    'entry_price': current_price,
                    'size': trade_size_in_btc,
                    'direction': 'long' if signal == 1 else 'short',
                    'prediction': prediction,
                }

    if show_progress:
        progress_bar.close()

    # 4. 计算并返回统计数据
    if not trades:
        return {
            "Total Return [%]": 0, "Total Trades": 0, "Win Rate [%]": 0,
            "Profit Factor": 0, "Avg Win": 0, "Avg Loss": 0, "Threshold": prediction_threshold
        }

    trades_df = pd.DataFrame(trades)
    final_equity = cash + (position * ohlcv_df['close'].iloc[-1] if in_position else 0)
    total_return = (final_equity / initial_capital - 1) * 100
    
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    win_rate = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0
    
    # 修正 profit_factor 计算
    sum_of_losses = losses['pnl'].sum()
    profit_factor = abs(wins['pnl'].sum() / sum_of_losses) if sum_of_losses != 0 else np.inf

    return {
        "Total Return [%]": total_return,
        "Total Trades": len(trades_df),
        "Win Rate [%]": win_rate,
        "Profit Factor": profit_factor,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss,
        "Threshold": prediction_threshold
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行或优化事件驱动的交易策略回测。")
    parser.add_argument('--model-name', type=str, required=True, help='要回测的模型名称。')
    parser.add_argument('--threshold', type=float, default=0.01, help='进行交易的预测回报率阈值（单次运行时）。')
    parser.add_argument('--optimize', action='store_true', help='如果设置，则对阈值参数进行优化。')
    
    args = parser.parse_args()
    
    # --- 1. 一次性加载资源 ---
    LOGGER.info(f"正在加载模型 '{args.model_name}' 及其所有工件...")
    try:
        artifacts = load_model_artifacts(args.model_name)
    except Exception as e:
        LOGGER.error(f"加载模型工件失败: {e}。请先确保模型已训练。")
        sys.exit(1)
    
    LOGGER.info("正在加载价格数据...")
    ohlcv_df = get_data(BACKTEST_CONFIG['symbol'], BACKTEST_CONFIG['timeframe'])
    if ohlcv_df is None or ohlcv_df.empty:
        LOGGER.error("无法加载数据，回测中止。")
        sys.exit(1)

    model_config = get_model_config(args.model_name)
    ma_window = model_config.get('ma_window', 60)
    
    if args.optimize:
        # --- 优化模式 ---
        LOGGER.info("🚀 开始阈值优化...")
        threshold_range = np.arange(0.005, 0.105, 0.005)
        results = []
        
        # 主进度条
        opt_progress = tqdm(threshold_range, desc="[优化进度]", leave=True)

        for threshold in opt_progress:
            # 在这里，我们为每次回测传递已经加载的数据和模型
            result = run_event_driven_backtest(
                model_name=args.model_name,
                prediction_threshold=threshold,
                ma_window=ma_window,
                ohlcv_df=ohlcv_df,
                artifacts=artifacts,
                show_progress=False # 在优化时不显示内部循环的进度条
            )
            results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df.set_index('Threshold', inplace=True)
        
        # 找到最佳结果
        best_threshold = results_df['Total Return [%]'].idxmax()
        best_result = results_df.loc[best_threshold]

        print("\n--- 阈值优化结果 ---")
        print(results_df.round(2).to_string())
        print("\n--- 最佳结果 (基于总回报率) ---")
        print(f"最佳阈值: {best_threshold:.4f}")
        print(best_result.round(2).to_string())

    else:
        # --- 单次运行模式 ---
        LOGGER.info(f"🚀 开始单次回测 (阈值 = {args.threshold})...")
        final_stats = run_event_driven_backtest(
            model_name=args.model_name,
            prediction_threshold=args.threshold,
            ma_window=ma_window,
            ohlcv_df=ohlcv_df,
            artifacts=artifacts,
            show_progress=True
        )
        
        print("\n--- 回测结果 ---")
        for key, value in final_stats.items():
            print(f"{key}: {value:.2f}" if isinstance(value, (int, float)) else f"{key}: {value}")