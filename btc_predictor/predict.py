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
    该版本接收已加载的模型和scaler，以提升性能。
    
    参数：
        model (torch.nn.Module): 已加载的PyTorch模型。
        scaler_X: 已加载的特征缩放器。
        feature_names (list): 模型使用的特征名列表。
        event_data (pd.DataFrame): 包含单行事件数据的DataFrame。

    返回：
        Optional[float]: 对未来最大回报率的单个浮点数预测，出错时返回None。
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
            "info": "信号处理成功。" # 明确的成功信息
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

def run_rf4_backtest(ohlcv_df: pd.DataFrame, period: int = 14, order: int = 5, 
                     position_size: float = 0.95, stop_loss_pct: float = 0.05, 
                     dynamic_stop_loss: bool = False, atr_multiplier: float = 2.0, 
                     quiet: bool = False) -> Dict[str, Any]:
    """
    基于predict.py的RF4回测引擎 - 使用信号生成函数进行回测
    
    Args:
        ohlcv_df: OHLCV数据
        period: RSI周期
        order: 背离检测范围
        position_size: 仓位大小 (0.95 = 95% 满仓)
        stop_loss_pct: 止损百分比
        dynamic_stop_loss: 是否使用动态止损
        atr_multiplier: ATR倍数
        quiet: 是否静默运行
        
    Returns:
        Dict: 回测结果
    """
    from .rf4_features import generate_rf4_signals
    from ta.volatility import average_true_range
    
    initial_capital = 100000
    cash = initial_capital
    btc_holdings = 0.0
    commission = 0.001
    trades = []
    equity_curve = [initial_capital]
    current_trade = None
    
    # 计算ATR用于动态止损
    if dynamic_stop_loss:
        ohlcv_df = ohlcv_df.copy()
        ohlcv_df['atr'] = average_true_range(
            high=ohlcv_df['high'], 
            low=ohlcv_df['low'], 
            close=ohlcv_df['close'], 
            window=14
        )
    
    # 生成信号
    signals_df = generate_rf4_signals(ohlcv_df.copy(), period=period, order=order)
    
    for i in range(1, len(ohlcv_df)):
        current_price = ohlcv_df['close'].iloc[i]
        signal = signals_df['signal'].iloc[i]
        current_equity = cash + btc_holdings * current_price
        
        # 检查止损
        stop_loss_triggered = False
        if current_trade is not None:
            entry_price = current_trade['entry_price']
            is_long = current_trade['direction'] == 'long'
            
            if dynamic_stop_loss and i < len(ohlcv_df) and 'atr' in ohlcv_df.columns:
                current_atr = ohlcv_df['atr'].iloc[i]
                if not np.isnan(current_atr):
                    stop_loss_distance = current_atr * atr_multiplier
                    stop_loss_price = entry_price - stop_loss_distance if is_long else entry_price + stop_loss_distance
                else:
                    stop_loss_price = entry_price * (1 - stop_loss_pct) if is_long else entry_price * (1 + stop_loss_pct)
            else:
                stop_loss_price = entry_price * (1 - stop_loss_pct) if is_long else entry_price * (1 + stop_loss_pct)
            
            if (is_long and current_price <= stop_loss_price) or (not is_long and current_price >= stop_loss_price):
                stop_loss_triggered = True
        
        # 平仓条件
        should_exit = (current_trade is not None and 
                      ((current_trade['direction'] == 'long' and signal == -1) or
                       (current_trade['direction'] == 'short' and signal == 1) or
                       stop_loss_triggered))
        
        if should_exit:
            entry_price = current_trade['entry_price']
            trade_size = current_trade['size']
            is_long = current_trade['direction'] == 'long'
            
            if is_long:
                cash += trade_size * current_price * (1 - commission)
                btc_holdings -= trade_size
                pnl = trade_size * (current_price - entry_price) * (1 - commission) - trade_size * entry_price * commission
            else:
                cash -= trade_size * current_price * (1 + commission)
                btc_holdings += trade_size  # 做空平仓时归还BTC
                pnl = trade_size * (entry_price - current_price) - trade_size * entry_price * commission - trade_size * current_price * commission
            
            current_trade.update({
                'exit_price': current_price,
                'exit_date': ohlcv_df.index[i],
                'pnl': pnl,
                'return_pct': (pnl / current_trade['investment']) * 100,
                'exit_reason': 'stop_loss' if stop_loss_triggered else 'signal'
            })
            trades.append(current_trade)
            current_trade = None
        
        # 开仓条件
        if current_trade is None and signal != 0:
            trade_value = current_equity * position_size
            trade_size = trade_value / current_price
            
            if signal == 1:  # 做多
                if cash >= trade_value * (1 + commission):
                    cash -= trade_value * (1 + commission)
                    btc_holdings += trade_size
                    current_trade = {
                        'entry_date': ohlcv_df.index[i],
                        'entry_price': current_price,
                        'size': trade_size,
                        'direction': 'long',
                        'investment': trade_value
                    }
            elif signal == -1:  # 做空
                cash += trade_value * (1 - commission)
                btc_holdings -= trade_size  # 做空时BTC持仓为负数
                current_trade = {
                    'entry_date': ohlcv_df.index[i],
                    'entry_price': current_price,
                    'size': trade_size,
                    'direction': 'short',
                    'investment': trade_value
                }
        
        equity_curve.append(cash + btc_holdings * current_price)
    
    # 计算结果
    final_equity = equity_curve[-1]
    total_return = (final_equity / initial_capital - 1) * 100
    
    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    win_rate = 0
    profit_factor = 0
    
    if total_trades > 0:
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        win_rate = len(wins) / total_trades * 100
        
        sum_of_wins = wins['pnl'].sum()
        sum_of_losses = abs(losses['pnl'].sum())
        profit_factor = sum_of_wins / sum_of_losses if sum_of_losses > 0 else float('inf')
    
    # 计算最大回撤
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min()) * 100 if not drawdown.empty else 0
    
    results = {
        "total_return": total_return,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "final_equity": final_equity
    }
    
    if not quiet:
        print("\n--- RF4背离策略回测结果 ---")
        config_str = f"period={period}, order={order}, 仓位={position_size*100:.0f}%"
        if dynamic_stop_loss:
            config_str += f", 动态止损(ATR*{atr_multiplier:.1f})"
        else:
            config_str += f", 止损={stop_loss_pct*100:.1f}%"
        print(f"参数: {config_str}")
        print(f"回测周期: {ohlcv_df.index.min()} 到 {ohlcv_df.index.max()}")
        print(f"最终资产: ${results['final_equity']:,.2f}")
        print(f"总回报率: {results['total_return']:.2f}%")
        print(f"最大回撤: {results['max_drawdown']:.2f}%")
        print(f"总交易次数: {results['total_trades']}")
        print(f"胜率: {results['win_rate']:.2f}%")
        print(f"盈亏比: {results['profit_factor']:.2f}")
    
    return results

def get_rf4_signal(period: int = 14, order: int = 5) -> Optional[Dict[str, Any]]:
    """
    获取RF4背离策略的实时交易信号
    
    Args:
        period: RSI周期
        order: 背离检测的波峰/波谷查找范围
    
    Returns:
        Dict: 包含信号、当前价格、时间戳等信息的字典
    """
    from .data import get_data
    from .config import DATA_CONFIG
    from .rf4_features import generate_rf4_signals
    
    LOGGER.info(f"正在获取RF4背离策略信号 (period={period}, order={order})...")
    
    try:
        # 获取足够的历史数据以计算背离
        lookback_periods = max(order * 10, 100)  # 确保有足够数据检测背离
        price_data = get_data(
            symbol=DATA_CONFIG['symbol'],
            timeframe=DATA_CONFIG['timeframe'],
            limit=lookback_periods
        )
        
        if price_data is None or len(price_data) < lookback_periods:
            LOGGER.warning("获取的数据不足以计算RF4指标，无法生成信号。")
            return None
        
        # 生成RF4信号
        signals_df = generate_rf4_signals(price_data.copy(), period=period, order=order)
        
        # 安全获取最新信号
        if signals_df.empty or len(signals_df) == 0:
            LOGGER.warning("信号生成失败，返回持有信号")
            return {
                "signal": "HOLD",
                "action": "持有",
                "current_price": float(price_data['close'].iloc[-1]) if not price_data.empty else 0.0,
                "timestamp": price_data.index[-1].isoformat() if not price_data.empty else datetime.now(timezone.utc).isoformat(),
                "error": "信号生成失败"
            }
        
        latest_signal = signals_df['signal'].iloc[-1]
        current_price = price_data['close'].iloc[-1]
        
        # 验证信号值
        if pd.isna(latest_signal):
            latest_signal = 0  # NaN转为持有信号
        
        # 转换信号
        if latest_signal == 1:
            signal = "BUY"
            action = "做多"
        elif latest_signal == -1:
            signal = "SELL" 
            action = "做空"
        else:
            signal = "HOLD"
            action = "持有"
        
        result = {
            "signal": signal,
            "action": action,
            "current_price": float(current_price),
            "timestamp": price_data.index[-1].isoformat(),
            "rf4_value": float(signals_df['rf4'].iloc[-1]) if 'rf4' in signals_df.columns else None,
            "bullish_divergence": bool(signals_df['bullish_divergence'].iloc[-1]) if 'bullish_divergence' in signals_df.columns else False,
            "bearish_divergence": bool(signals_df['bearish_divergence'].iloc[-1]) if 'bearish_divergence' in signals_df.columns else False
        }
        
        LOGGER.info(f"RF4信号获取成功: {result['signal']} - {result['action']}")
        return result
        
    except Exception as e:
        LOGGER.error(f"获取RF4信号时发生错误: {e}")
        return {
            "signal": "HOLD",
            "action": "持有",
            "current_price": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": f"获取RF4信号时出错: {e}"
        }

def optimize_rf4_parameters(days: int = 365, n_trials: int = 50) -> Dict[str, Any]:
    """
    RF4策略参数优化
    
    Args:
        days: 回测天数
        n_trials: 优化试验次数
        
    Returns:
        Dict: 优化结果包含最佳参数和表现
    """
    try:
        import optuna
        from .data import get_data
        from .config import DATA_CONFIG
        from datetime import timedelta
        
        # 获取数据
        since_date = datetime.now() - timedelta(days=days)
        since_str = since_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        ohlcv_df = get_data(
            symbol=DATA_CONFIG['symbol'],
            timeframe=DATA_CONFIG['timeframe'],
            since=since_str
        )
        
        if ohlcv_df is None or ohlcv_df.empty:
            return {"error": "无法获取数据进行优化"}
        
        # 修正时区
        since_ts = pd.to_datetime(since_str)
        if ohlcv_df.index.tz is None:
            ohlcv_df.index = ohlcv_df.index.tz_localize('UTC')
        ohlcv_df = ohlcv_df[ohlcv_df.index >= since_ts]
        
        def objective(trial):
            period = trial.suggest_int('period', 5, 50)
            order = trial.suggest_int('order', 3, 20)
            position_size = trial.suggest_float('position_size', 0.1, 0.95, step=0.05)
            stop_loss_pct = trial.suggest_float('stop_loss_pct', 0.01, 0.10, step=0.005)
            
            # 动态止损选择
            use_dynamic_stop = trial.suggest_categorical('dynamic_stop_loss', [True, False])
            atr_multiplier = trial.suggest_float('atr_multiplier', 1.0, 4.0, step=0.25) if use_dynamic_stop else 2.0
            
            try:
                results = run_rf4_backtest(
                    ohlcv_df,
                    period=period,
                    order=order,
                    position_size=position_size,
                    stop_loss_pct=stop_loss_pct,
                    dynamic_stop_loss=use_dynamic_stop,
                    atr_multiplier=atr_multiplier,
                    quiet=True
                )
                
                # 风险调整收益 - 兼顾收益和回撤
                risk_adjusted_return = results['total_return'] / max(results['max_drawdown'], 1.0)
                return risk_adjusted_return
                
            except Exception:
                return -1000  # 回测失败返回极低分数
        
        # 运行优化
        study = optuna.create_study(direction='maximize')
        study.enqueue_trial({'period': 14, 'order': 5, 'position_size': 0.95, 'stop_loss_pct': 0.05, 'dynamic_stop_loss': False})
        study.optimize(objective, n_trials=n_trials)
        
        # 使用最佳参数运行最终回测
        best_params = study.best_params
        final_results = run_rf4_backtest(
            ohlcv_df,
            period=best_params['period'],
            order=best_params['order'],
            position_size=best_params['position_size'],
            stop_loss_pct=best_params['stop_loss_pct'],
            dynamic_stop_loss=best_params['dynamic_stop_loss'],
            atr_multiplier=best_params.get('atr_multiplier', 2.0),
            quiet=True
        )
        
        return {
            "best_params": best_params,
            "best_score": study.best_value,
            "backtest_results": final_results,
            "data_period": f"{days}天",
            "trials_completed": len(study.trials),
            "optimization_success": True
        }
        
    except ImportError:
        return {"error": "需要安装optuna: pip install optuna"}
    except Exception as e:
        return {"error": f"优化过程出错: {str(e)}"}


def get_bollinger_breakout_signal(window: int = 20, std_dev: float = 2.0) -> Optional[Dict[str, Any]]:
    """
    获取布林带突破策略的实时交易信号。

    Args:
        window (int): 布林带的时间窗口。
        std_dev (float): 布林带的标准差倍数。

    Returns:
        Dict: 包含信号、当前价格、时间戳等信息的字典。
    """
    from .data import get_data
    from .config import DATA_CONFIG
    from ta.volatility import BollingerBands

    LOGGER.info(f"正在获取布林带突破策略信号 (window={window}, std_dev={std_dev})...")

    try:
        # 获取足够的数据来计算布林带
        price_data = get_data(
            symbol=DATA_CONFIG['symbol'],
            timeframe=DATA_CONFIG['timeframe'],
            limit=window * 2  # 获取窗口两倍的数据量以确保准确性
        )

        if price_data is None or len(price_data) < window:
            LOGGER.warning("获取的数据不足以计算布林带，无法生成信号。")
            return None

        # 计算布林带
        indicator_bb = BollingerBands(close=price_data['close'], window=window, window_dev=std_dev)
        df = price_data.copy()
        df['bb_upper'] = indicator_bb.bollinger_hband()
        df['bb_lower'] = indicator_bb.bollinger_lband()

        # 获取最新的数据点
        latest = df.iloc[-1]
        previous = df.iloc[-2]

        signal = "HOLD"
        action = "持有"

        # 突破上轨
        if previous['close'] <= previous['bb_upper'] and latest['close'] > latest['bb_upper']:
            signal = "BUY"
            action = "做多 (布林带上轨突破)"
        # 跌破下轨
        elif previous['close'] >= previous['bb_lower'] and latest['close'] < latest['bb_lower']:
            signal = "SELL"
            action = "做空 (布林带下轨跌破)"

        result = {
            "signal": signal,
            "action": action,
            "current_price": float(latest['close']),
            "timestamp": latest.name.isoformat(),
            "strategy": "Bollinger_Breakout",
            "params": f"window={window}, std_dev={std_dev}"
        }

        LOGGER.info(f"布林带突破信号获取成功: {signal} - {action}")
        return result

    except Exception as e:
        LOGGER.error(f"获取布林带突破信号时发生错误: {e}")
        return {
            "signal": "HOLD",
            "action": "持有",
            "current_price": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": f"获取布林带信号时出错: {e}",
            "strategy": "Bollinger_Breakout"
        }