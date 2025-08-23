import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
import numpy as np
from btc_predictor.data import get_data
from btc_predictor.features import create_features

def check_kline_with_features(df: pd.DataFrame):
    print(f"原始数据行数: {len(df)}")
    print(f"时间范围: {df.index.min()} ~ {df.index.max()}")
    print(f"时间间隔检查:")
    time_diffs = df.index.to_series().diff().dropna()
    print(f"  最小间隔: {time_diffs.min()}")
    print(f"  最大间隔: {time_diffs.max()}")
    print(f"  标准间隔: {time_diffs.mode().iloc[0] if not time_diffs.mode().empty else '未知'}")
    print(f"  间隔异常数: {(time_diffs != time_diffs.mode().iloc[0]).sum() if not time_diffs.mode().empty else '未知'}")
    print(f"是否按时间升序: {df.index.is_monotonic_increasing}")
    print(f"是否有缺失值: {df.isnull().any().any()}")
    
    # 检查基础OHLCV数据
    for col in ['open','high','low','close','volume']:
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            print(f"{col} 缺失值: {nan_count}")
        jumps = (df[col].diff().abs() > (df[col].std() * 5))
        if jumps.any():
            print(f"{col} 存在极端跳变: {jumps.sum()} 次")
    
    # 应用特征工程，生成均线等指标
    print("\n=== 应用特征工程生成均线等指标 ===")
    try:
        df_with_features = create_features(df.copy(),"btc-crossover-regression-v1")
        print(f"特征工程后数据行数: {len(df_with_features)}")
        print(f"新增列: {set(df_with_features.columns) - set(df.columns)}")
        
        # 检查均线指标
        ma_cols = ['ma_short', 'ma_long', 'ma60']
        for ma_col in ma_cols:
            if ma_col in df_with_features.columns:
                nan_count = df_with_features[ma_col].isnull().sum()
                print(f"{ma_col} 缺失值: {nan_count}")
                
                # 检查是否存在平线（连续相同值）
                flat_lines = (df_with_features[ma_col].diff() == 0).sum()
                print(f"{ma_col} 平线点数（连续相同值）: {flat_lines}")
                
                # 检查极端跳变
                if len(df_with_features[ma_col].dropna()) > 1:
                    jumps = (df_with_features[ma_col].diff().abs() > (df_with_features[ma_col].std() * 5))
                    if jumps.any():
                        print(f"{ma_col} 极端跳变: {jumps.sum()} 次")
                        jump_indices = df_with_features[jumps].index
                        for idx in jump_indices[:3]:  # 只显示前3个
                            prev_val = df_with_features[ma_col].shift(1).loc[idx]
                            curr_val = df_with_features[ma_col].loc[idx]
                            print(f"  {idx}: {prev_val:.2f} -> {curr_val:.2f}")
        
        # 检查布林带
        bb_cols = ['bb_upper', 'bb_lower', 'bb_width']
        for bb_col in bb_cols:
            if bb_col in df_with_features.columns:
                nan_count = df_with_features[bb_col].isnull().sum()
                if nan_count > 0:
                    print(f"{bb_col} 缺失值: {nan_count}")
        
        # 输出最后几行数据供检查
        print("\n=== 最后5行数据样本 ===")
        display_cols = ['close', 'ma_short', 'ma_long', 'ma60']
        available_cols = [col for col in display_cols if col in df_with_features.columns]
        print(df_with_features[available_cols].tail())
        
        return df_with_features
        
    except Exception as e:
        print(f"特征工程失败: {e}")
        return df

if __name__ == "__main__":
    df = get_data(symbol='BTC/USDT', timeframe='1h', limit=200)
    df_final = check_kline_with_features(df)
