import pandas as pd
import numpy as np
import sys

# 用法: python check_kline_data.py <csv_path>
def check_kline(df: pd.DataFrame):
    print(f"数据行数: {len(df)}")
    print(f"时间范围: {df.index.min()} ~ {df.index.max()}")
    print(f"是否按时间升序: {df.index.is_monotonic_increasing}")
    print(f"是否有缺失值: {df.isnull().any().any()}")
    print(f"close列异常: {df['close'].describe()}")
    print(f"volume列异常: {df['volume'].describe()}")
    for col in ['open','high','low','close','volume']:
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            print(f"{col} 缺失值: {nan_count}")
    # 检查是否有极端跳变
    for col in ['close','open','high','low']:
        jumps = np.abs(df[col].diff()) > (df[col].std() * 5)
        if jumps.any():
            print(f"{col} 存在极端跳变: {jumps.sum()} 次")
    # 检查均线
    for ma in [5,10,20,60]:
        ma_col = f"ma{ma}"
        if ma_col in df.columns:
            nan_count = df[ma_col].isnull().sum()
            print(f"{ma_col} 均线缺失值: {nan_count}")
            jumps = np.abs(df[ma_col].diff()) > (df[ma_col].std() * 5)
            if jumps.any():
                print(f"{ma_col} 均线极端跳变: {jumps.sum()} 次")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_kline_data.py <csv_path>")
        sys.exit(1)
    df = pd.read_csv(sys.argv[1], index_col=0, parse_dates=True)
    check_kline(df)
