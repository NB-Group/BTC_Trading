"""
高频策略参数优化器：同时测试多个参数组合并对比结果
"""
import pandas as pd
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from btc_predictor.utils import LOGGER
from .hf_orderbook import (
    record_orderbook_series,
    backtest_micro_ofi,
)


def test_single_param_set(
    symbol: str,
    param_set: Dict[str, Any],
    samples: int = 1800,
    interval_sec: float = 1.0,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    测试单个参数组合。
    
    Args:
        symbol: 交易对
        param_set: 参数字典
        samples: 样本数（如果df为None时使用）
        interval_sec: 采样间隔（如果df为None时使用）
        df: 可选的共享数据，如果提供则跳过录制
    
    Returns:
        包含参数和回测结果的字典
    """
    try:
        LOGGER.info(f"测试参数组合: {param_set}")
        
        # 如果提供了共享数据，直接使用；否则录制
        if df is None:
            # 多线程安全：重置exchange实例
            from btc_predictor import data as d
            d._exchange = None
            
            # 录制盘口数据
            df = record_orderbook_series(
                symbol=symbol,
                samples=samples,
                interval_sec=interval_sec,
                depth=5,
                levels=3,
            )
        else:
            # 使用共享数据，创建副本避免修改
            df = df.copy()
        
        if df.empty:
            return {**param_set, "trades": 0, "win_rate": 0.0, "pnl_pct": 0.0, "final_equity": 1.0, "error": "empty_data"}
        
        # 回测
        res = backtest_micro_ofi(
            df,
            bias_long=param_set.get("bias_long", 5e-7),
            bias_short=param_set.get("bias_short", -5e-7),
            ofi_long=param_set.get("ofi_long", 0.0),
            ofi_short=param_set.get("ofi_short", 0.0),
            tp_pct=param_set.get("tp_pct", 0.0010),
            sl_pct=param_set.get("sl_pct", 0.0012),
            maker_rebate=param_set.get("maker_rebate", 0.0),
            taker_fee=param_set.get("taker_fee", 0.00028),
            use_taker=param_set.get("use_taker", False),
            min_depth_total=param_set.get("min_depth_total", 2.0),
            max_spread_bps=param_set.get("max_spread_bps", 2.0),
            cooldown_bars=param_set.get("cooldown_bars", 10),
            time_stop_bars=param_set.get("time_stop_bars", 30),
            slippage_bps=param_set.get("slippage_bps", 0.2),
            verbose=False,
        )
        
        if res is None:
            return {**param_set, "trades": 0, "win_rate": 0.0, "pnl_pct": 0.0, "final_equity": 1.0, "error": "backtest_failed"}
        
        return {**param_set, **res}
        
    except Exception as e:
        LOGGER.error(f"参数组合测试失败: {param_set}, 错误: {e}")
        return {**param_set, "trades": 0, "win_rate": 0.0, "pnl_pct": 0.0, "final_equity": 1.0, "error": str(e)}


def optimize_micro_ofi_params(
    symbol: str = "BTC/USDT",
    samples: int = 1800,
    interval_sec: float = 1.0,
    max_workers: int = 1,  # 串行测试避免API限流
) -> pd.DataFrame:
    """
    测试多个参数组合并返回对比结果。
    
    参数网格：
    - bias_long/bias_short: [5e-7, 7e-7, 1e-6]
    - min_depth_total: [2.0, 2.3, 2.5]
    - max_spread_bps: [1.5, 1.8, 2.0]
    """
    # 参数网格（可根据需要调整）
    # 快速版本：只测试关键组合
    bias_values = [5e-7, 7e-7, 1e-6]
    depth_values = [2.0, 2.3, 2.5]
    spread_values = [1.5, 1.8, 2.0]
    
    # 如果想快速测试，可以只选几个值：
    # bias_values = [5e-7, 7e-7]
    # depth_values = [2.0, 2.3]
    # spread_values = [1.8, 2.0]
    
    param_sets = []
    for bias in bias_values:
        for depth in depth_values:
            for spread in spread_values:
                param_sets.append({
                    "bias_long": bias,
                    "bias_short": -bias,
                    "ofi_long": 0.0,
                    "ofi_short": 0.0,
                    "tp_pct": 0.0010,
                    "sl_pct": 0.0012,
                    "maker_rebate": 0.0,
                    "taker_fee": 0.00028,
                    "use_taker": False,
                    "min_depth_total": depth,
                    "max_spread_bps": spread,
                    "cooldown_bars": 10,
                    "time_stop_bars": 30,
                    "slippage_bps": 0.2,
                })
    
    LOGGER.info(f"开始测试 {len(param_sets)} 个参数组合...")
    
    results = []
    if max_workers == 1:
        # 串行测试
        for i, param_set in enumerate(param_sets, 1):
            LOGGER.info(f"[{i}/{len(param_sets)}] 测试参数组合...")
            result = test_single_param_set(symbol, param_set, samples, interval_sec)
            results.append(result)
            print(f"完成 {i}/{len(param_sets)}: trades={result.get('trades', 0)}, pnl={result.get('pnl_pct', 0):.4f}%")
    else:
        # 并行测试（谨慎使用，避免API限流）
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(test_single_param_set, symbol, param_set, samples, interval_sec): param_set
                for param_set in param_sets
            }
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(f"完成 {i}/{len(param_sets)}: trades={result.get('trades', 0)}, pnl={result.get('pnl_pct', 0):.4f}%")
    
    # 转换为DataFrame并排序
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("pnl_pct", ascending=False)
    
    return df_results


def print_optimization_results(df: pd.DataFrame, top_n: int = 10):
    """打印优化结果，显示前N个最佳参数组合"""
    print("\n" + "="*100)
    print(f"参数优化结果 - 前 {top_n} 个最佳组合")
    print("="*100)
    
    top_results = df.head(top_n)
    
    for i, row in top_results.iterrows():
        print(f"\n排名 #{list(top_results.index).index(i) + 1}:")
        print(f"  trades: {row.get('trades', 0)}")
        print(f"  win_rate: {row.get('win_rate', 0):.3f}")
        print(f"  pnl_pct: {row.get('pnl_pct', 0):.4f}%")
        print(f"  final_equity: {row.get('final_equity', 1.0):.5f}")
        print(f"  参数:")
        print(f"    bias_long: {row.get('bias_long', 0)}")
        print(f"    min_depth_total: {row.get('min_depth_total', 0)}")
        print(f"    max_spread_bps: {row.get('max_spread_bps', 0)}")
        if 'error' in row and pd.notna(row['error']):
            print(f"    错误: {row['error']}")
    
    print("\n" + "="*100)
    print("完整结果已保存，可用 df_results 查看")
    print("="*100 + "\n")


if __name__ == "__main__":
    from btc_predictor import data as d
    d.DATA_CONFIG['exchange'] = 'okx'
    d._exchange = None
    
    # 运行优化（30分钟测试，串行避免限流）
    df_results = optimize_micro_ofi_params(
        symbol="BTC/USDT",
        samples=1800,  # 30分钟
        interval_sec=1.0,
        max_workers=1,  # 串行
    )
    
    # 打印结果
    print_optimization_results(df_results, top_n=10)
    
    # 保存到CSV
    df_results.to_csv("hf_param_optimization_results.csv", index=False)
    print("结果已保存到: hf_param_optimization_results.csv")

