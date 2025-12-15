"""
快速参数对比测试（仅测试关键组合）- 多线程并行版本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from btc_predictor import data as d
from btc_predictor.utils import LOGGER
from highfreq.param_optimizer import test_single_param_set, print_optimization_results

# 设置交易所
d.DATA_CONFIG['exchange'] = 'okx'
d._exchange = None

# 快速测试：只测试8个关键组合
param_sets = [
    # 基础组合（已验证有效）
    {"bias_long": 5e-7, "bias_short": -5e-7, "min_depth_total": 2.0, "max_spread_bps": 2.0},
    {"bias_long": 7e-7, "bias_short": -7e-7, "min_depth_total": 2.0, "max_spread_bps": 2.0},
    {"bias_long": 5e-7, "bias_short": -5e-7, "min_depth_total": 2.3, "max_spread_bps": 1.8},
    {"bias_long": 7e-7, "bias_short": -7e-7, "min_depth_total": 2.3, "max_spread_bps": 1.8},
    # 优化组合
    {"bias_long": 5e-7, "bias_short": -5e-7, "min_depth_total": 2.3, "max_spread_bps": 2.0},
    {"bias_long": 7e-7, "bias_short": -7e-7, "min_depth_total": 2.0, "max_spread_bps": 1.8},
    {"bias_long": 1e-6, "bias_short": -1e-6, "min_depth_total": 2.3, "max_spread_bps": 1.8},
    {"bias_long": 1e-6, "bias_short": -1e-6, "min_depth_total": 2.5, "max_spread_bps": 1.5},
]

# 添加默认参数
for ps in param_sets:
    ps.setdefault("ofi_long", 0.0)
    ps.setdefault("ofi_short", 0.0)
    ps.setdefault("tp_pct", 0.0010)
    ps.setdefault("sl_pct", 0.0012)
    ps.setdefault("maker_rebate", 0.0)
    ps.setdefault("taker_fee", 0.00028)
    ps.setdefault("use_taker", False)
    ps.setdefault("cooldown_bars", 10)
    ps.setdefault("time_stop_bars", 30)
    ps.setdefault("slippage_bps", 0.2)

if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="高频策略参数优化测试")
    parser.add_argument("--wait", type=int, default=10, help="等待时间（分钟），让API限流恢复，默认10分钟（OKX限流较严格）")
    args = parser.parse_args()
    
    print(f"开始并行测试 {len(param_sets)} 个参数组合（30分钟数据/组合）...")
    print("注意：使用2秒间隔避免API限流，录制30分钟数据需要约30分钟")
    print("多线程并行执行，预计总时间: 约 40 分钟（10分钟等待 + 30分钟录制）\n")
    
    # 等待API限流恢复
    if args.wait > 0:
        print(f"等待 {args.wait} 分钟让API限流恢复...")
        for i in range(args.wait * 60, 0, -10):
            if i % 60 == 0:
                print(f"  剩余 {i//60} 分钟...")
            time.sleep(10)
        print("等待完成，开始录制数据\n")
    
    # 先录制一次数据，所有线程共享
    # 注意：使用2秒间隔避免API限流，30分钟需要900个样本
    print("正在录制共享数据（30分钟，2秒间隔）...")
    from highfreq.hf_orderbook import record_orderbook_series
    shared_df = record_orderbook_series(
        symbol="BTC/USDT",
        samples=900,  # 30分钟 / 2秒 = 900个样本
        interval_sec=2.0,  # 2秒间隔，避免API限流
        depth=5,
        levels=3,
    )
    print(f"数据录制完成，共 {len(shared_df)} 条记录\n")
    
    if shared_df.empty:
        print("错误：录制数据为空，无法继续测试")
        exit(1)
    
    results = []
    with ThreadPoolExecutor(max_workers=len(param_sets)) as executor:
        # 提交所有任务，传递共享数据
        futures = {
            executor.submit(test_single_param_set, "BTC/USDT", param_set, 1800, 1.0, shared_df): (i, param_set)
            for i, param_set in enumerate(param_sets, 1)
        }
        
        # 收集结果
        for future in as_completed(futures):
            i, param_set = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[完成 {i}/{len(param_sets)}] bias={param_set['bias_long']}, depth={param_set['min_depth_total']}, spread={param_set['max_spread_bps']}")
                print(f"  结果: trades={result.get('trades', 0)}, win_rate={result.get('win_rate', 0):.3f}, pnl={result.get('pnl_pct', 0):.4f}%")
            except Exception as e:
                print(f"[失败 {i}/{len(param_sets)}] 错误: {e}")
                results.append({**param_set, "trades": 0, "win_rate": 0.0, "pnl_pct": 0.0, "final_equity": 1.0, "error": str(e)})
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("pnl_pct", ascending=False)
    
    print_optimization_results(df_results, top_n=len(df_results))
    
    # 保存结果
    df_results.to_csv("hf_quick_param_test_results.csv", index=False)
    print("结果已保存到: hf_quick_param_test_results.csv")

