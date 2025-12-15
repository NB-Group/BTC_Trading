"""2小时micro+OFI策略验证测试 - 详细日志版本"""
import time
from datetime import datetime
from btc_predictor import data as d
d.DATA_CONFIG['exchange'] = 'okx'
d._exchange = None

from highfreq.hf_orderbook import stream_record_and_backtest_micro, backtest_micro_ofi

print("="*80)
print("2小时micro+OFI策略验证测试")
print("="*80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n策略参数:")
print("  - bias_long: 5e-7, bias_short: -5e-7")
print("  - min_depth_total: 2.0")
print("  - max_spread_bps: 2.0")
print("  - tp_pct: 0.10%, sl_pct: 0.12%")
print("  - cooldown: 10 bars, time_stop: 30 bars")
print("  - slippage: 0.2bp, maker-only")
print("\n测试配置:")
print("  - 总样本数: 7200 (2小时 @ 1秒间隔)")
print("  - 分段大小: 600样本 (10分钟)")
print("  - 预计总时间: 约2小时")
print("="*80)
print("\n开始录制数据...\n")

start_time = time.time()

df = stream_record_and_backtest_micro(
    'BTC/USDT',
    total_samples=7200,
    interval_sec=1.0,
    chunk_size=600,
    depth=5,
    levels=3,
    bias_long=5e-7,
    bias_short=-5e-7,
    ofi_long=0.0,
    ofi_short=0.0,
    tp_pct=0.0010,
    sl_pct=0.0012,
    maker_rebate=0.0,
    taker_fee=0.00028,
    use_taker=False,
    min_depth_total=2.0,
    max_spread_bps=2.0,
    cooldown_bars=10,
    time_stop_bars=30,
    slippage_bps=0.2,
    ofi_span=8,
    log_every=100  # 更频繁的日志
)

record_time = time.time() - start_time
print(f"\n数据录制完成，耗时: {record_time/60:.1f} 分钟")
print(f"录制样本数: {len(df)}")
print(f"\n开始回测...\n")

backtest_start = time.time()

res = backtest_micro_ofi(
    df,
    bias_long=5e-7,
    bias_short=-5e-7,
    ofi_long=0.0,
    ofi_short=0.0,
    tp_pct=0.0010,
    sl_pct=0.0012,
    maker_rebate=0.0,
    taker_fee=0.00028,
    use_taker=False,
    min_depth_total=2.0,
    max_spread_bps=2.0,
    cooldown_bars=10,
    time_stop_bars=30,
    slippage_bps=0.2,
    log_every=50,  # 更频繁的日志
    verbose=True
)

backtest_time = time.time() - backtest_start
total_time = time.time() - start_time

print('\n' + '='*80)
print('最终回测结果')
print('='*80)
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"总耗时: {total_time/60:.1f} 分钟 (录制: {record_time/60:.1f}分钟, 回测: {backtest_time:.1f}秒)")
print()
print("交易统计:")
print(f"  - 总交易数: {res['trades']}")
if res['trades'] > 0:
    print(f"  - 胜率: {res['win_rate']*100:.2f}%")
    print(f"  - 盈利交易: {int(res['trades'] * res['win_rate'])}")
    print(f"  - 亏损交易: {int(res['trades'] * (1 - res['win_rate']))}")
print()
print("收益统计:")
print(f"  - 总收益率: {res['pnl_pct']:.4f}%")
print(f"  - 最终权益: {res['final_equity']:.5f}")
if res['trades'] > 0:
    avg_pnl = res['pnl_pct'] / res['trades']
    print(f"  - 平均每笔收益: {avg_pnl:.4f}%")
print()
print("性能评估:")
if res['pnl_pct'] > 0:
    print("  ✓ 策略盈利")
    if res['win_rate'] >= 0.5:
        print("  ✓ 胜率 >= 50%")
    else:
        print("  ⚠ 胜率 < 50%，但总体盈利")
else:
    print("  ✗ 策略亏损")
    if res['win_rate'] < 0.5:
        print("  ✗ 胜率 < 50%")
    else:
        print("  ⚠ 胜率 >= 50%，但总体亏损")
print('='*80)

