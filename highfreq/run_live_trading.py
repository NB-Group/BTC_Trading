"""
高频策略实盘交易启动脚本
"""
import sys
import os
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from highfreq.hf_live_trader import HFLiveTrader
import config
from btc_predictor.utils import LOGGER


def main():
    parser = argparse.ArgumentParser(description='高频策略实盘交易')
    parser.add_argument(
        '--demo',
        action='store_true',
        help='使用模拟模式（不真实交易）'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=500.0,
        help='最大使用资金（美元），默认500'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='运行时长（分钟），默认60分钟'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='检查间隔（秒），默认1秒'
    )
    parser.add_argument(
        '--use-taker',
        action='store_true',
        help='使用Taker订单（默认使用Maker订单）'
    )
    
    args = parser.parse_args()
    
    # 确定是否使用模拟模式（默认实盘）
    demo_mode = args.demo  # 只有明确指定--demo才使用模拟模式
    
    if not demo_mode:
        print("="*80)
        print("🚀 高频策略实盘交易")
        print("="*80)
        print(f"交易对: BTC-USDT-SWAP")
        print(f"最大资金: ${args.capital}")
        print(f"运行时长: {args.duration}分钟")
        print(f"检查间隔: {args.interval}秒")
        print(f"订单类型: {'Taker' if args.use_taker else 'Maker'}")
        print("="*80)
        print("⚠️  这是实盘交易，将使用真实资金！")
        print("="*80)
        confirm = input("\n确认开始实盘交易？(输入 yes 继续): ")
        if confirm.lower() != 'yes':
            print("已取消")
            return
        print("\n开始运行...\n")
    else:
        print("="*80)
        print("🧪 模拟模式：不会真实交易")
        print("="*80)
        print()
    
    # 创建交易器
    trader = HFLiveTrader(
        symbol="BTC-USDT-SWAP",
        max_capital_usd=args.capital,
        bias_long=5e-7,
        bias_short=-5e-7,
        ofi_long=0.0,
        ofi_short=0.0,
        tp_pct=0.0010,
        sl_pct=0.0012,
        min_depth_total=2.0,
        max_spread_bps=2.0,
        cooldown_sec=10,
        time_stop_sec=30,
        ofi_span=8,
        use_taker=args.use_taker,
        demo_mode=demo_mode,
    )
    
    # 开始交易
    try:
        LOGGER.info(f"[HF-LIVE] 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        trader.run_continuous(interval_sec=args.interval, duration_min=args.duration)
        LOGGER.info(f"[HF-LIVE] 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except KeyboardInterrupt:
        LOGGER.info("[HF-LIVE] 用户中断")
    except Exception as e:
        LOGGER.error(f"[HF-LIVE] 运行失败: {e}", exc_info=True)


if __name__ == "__main__":
    main()

