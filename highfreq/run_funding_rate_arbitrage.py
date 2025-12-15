"""
资金费率套利启动脚本
"""
import sys
import os
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from highfreq.funding_rate_arbitrage import FundingRateArbitrage
from btc_predictor.utils import LOGGER


def main():
    parser = argparse.ArgumentParser(description='资金费率套利策略')
    parser.add_argument(
        '--capital',
        type=float,
        default=1000.0,
        help='最大使用资金（美元），默认1000'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=1440,
        help='运行时长（分钟），默认1440分钟（24小时）'
    )
    parser.add_argument(
        '--min-funding-rate',
        type=float,
        default=0.008,
        help='最小资金费率（百分比），默认0.008%（即0.00008），降低阈值以增加交易机会'
    )
    parser.add_argument(
        '--leverage',
        type=int,
        default=3,
        help='杠杆倍数，默认3x'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=60.0,
        help='检查间隔（秒），默认60秒'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("💰 资金费率套利策略")
    print("="*80)
    print("策略说明：")
    print("  ✓ 监控永续合约资金费率")
    print("  ✓ 当资金费率足够高时，通过现货+期货对冲赚取资金费率")
    print("  ✓ 每8小时收取一次资金费率")
    print("  ⚠️  需要现货账户和期货账户")
    print("="*80)
    print(f"现货交易对: BTC/USDT")
    print(f"期货交易对: BTC-USDT-SWAP")
    print(f"最大资金: ${args.capital}")
    print(f"运行时长: {args.duration}分钟 ({args.duration/60:.1f}小时)")
    print(f"最小资金费率: {args.min_funding_rate}%")
    print(f"杠杆: {args.leverage}x")
    print(f"检查间隔: {args.interval}秒")
    print("="*80)
    print("⚠️  这是实盘交易，将使用真实资金！")
    print("⚠️  需要现货账户和期货账户同时可用！")
    print("="*80)
    confirm = input("\n确认开始实盘交易？(输入 yes 继续): ")
    if confirm.lower() != 'yes':
        print("已取消")
        return
    print("\n开始运行...\n")
    
    # 创建资金费率套利交易器
    trader = FundingRateArbitrage(
        spot_symbol="BTC/USDT",
        swap_symbol="BTC-USDT-SWAP",
        min_funding_rate=args.min_funding_rate,
        max_capital_usd=args.capital,
        leverage=args.leverage,
        demo_mode=False,
    )
    
    # 开始交易
    try:
        LOGGER.info(f"[FR-ARB] 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        trader.run_continuous(interval_sec=args.interval, duration_min=args.duration)
        LOGGER.info(f"[FR-ARB] 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except KeyboardInterrupt:
        LOGGER.info("[FR-ARB] 用户中断")
        # 如果有持仓，平仓
        if trader.spot_position and trader.swap_position:
            trader._close_arbitrage_position("用户中断")
    except Exception as e:
        LOGGER.error(f"[FR-ARB] 运行失败: {e}", exc_info=True)


if __name__ == "__main__":
    main()

