"""
网格交易启动脚本
"""
import sys
import os
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from highfreq.grid_trading import GridTrader
from btc_predictor.utils import LOGGER


def main():
    parser = argparse.ArgumentParser(description='网格交易策略')
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
        '--grid-count',
        type=int,
        default=10,
        help='网格数量，默认10'
    )
    parser.add_argument(
        '--price-range',
        type=float,
        default=0.05,
        help='价格区间（百分比），默认5%（±5%）'
    )
    parser.add_argument(
        '--order-amount',
        type=float,
        default=0.01,
        help='每格订单金额（BTC），默认0.01'
    )
    parser.add_argument(
        '--leverage',
        type=int,
        default=3,
        help='杠杆倍数，默认3x'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("📊 网格交易策略")
    print("="*80)
    print("策略说明：")
    print("  ✓ 在价格区间内设置多个买卖订单")
    print("  ✓ 价格下跌时买入，上涨时卖出")
    print("  ✓ 赚取价格波动差价")
    print("="*80)
    print(f"交易对: BTC-USDT-SWAP")
    print(f"最大资金: ${args.capital}")
    print(f"运行时长: {args.duration}分钟")
    print(f"网格数量: {args.grid_count}")
    print(f"价格区间: ±{args.price_range*100:.1f}%")
    print(f"每格金额: {args.order_amount} BTC")
    print(f"杠杆: {args.leverage}x")
    print("="*80)
    print("⚠️  这是实盘交易，将使用真实资金！")
    print("="*80)
    confirm = input("\n确认开始实盘交易？(输入 yes 继续): ")
    if confirm.lower() != 'yes':
        print("已取消")
        return
    print("\n开始运行...\n")
    
    # 创建网格交易器
    trader = GridTrader(
        symbol="BTC-USDT-SWAP",
        grid_count=args.grid_count,
        price_range_pct=args.price_range,
        order_amount=args.order_amount,
        max_capital_usd=args.capital,
        leverage=args.leverage,
        demo_mode=False,
    )
    
    # 开始交易
    try:
        LOGGER.info(f"[GRID] 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        trader.run_continuous(interval_sec=10.0, duration_min=args.duration)
        LOGGER.info(f"[GRID] 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except KeyboardInterrupt:
        LOGGER.info("[GRID] 用户中断")
    except Exception as e:
        LOGGER.error(f"[GRID] 运行失败: {e}", exc_info=True)


if __name__ == "__main__":
    main()

