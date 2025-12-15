"""
高频策略实盘交易启动脚本 - 优化版本
- 使用Taker订单（立即成交）
- 更保守的参数（提高信号质量）
- 小资金短时间测试
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
    parser = argparse.ArgumentParser(description='高频策略实盘交易 - 优化版本')
    parser.add_argument(
        '--capital',
        type=float,
        default=300.0,
        help='最大使用资金（美元），默认300（更保守）'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='运行时长（分钟），默认30分钟'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='检查间隔（秒），默认1秒'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 高频策略实盘交易 - 优化版本")
    print("="*80)
    print("优化措施：")
    print("  ✓ 使用Taker订单（立即成交，避免信号失效）")
    print("  ✓ 优化信号阈值（4e-7，平衡质量和频率）")
    print("  ✓ 信号确认机制（需要连续2个信号才开仓）")
    print("  ✓ 优化止盈止损（止盈0.12%覆盖成本，止损0.08%快速止损）")
    print("  ✓ 更严格的过滤条件（提高信号质量）")
    print("  ✓ 小资金短时间测试（降低风险）")
    print("="*80)
    print(f"交易对: BTC-USDT-SWAP")
    print(f"最大资金: ${args.capital}")
    print(f"运行时长: {args.duration}分钟")
    print(f"检查间隔: {args.interval}秒")
    print("="*80)
    print("⚠️  这是实盘交易，将使用真实资金！")
    print("="*80)
    confirm = input("\n确认开始实盘交易？(输入 yes 继续): ")
    if confirm.lower() != 'yes':
        print("已取消")
        return
    print("\n开始运行...\n")
    
    # 创建交易器 - 使用优化参数（策略优化版）
    trader = HFLiveTrader(
        symbol="BTC-USDT-SWAP",
        max_capital_usd=args.capital,
        # 信号阈值优化：平衡信号质量和频率
        # 从日志看，micro_bias=-2.47e-07，阈值6e-7太高导致无信号
        # 调整为4e-7，既能捕捉信号，又能保持一定质量
        bias_long=4e-7,      # 降低阈值（从6e-7降到4e-7），增加信号频率
        bias_short=-4e-7,    # 降低阈值（从-6e-7降到-4e-7），增加信号频率
        ofi_long=0.0,        # OFI过滤暂时关闭（简化逻辑）
        ofi_short=0.0,
        # 止盈止损优化：考虑手续费后的净收益
        # 实际手续费可能>0.10%（开仓+平仓），所以止盈至少需要0.12%才能覆盖成本
        tp_pct=0.0012,       # 提高止盈到0.12%（确保覆盖手续费成本）
        sl_pct=0.0008,       # 降低止损到0.08%（更快止损，减少大亏）
        # 过滤条件优化：更严格的过滤，提高信号质量
        min_depth_total=3.0,  # 提高深度要求（从2.0提高到3.0），确保流动性充足
        max_spread_bps=1.5,   # 降低点差要求（从2.0降低到1.5），避免在点差过大时交易
        cooldown_sec=15,      # 增加冷却期（从10秒提高到15秒），减少过度交易
        time_stop_sec=120,    # 延长时间止损（从60秒提高到120秒），给价格更多时间移动
        ofi_span=8,
        use_taker=True,       # 使用Taker订单（立即成交）
        demo_mode=False,      # 实盘模式
    )
    
    # 开始交易
    try:
        LOGGER.info(f"[HF-LIVE-OPT] 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        trader.run_continuous(interval_sec=args.interval, duration_min=args.duration)
        LOGGER.info(f"[HF-LIVE-OPT] 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except KeyboardInterrupt:
        LOGGER.info("[HF-LIVE-OPT] 用户中断")
    except Exception as e:
        LOGGER.error(f"[HF-LIVE-OPT] 运行失败: {e}", exc_info=True)


if __name__ == "__main__":
    main()

