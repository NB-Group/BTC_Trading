from execution_engine.okx_trader import OKXTrader
import time

if __name__ == "__main__":
    trader = OKXTrader(demo_mode=False)
    print("准备开空仓...\n")
    # 直接调用最大可开张数逻辑
    # 只需传递最小决策参数，实际张数由trader自动获取
    decision_data = {
        'decision': 'LONG',
        'reasoning': '测试脚本开多仓',
        'key_signals_detected': '无',
        'confidence': 0.8,
        'trade_params': {
            'leverage': 3
        },
        'risk_assessment': '测试用，无实际风险'
    }
    trader.execute_decision(decision_data)
    print("已调用开仓逻辑，请检查日志和OKX账户。")
    time.sleep(5)
