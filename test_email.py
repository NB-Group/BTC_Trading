#!/usr/bin/env python3
"""
邮件通知功能测试脚本
"""

import os
from dotenv import load_dotenv
from utils.email_notifier import EmailNotifier

def test_email_notification():
    """测试邮件通知功能"""
    
    # 加载环境变量
    load_dotenv()
    
    # 初始化邮件通知器
    email_notifier = EmailNotifier()
    
    if not email_notifier.enabled:
        print("❌ 邮件通知功能未启用，请检查 .env 配置")
        return
    
    print("✅ 邮件通知器初始化成功")
    
    # 测试决策通知
    test_decision = {
        "decision": "LONG",
        "confidence": 0.75,
        "reasoning": "根据1H K线分析，均线呈现多头排列，价格在布林带中轨和上轨之间运行，显示短期上升趋势强劲。",
        "key_signals_detected": "无关键风险信号",
        "risk_assessment": "主要风险：价格未能突破阻力位可能导致回调。",
        "trade_params": {
            "leverage": 3,
            "take_profit_pct": 5.0,
            "stop_loss_pct": 2.5
        }
    }
    
    print("📧 发送测试决策通知邮件...")
    try:
        email_notifier.send_decision_notification(test_decision, execution_success=True)
        print("✅ 决策通知邮件发送成功")
    except Exception as e:
        print(f"❌ 决策通知邮件发送失败: {e}")
    
    # 测试错误通知
    print("📧 发送测试错误通知邮件...")
    try:
        email_notifier.send_error_notification(
            "测试错误", 
            "这是一个测试错误消息，用于验证邮件通知功能。",
            context={
                "测试字段1": "测试值1",
                "测试字段2": "测试值2"
            }
        )
        print("✅ 错误通知邮件发送成功")
    except Exception as e:
        print(f"❌ 错误通知邮件发送失败: {e}")

if __name__ == "__main__":
    test_email_notification() 