#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发送测试邮件脚本
运行方式：python send_test_email.py
"""

from datetime import datetime, timezone
from utils.email_notifier import EmailNotifier


def main():
    notifier = EmailNotifier()
    if not notifier.enabled:
        print("❌ 邮件通知功能未启用，请检查 EMAIL_CONFIG")
        return 1

    decision = {
        "decision": "LONG",
        "reasoning": "测试邮件：验证模板、图标与f-string渲染是否正确。",
        "key_signals_detected": "MA交叉；成交量放大；RSI回踩50上方",
        "risk_assessment": "波动中等，控制仓位，严格止损。",
        "trade_params": {
            "leverage": 3,
            "take_profit_pct": 6.0,
            "stop_loss_pct": 3.0,
        },
        "position_snapshot": {
            "pnl_usd": 18.75,
            "desc": "多仓 | 数量: 0.001 BTC | 开仓均价: $43,250",
        },
        "symbol": "BTC-USDT-SWAP",
    }

    process_status = {
        "data_collection": {"status": "success", "duration": "1.2s", "message": "数据获取完成"},
        "vlm_analysis": {"status": "success", "duration": "1.0s", "message": "VLM分析完成"},
        "news_intelligence": {"status": "warning", "duration": "0.6s", "message": "部分新闻源超时"},
        "llm_decision": {"status": "success", "duration": "2.3s", "message": "决策生成完成"},
        "trade_execution": {"status": "success", "duration": "0.9s", "message": "模拟执行完成"},
    }

    try:
        notifier.send_decision_notification(decision, execution_success=True, process_status=process_status)
        print("📧 已触发发送测试决策邮件，请查收收件箱。")
        return 0
    except Exception as e:
        print(f"❌ 发送失败: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


