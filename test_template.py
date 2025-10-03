#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
邮件模板测试脚本 - 验证f-string渲染
"""

import sys
import os
from datetime import datetime, timezone

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.email_notifier import EmailNotifier


def test_template_rendering():
    """测试邮件模板f-string渲染"""
    print("🧪 测试邮件模板f-string渲染...")

    # 初始化邮件通知器
    email_notifier = EmailNotifier()

    if not email_notifier.enabled:
        print("❌ 邮件通知功能未启用")
        return

    # 准备测试数据
    test_decision_data = {
        "decision": "LONG",
        "confidence": 0.85,
        "reasoning": "这是测试决策理由，应该正确显示",
        "key_signals_detected": "RSI超卖反弹、布林带上轨突破、成交量放大",
        "risk_assessment": "市场波动较大，建议控制仓位大小，严格执行止损",
        "trade_params": {
            "leverage": 3,
            "take_profit_pct": 8.0,
            "stop_loss_pct": 4.0
        },
        "position_snapshot": {
            "pnl_usd": 125.50,
            "desc": "多仓 | 数量: 0.001 BTC | 开仓均价: $43,250"
        },
        "symbol": "BTC-USDT-SWAP"
    }

    try:
        # 生成邮件HTML内容
        html_content = email_notifier._create_decision_email_html(
            decision_data=test_decision_data,
            execution_success=True,
            error_msg=None,
            process_status=None
        )

        # 检查关键内容是否正确渲染
        checks = [
            ("决策理由", "这是测试决策理由，应该正确显示"),
            ("置信度", "85.0%"),
            ("杠杆", "3x"),
            ("止盈", "8.0%"),
            ("止损", "4.0%"),
            ("持仓盈亏", "+$125.50 USDT"),
            ("图标", "fas fa-robot"),  # 检查图标是否正确
        ]

        print("\n📋 检查渲染结果:")
        all_passed = True

        for check_name, expected_content in checks:
            if expected_content in html_content:
                print(f"✅ {check_name}: 正确渲染")
            else:
                print(f"❌ {check_name}: 未找到预期内容 '{expected_content}'")
                all_passed = False

        # 检查是否还有未渲染的占位符
        if "{trade_params.get" in html_content:
            print("❌ 发现未渲染的f-string占位符")
            all_passed = False
        else:
            print("✅ 所有f-string占位符已正确渲染")

        if all_passed:
            print("\n🎉 邮件模板渲染测试通过！")
            return True
        else:
            print("\n❌ 邮件模板渲染测试失败！")
            return False

    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        return False


if __name__ == "__main__":
    success = test_template_rendering()
    sys.exit(0 if success else 1)
