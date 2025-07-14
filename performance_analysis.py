#!/usr/bin/env python3
"""
综合性能分析脚本
计算风险管理模型的详细性能指标
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_comprehensive_metrics(portfolio_df, signals_df, trades_df, initial_capital=10000):
    """计算综合性能指标"""
    
    print("=" * 80)
    print("📊 风险管理模型综合性能分析")
    print("=" * 80)
    
    # 基础数据
    final_value = portfolio_df['total_value'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    # 计算收益率序列
    portfolio_df['returns'] = portfolio_df['total_value'].pct_change()
    returns = portfolio_df['returns'].dropna()
    
    # === 1. 收益性指标 ===
    print("\n🎯 收益性指标:")
    print(f"总收益率: {total_return:.2%}")
    print(f"绝对收益: ${final_value - initial_capital:,.2f}")
    
    # 年化收益率 (假设数据跨度为测试期间)
    trading_days = len(portfolio_df)
    years = trading_days / (365 * 24)  # 假设是小时数据
    annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    print(f"年化收益率: {annualized_return:.2%}")
    
    # === 2. 风险指标 ===
    print("\n🛡️ 风险指标:")
    volatility = returns.std() * np.sqrt(252 * 24)  # 年化波动率
    print(f"年化波动率: {volatility:.2%}")
    
    # 最大回撤
    rolling_max = portfolio_df['total_value'].expanding().max()
    drawdown = (portfolio_df['total_value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    print(f"最大回撤: {max_drawdown:.2%}")
    
    # 夏普比率
    risk_free_rate = 0.02  # 假设无风险利率2%
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0
    print(f"夏普比率: {sharpe_ratio:.3f}")
    
    # VaR (Value at Risk)
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    print(f"VaR (95%): {var_95:.4f}")
    print(f"VaR (99%): {var_99:.4f}")
    
    # === 3. 风险管理效果 ===
    print("\n🎯 风险管理效果:")
    
    # 风险信号分析
    risk_signals = signals_df['risk_signal'].value_counts()
    total_signals = len(signals_df)
    risk_events = risk_signals.get('STOP', 0)
    risk_ratio = risk_events / total_signals
    
    print(f"风险信号触发率: {risk_ratio:.2%}")
    print(f"风险事件次数: {risk_events}")
    print(f"风险信号分布: {risk_signals.to_dict()}")
    
    # 风险预警准确性
    if risk_events > 0:
        risk_probabilities = signals_df[signals_df['risk_signal'] == 'STOP']['risk_prob']
        avg_risk_prob = risk_probabilities.mean()
        min_risk_prob = risk_probabilities.min()
        max_risk_prob = risk_probabilities.max()
        
        print(f"平均风险概率: {avg_risk_prob:.3f}")
        print(f"风险概率范围: {min_risk_prob:.3f} - {max_risk_prob:.3f}")
    
    # === 4. 交易效率指标 ===
    print("\n📈 交易效率指标:")
    
    if len(trades_df) > 0:
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        print(f"总交易次数: {len(trades_df)}")
        print(f"买入次数: {len(buy_trades)}")
        print(f"卖出次数: {len(sell_trades)}")
        
        # 风险止损分析
        risk_stops = trades_df[trades_df['reason'].str.contains('风险止损', na=False)]
        print(f"风险止损次数: {len(risk_stops)}")
        
        if len(sell_trades) > 0:
            risk_stop_ratio = len(risk_stops) / len(sell_trades)
            print(f"风险止损比例: {risk_stop_ratio:.2%}")
    
    # === 5. 稳定性指标 ===
    print("\n🔄 稳定性指标:")
    
    # 收益率稳定性
    positive_returns = (returns > 0).sum()
    negative_returns = (returns < 0).sum()
    win_rate = positive_returns / len(returns) if len(returns) > 0 else 0
    
    print(f"正收益率天数: {positive_returns}")
    print(f"负收益率天数: {negative_returns}")
    print(f"胜率: {win_rate:.2%}")
    
    # 收益率偏度和峰度
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    print(f"收益率偏度: {skewness:.3f}")
    print(f"收益率峰度: {kurtosis:.3f}")
    
    # === 6. 基准比较 ===
    print("\n📊 基准比较:")
    
    # 计算买入持有策略的表现
    first_price = portfolio_df['price'].iloc[0]
    last_price = portfolio_df['price'].iloc[-1]
    buy_hold_return = (last_price - first_price) / first_price
    
    print(f"买入持有收益率: {buy_hold_return:.2%}")
    print(f"策略超额收益: {total_return - buy_hold_return:.2%}")
    
    # 信息比率
    tracking_error = returns.std() * np.sqrt(252 * 24)
    information_ratio = (annualized_return - buy_hold_return) / tracking_error if tracking_error > 0 else 0
    print(f"信息比率: {information_ratio:.3f}")
    
    # === 7. 综合评分 ===
    print("\n🏆 综合评分:")
    
    # 风险调整后收益评分 (0-100)
    risk_score = max(0, 100 - abs(max_drawdown * 1000))  # 回撤越小分数越高
    return_score = min(100, max(0, total_return * 1000))  # 收益率评分
    stability_score = max(0, 100 - volatility * 100)  # 稳定性评分
    
    comprehensive_score = (risk_score * 0.4 + return_score * 0.3 + stability_score * 0.3)
    
    print(f"风险控制评分: {risk_score:.1f}/100")
    print(f"收益能力评分: {return_score:.1f}/100")
    print(f"稳定性评分: {stability_score:.1f}/100")
    print(f"综合评分: {comprehensive_score:.1f}/100")
    
    # === 8. 模型等级评定 ===
    print("\n🎖️ 模型等级评定:")
    
    if comprehensive_score >= 90:
        grade = "A+ (优秀)"
    elif comprehensive_score >= 80:
        grade = "A (良好)"
    elif comprehensive_score >= 70:
        grade = "B+ (中等偏上)"
    elif comprehensive_score >= 60:
        grade = "B (中等)"
    elif comprehensive_score >= 50:
        grade = "C+ (中等偏下)"
    else:
        grade = "C (需要改进)"
    
    print(f"模型等级: {grade}")
    
    # === 9. 改进建议 ===
    print("\n💡 改进建议:")
    
    if total_return < 0.05:
        print("- 收益率偏低，建议优化交易策略以提高收益")
    
    if sharpe_ratio < 1.0:
        print("- 夏普比率偏低，建议平衡风险与收益")
    
    if risk_ratio > 0.3:
        print("- 风险信号过于频繁，可能需要调整风险阈值")
    elif risk_ratio < 0.1:
        print("- 风险信号较少，可能需要提高风险敏感度")
    
    if max_drawdown < -0.05:
        print("- 最大回撤较大，建议加强风险控制")
    else:
        print("- 风险控制表现优秀，回撤控制良好")
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'var_95': var_95,
        'var_99': var_99,
        'risk_ratio': risk_ratio,
        'win_rate': win_rate,
        'comprehensive_score': comprehensive_score,
        'grade': grade,
        'buy_hold_return': buy_hold_return,
        'information_ratio': information_ratio
    }

def create_performance_dashboard(portfolio_df, signals_df, metrics):
    """创建性能仪表板"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 收益率分布
    ax1 = axes[0, 0]
    returns = portfolio_df['total_value'].pct_change().dropna()
    ax1.hist(returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('收益率分布', fontsize=12, fontweight='bold')
    ax1.set_xlabel('收益率')
    ax1.set_ylabel('频数')
    ax1.grid(True, alpha=0.3)
    
    # 2. 累积收益
    ax2 = axes[0, 1]
    cumulative_returns = (1 + returns).cumprod() - 1
    ax2.plot(cumulative_returns.index, cumulative_returns, linewidth=2, color='green')
    ax2.set_title('累积收益率', fontsize=12, fontweight='bold')
    ax2.set_ylabel('累积收益率')
    ax2.grid(True, alpha=0.3)
    
    # 3. 回撤分析
    ax3 = axes[0, 2]
    rolling_max = portfolio_df['total_value'].expanding().max()
    drawdown = (portfolio_df['total_value'] - rolling_max) / rolling_max
    ax3.fill_between(portfolio_df.index, drawdown, 0, alpha=0.3, color='red')
    ax3.plot(portfolio_df.index, drawdown, color='red', linewidth=1)
    ax3.set_title('回撤分析', fontsize=12, fontweight='bold')
    ax3.set_ylabel('回撤比例')
    ax3.grid(True, alpha=0.3)
    
    # 4. 风险概率分布
    ax4 = axes[1, 0]
    risk_probs = signals_df['risk_prob']
    ax4.hist(risk_probs, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax4.axvline(x=0.5546, color='red', linestyle='--', label='风险阈值')
    ax4.set_title('风险概率分布', fontsize=12, fontweight='bold')
    ax4.set_xlabel('风险概率')
    ax4.set_ylabel('频数')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 性能指标雷达图
    ax5 = axes[1, 1]
    categories = ['收益能力', '风险控制', '稳定性', '交易效率', '综合表现']
    values = [
        min(100, metrics['total_return'] * 1000),
        max(0, 100 - abs(metrics['max_drawdown'] * 1000)),
        max(0, 100 - metrics['volatility'] * 100),
        80,  # 假设交易效率80分
        metrics['comprehensive_score']
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # 闭合图形
    angles += angles[:1]
    
    ax5.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax5.fill(angles, values, alpha=0.25, color='blue')
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories)
    ax5.set_ylim(0, 100)
    ax5.set_title('性能雷达图', fontsize=12, fontweight='bold')
    ax5.grid(True)
    
    # 6. 综合评分
    ax6 = axes[1, 2]
    score = metrics['comprehensive_score']
    colors = ['red' if score < 50 else 'orange' if score < 70 else 'green']
    ax6.bar(['综合评分'], [score], color=colors[0], alpha=0.7)
    ax6.set_ylim(0, 100)
    ax6.set_title(f'综合评分: {score:.1f}', fontsize=12, fontweight='bold')
    ax6.set_ylabel('分数')
    
    # 添加等级标注
    ax6.text(0, score + 5, metrics['grade'], ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 性能仪表板已保存为 'performance_dashboard.png'")

if __name__ == "__main__":
    # 读取回测结果
    try:
        portfolio_df = pd.read_csv('simple_portfolio_history.csv')
        signals_df = pd.read_csv('simple_signals_history.csv')
        trades_df = pd.read_csv('simple_trades_history.csv')
        
        # 转换时间戳
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # 设置索引
        portfolio_df.set_index('timestamp', inplace=True)
        signals_df.set_index('timestamp', inplace=True)
        
        # 计算综合指标
        metrics = calculate_comprehensive_metrics(portfolio_df, signals_df, trades_df)
        
        # 创建性能仪表板
        create_performance_dashboard(portfolio_df, signals_df, metrics)
        
    except FileNotFoundError:
        print("❌ 请先运行 simple_backtest.py 生成回测结果文件")
    except Exception as e:
        print(f"❌ 分析失败: {e}") 