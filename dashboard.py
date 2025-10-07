from tkinter import NO
import streamlit as st
import pandas as pd
import json
import os
import time
import glob
from pathlib import Path
import asyncio
import sys
import streamlit.components.v1 as components

# 在Windows上，为Playwright和Streamlit的兼容性设置事件循环策略
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import config # 导入根配置
from btc_predictor.utils import LOGGER
from execution_engine.okx_trader import OKXTrader
from btc_predictor.predict import get_live_trade_signal
from btc_predictor.data import get_data
from btc_predictor.config import DATA_CONFIG
from btc_predictor.kline_plot import create_kline_image


# --- 页面配置 ---
st.set_page_config(layout="wide", page_title="BTC智能决策看板")

# --- 辅助函数 ---
@st.cache_resource
def get_trader():
    return OKXTrader(demo_mode=config.DEMO_MODE)

@st.cache_data(ttl=60)
def get_okx_data():
    """缓存函数，用于获取OKX数据，避免触发频率限制。"""
    if config.API_KEYS.get('okx', {}).get('api_key') == 'YOUR_OKX_API_KEY':
        return None, None
    try:
        trader = get_trader()
        balance = trader.get_balance()
        positions = trader.get_positions()
        return balance, positions
    except Exception as e:
        st.error(f"加载OKX数据时出错: {e}")
        return None, None

def get_latest_kline_png_from_cache():
    import glob
    import os
    png_files = glob.glob(os.path.join("cache", "kline_chart_*.png"))
    if not png_files:
        return None, None
    latest_file = max(png_files, key=os.path.getctime)
    time_range = os.path.splitext(os.path.basename(latest_file))[0].replace("kline_chart_", "")
    return latest_file, time_range

def get_latest_kline_html_from_cache():
    html_files = glob.glob(os.path.join("cache", "*.html"))
    if not html_files:
        return None, None
    latest_file = max(html_files, key=os.path.getmtime)
    time_range = os.path.splitext(os.path.basename(latest_file))[0].replace("kline_chart_", "")
    return latest_file, time_range

@st.cache_data(ttl=300)  # 缓存5分钟
def get_latest_kline_data() -> dict:
    """获取最新的K线数据和图表"""
    # 优先从缓存图片获取
    image_path, data_time_range = get_latest_kline_png_from_cache()
    if image_path and os.path.exists(image_path):
        bars = 48
        if data_time_range:
            try:
                bars = int(data_time_range.split('_')[-1].replace('bars',''))
            except Exception:
                bars = 48
        # 自动补全价格数据
        current_price = price_change = price_change_pct = None
        price_data = None
        try:
            price_data = get_data(symbol=DATA_CONFIG['symbol'], timeframe=DATA_CONFIG['timeframe'])
            if price_data is not None and not price_data.empty:
                price_data = price_data.tail(48)
                current_price = float(price_data['close'].iloc[-1])
                price_change = float(price_data['close'].iloc[-1] - price_data['close'].iloc[-2])
                price_change_pct = float((price_data['close'].iloc[-1] - price_data['close'].iloc[-2]) / price_data['close'].iloc[-2] * 100)
        except Exception as e:
            LOGGER.error(f"补全价格数据失败: {e}")
        return {
            'price_data': price_data,
            'image_path': image_path,
            'data_time_range': data_time_range,
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'bars': bars
        }
    # 如果没有缓存图片，才尝试生成
    try:
        # 获取价格数据
        full_price_data = get_data(
            symbol=DATA_CONFIG['symbol'], 
            timeframe=DATA_CONFIG['timeframe']
        )
        if full_price_data is not None and not full_price_data.empty:
            price_data_for_ma = full_price_data.tail(200)
            kline_result = create_kline_image(price_data_for_ma, save_dir="cache")
            if kline_result:
                kline_image_path, data_time_range = kline_result
                price_data = price_data_for_ma.tail(48)
                return {
                    'price_data': price_data,
                    'image_path': kline_image_path,
                    'data_time_range': data_time_range,
                    'current_price': float(price_data['close'].iloc[-1]),
                    'price_change': float(price_data['close'].iloc[-1] - price_data['close'].iloc[-2]),
                    'price_change_pct': float((price_data['close'].iloc[-1] - price_data['close'].iloc[-2]) / price_data['close'].iloc[-2] * 100),
                    'bars': len(price_data)
                }
    except Exception as e:
        LOGGER.error(f"获取K线数据失败: {e}")
        return {'price_data': None, 'image_path': None, 'data_time_range': None, 'current_price': None, 'price_change': None, 'price_change_pct': None, 'bars': 48}
    # 如果所有分支都未返回，补一个兜底dict
    return {'price_data': None, 'image_path': None, 'data_time_range': None, 'current_price': None, 'price_change': None, 'price_change_pct': None, 'bars': 48}

@st.cache_data(ttl=300)  # 缓存5分钟
def get_quant_signal():
    """获取量化模型信号"""
    try:
        signal_data = get_live_trade_signal(model_name=config.DEFAULTS['model_name'])
        return signal_data
    except Exception as e:
        LOGGER.error(f"获取量化信号失败: {e}")
        return None

def get_latest_vlm_analysis():
    """获取最新的VLM分析结果（从缓存或决策报告）"""
    try:
        # 首先尝试从决策报告获取
        report = load_report()
        if report and report.get('kline_analysis'):
            return report.get('kline_analysis')
        
        # 如果没有，尝试从VLM缓存文件获取最新的分析
        cache_file = Path("cache/vlm_kline_cache.json")
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            if cache_data:
                # 获取最新的缓存条目
                latest_entry = max(cache_data.values(), key=lambda x: x.get('timestamp', ''))
                return latest_entry.get('analysis')
        
        return None
    except Exception as e:
        LOGGER.error(f"获取VLM分析失败: {e}")
        return None

def load_report():
    """加载最新的决策报告。"""
    report_file = "decision_report.json"
    if not os.path.exists(report_file):
        return None
    with open(report_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_account_status():
    """从OKX获取实时账户状态。"""
    try:
        trader = get_trader()
        
        balance_details = trader.get_balance() # 使用新的 get_balance 方法
        positions = trader.get_positions()
        
        position_status = "无持仓"
        pnl = "N/A"
        
        if positions:
            pos = positions[0]
            pos_side_from_api = pos.get('posSide')
            position_size = float(pos.get('pos', '0'))

            if pos_side_from_api == 'long':
                pos_side_display = "多头"
            elif pos_side_from_api == 'short':
                pos_side_display = "空头"
            elif pos_side_from_api == 'net':
                pos_side_display = "多头" if position_size > 0 else "空头"
            else:
                pos_side_display = "未知" # 不应该发生

            pos_size = pos.get('pos', '0')
            entry_price = pos.get('avgPx', '0')
            upl = float(pos.get('upl', 0))
            position_status = f"{pos_side_display} {pos_size} @ {entry_price}"
            pnl = f"{upl:.2f} USDT"
            
        return {
            "balance": f"{balance_details:.2f} USDT" if balance_details is not None else "获取失败",
            "position": position_status,
            "pnl": pnl
        }
    except ImportError:
        return {"balance": "配置模块加载失败", "position": "N/A", "pnl": "N/A"}
    except Exception as e:
        LOGGER.error(f"从看板连接OKX失败: {e}")
        return {"balance": "连接失败", "position": str(e), "pnl": "N/A"}

# --- 主看板 ---
st.title("📊 BTC智能决策看板")
st.markdown("""
本看板旨在提供一个清晰、全面的视角，用以监控和分析BTC量化交易系统的各项关键指标。
""")
st.markdown("---")

# 实时账户状态
with st.container():
    st.subheader("📡 实时账户状态")
    status_cols = st.columns(3)
    account_status = get_account_status()
    status_cols[0].metric("总权益", account_status.get("balance"))
    status_cols[1].metric("当前持仓", account_status.get("position"))
    status_cols[2].metric("未实现盈亏", account_status.get("pnl"))

st.markdown("---")

# K线图表和技术分析
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 实时K线图表")
    # 只展示最新图片
    image_path, image_time_range = get_latest_kline_png_from_cache()
    if image_path and os.path.exists(image_path):
        st.image(image_path, caption=f"BTC/USDT K线图（{image_time_range}）", use_container_width=True)
        st.caption(f"时间范围: {image_time_range}")
    else:
        st.warning("K线图文件未找到")
            
    # 显示最新的价格数据表格
    with st.expander("📊 最新价格数据 (最后10根K线)"):
        latest_data = get_latest_kline_data()['price_data'].tail(10).copy()
        latest_data.index = latest_data.index.strftime("%m-%d %H:%M")
        st.dataframe(latest_data[['open', 'high', 'low', 'close', 'volume']], use_container_width=True)


with col2:
    st.subheader("🤖 VLM技术分析")
    vlm_analysis = get_latest_vlm_analysis()
    
    if vlm_analysis:
        st.success("✅ VLM分析可用")
        with st.expander("🔍 查看完整VLM分析", expanded=False):
            st.markdown(vlm_analysis)
    else:
        st.warning("⏳ 暂无VLM分析结果")
        st.info("VLM分析结果将在下次运行主程序后显示")

st.markdown("---")

# 量化模型预测结果
st.subheader("🧠 量化模型预测")
quant_signal = get_quant_signal()

if quant_signal:
    signal_cols = st.columns(4)
    
    # 信号类型
    signal_type = quant_signal.get('signal', 'HOLD')
    if signal_type == 'BUY':
        signal_cols[0].success(f"📈 {signal_type}")
    elif signal_type == 'SELL':
        signal_cols[0].error(f"📉 {signal_type}")
    else:
        signal_cols[0].info(f"⏸️ {signal_type}")
    
    # 预测回报率
    predicted_return = quant_signal.get('predicted_return', 0.0)
    signal_cols[1].metric("预测回报率", f"{predicted_return:.4f}%")
    
    # 模型名称
    signal_cols[2].metric("模型", config.DEFAULTS['model_name'])
    
    # 信号强度指示器（用预测幅度近似展示）
    strength = "🟢" if abs(predicted_return) > 0.5 else "🟡" if abs(predicted_return) > 0.1 else "🔴"
    signal_cols[3].metric("信号强度", strength)
    
    # 详细信息
    with st.expander("📋 模型分析详情"):
        st.markdown("**模型输出信息:**")
        st.text(quant_signal.get('info', '无详细信息'))
        
        if 'features' in quant_signal:
            st.markdown("**特征数据:**")
            st.json(quant_signal['features'])
else:
    st.error("无法获取量化模型信号")

st.markdown("---")

# --- 决策报告 ---
st.subheader("📝 最新决策报告")
report = load_report()
if report:
    st.markdown(f"**报告生成时间:** `{report.get('timestamp')}`")
    
    st.subheader("🧠 最终决策与分析")
    decision = report.get('decision', 'N/A')
    reasoning = report.get('reasoning', '无分析。')
    key_signals = report.get('key_signals_detected', '无关键风险信号')

    if "BUY" in decision:
        st.success(f"#### 决策: {decision}")
    elif "SELL" in decision:
        st.error(f"#### 决策: {decision}")
    else:
        st.info(f"#### 决策: {decision}")
    
    # 显示关键信号检测结果
    if key_signals != '无关键风险信号':
        st.warning(f"🚨 **关键信号检测**: {key_signals}")
    else:
        st.success(f"✅ **关键信号检测**: {key_signals}")
    
    st.markdown("**核心分析理由:**")
    st.write(reasoning)
    
    # 显示建议仓位和风险评估
    col1, col2 = st.columns(2)
    with col1:
        trade_size = report.get('suggested_trade_size', 0.0)
        col1.metric("建议仓位", f"{trade_size:.1%}")
    
    risk_assessment = report.get('risk_assessment', '无风险评估')
    if risk_assessment != '无风险评估':
        st.markdown("**风险评估:**")
        st.info(risk_assessment)
    
    with st.expander("点击查看决策依据详情"):
        st.subheader("DeepSeek决策分析详情:")
        
        # 显示所有决策字段
        col1, col2 = st.columns(2)
        with col1:
            st.metric("决策", decision)
        with col2:
            st.metric("建议仓位", f"{trade_size:.1%}")
            if key_signals != '无关键风险信号':
                st.error(f"关键信号: {key_signals}")
            else:
                st.success("关键信号: 无")
        
        st.subheader("完整推理过程:")
        st.text(reasoning)

        st.subheader("决策时参考的市场情绪 (推文):")
        tweets = report.get('tweets', [])
        if tweets:
            for tweet in tweets:
                user_info = tweet.get('user', {})
                username = user_info.get('username') or user_info.get('screen_name', '未知用户')
                text = tweet.get('text', '无内容')
                created_at = tweet.get('creation_date') or tweet.get('created_at', '未知时间')
                source = tweet.get('source', '')
                
                # 根据来源添加标签
                if 'Influencer' in source:
                    st.markdown(f"🌟 **@{username}** ({created_at}): {text}")
                elif 'Macro Event' in source:
                    st.markdown(f"📊 **@{username}** ({created_at}): {text}")
                else:
                    st.markdown(f"💬 **@{username}** ({created_at}): {text}")
        else:
            st.markdown("无有效的推文数据。")

else:
    st.warning("决策战报文件 (decision_report.json) 未找到。请先运行一次 `main.py` 来生成报告。")

# 缓存状态信息
st.markdown("---")
st.subheader("💾 系统状态")
status_info_cols = st.columns(3)

# VLM缓存状态
try:
    cache_file = Path("cache/vlm_kline_cache.json")
    kline_cache_count = 0
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            kline_cache_count = len(cache_data)
    
    tweet_cache_file = Path("cache/vlm_tweet_cache.json")
    tweet_cache_count = 0
    if tweet_cache_file.exists():
        with open(tweet_cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            tweet_cache_count = len(cache_data)
    
    status_info_cols[0].metric("VLM缓存", f"K线图: {kline_cache_count}, 推文: {tweet_cache_count}")
except:
    status_info_cols[0].metric("VLM缓存", "读取失败")

# 最后运行时间
try:
    with open("last_run.json", 'r') as f:
        last_run_data = json.load(f)
        last_run = last_run_data.get('last_run_utc', '未知')
    status_info_cols[1].metric("最后运行", last_run)
except:
    status_info_cols[1].metric("最后运行", "未知")

# 模式指示器
mode = "🔴 实盘模式" if not config.DEMO_MODE else "🟡 模拟盘模式"
status_info_cols[2].metric("运行模式", mode)

# --- 自动刷新机制 ---
st.markdown("---")
st.caption("⏱️ 页面每30秒自动刷新")

# 添加手动刷新按钮
if st.button("🔄 立即刷新", use_container_width=True):
    st.rerun()

time.sleep(30)
st.rerun() 