import time
import json
import os
from typing import Dict, Any, List, cast
import argparse
from datetime import datetime, timezone
import schedule
from functools import partial

import config
from btc_predictor.predict import get_live_trade_signal
from btc_predictor.utils import LOGGER
from btc_predictor.kline_plot import create_kline_image
from data_ingestion.news_feeds import fetch_coindesk_news
from decision_engine.vlm_analyzer import VLMAnalyzer
from decision_engine.deepseek_analyzer import DeepSeekAnalyzer
from execution_engine.okx_trader import OKXTrader
from utils.email_notifier import EmailNotifier

def save_decision_report(report: Dict[str, Any]):
    """将决策报告保存到文件。"""
    path = "decision_report.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    LOGGER.info(f"决策战报已保存到: {path}")

def print_decision_report(report: Dict[str, Any]):
    """在终端用美观的格式打印决策报告。"""
    print("\n" + "="*80)
    print("           📊 BTC 期货智能决策系统 - 最终决策报告")
    print("="*80)
    
    decision = report.get("decision", "N/A").upper()
    color_map = {
        "LONG": "\033[92m",   # 绿色
        "SHORT": "\033[91m",  # 红色
        "CLOSE_LONG": "\033[93m", # 黄色
        "CLOSE_SHORT": "\033[93m",# 黄色
        "HOLD": "\033[94m"    # 蓝色
    }
    color = color_map.get(decision, "\033[0m") # 默认无颜色
    print(f"  - 最终决策: {color}{decision}\033[0m")
        
    print(f"  - 置信度: {report.get('confidence', 'N/A')}")
    trade_params = report.get('trade_params')
    if trade_params:
        print("  - 交易参数:")
        print(f"    - 杠杆: {trade_params.get('leverage', 'N/A')}x")
        tp_pct = trade_params.get('take_profit_pct')
        sl_pct = trade_params.get('stop_loss_pct')
        print(f"    - 止盈: {tp_pct}%" if tp_pct is not None else "    - 止盈: N/A")
        print(f"    - 止损: {sl_pct}%" if sl_pct is not None else "    - 止损: N/A")

    print("-" * 80)
    print("  - 决策理由:")
    # 将长文本自动换行
    reasoning = report.get('reasoning', 'N/A')
    import textwrap
    wrapped_reasoning = "\n".join(["    " + line for line in textwrap.wrap(reasoning, width=100)])
    print(wrapped_reasoning)
    
    print("-" * 80)
    print("  - 关键信号:")
    print(f"    {report.get('key_signals_detected', 'N/A')}")
    print("-" * 80)
    print("  - 风险评估:")
    print(f"    {report.get('risk_assessment', 'N/A')}")
    print("="*80 + "\n")

def _get_last_run_timestamp() -> datetime | None:
    """从文件检索上次运行的UTC时间戳。"""
    if not os.path.exists("last_run.json"):
        return None
    with open("last_run.json", "r") as f:
        try:
            data = json.load(f)
            return datetime.fromisoformat(data['last_run_utc'])
        except (json.JSONDecodeError, KeyError):
            return None

def _save_last_run_timestamp():
    """将当前的UTC时间戳保存到文件。"""
    now_utc = datetime.now(timezone.utc)
    with open("last_run.json", "w") as f:
        json.dump({'last_run_utc': now_utc.isoformat()}, f)

def get_market_intelligence() -> List[Dict[str, Any]]:
    """
    从CoinDesk获取市场情报。
    """
    LOGGER.info("开始获取CoinDesk市场新闻情报...")
    
    news_items = fetch_coindesk_news(limit=cast(int, config.SOCIAL_MEDIA.get('news_limit', 15)))
    
    if not news_items:
        LOGGER.warning("未能从CoinDesk获取到新闻情报。")
        return []

    LOGGER.success(f"情报整合完毕，共获取 {len(news_items)} 条新闻。")
    return news_items

def _generate_and_analyze_kline(vlm_analyzer, price_data, timeframe_alias, timeframe=None):
    """
    辅助函数：为给定数据生成、保存并分析K线图。
    
    Args:
        vlm_analyzer: VLM分析器实例
        price_data: 价格数据
        timeframe_alias: 时间周期别名（用于日志）
        timeframe: 实际时间周期（如'1h', '1d', '1w'）
    """
    if price_data is None or price_data.empty:
        LOGGER.warning(f"没有价格数据可用于生成 {timeframe_alias} K线图。")
        return None, None

    LOGGER.info(f"正在为 {timeframe_alias} 生成K线图...")
    kline_result = create_kline_image(price_data) # 移除 timeframe_alias
    if not kline_result:
        return None, None
        
    kline_image_path, data_time_range = kline_result
    analysis = vlm_analyzer.analyze_kline_chart(kline_image_path, data_time_range, timeframe)
    LOGGER.info(f"{timeframe_alias} K线图VLM分析结果: {analysis}")
    return analysis, data_time_range

def run_trading_cycle(skip_llm: bool = False):
    """
    运行一个完整的交易决策周期。
    """
    LOGGER.info("========== 开始新一轮决策周期 ==========")
    
    # 初始化邮件通知器
    email_notifier = EmailNotifier()
    
    # 初始化流程状态跟踪器
    process_status = {}
    
    def track_process(process_name: str, func, *args, **kwargs):
        """跟踪流程执行状态"""
        import time
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = f"{time.time() - start_time:.1f}s"
            process_status[process_name] = {
                'status': 'success',
                'duration': duration,
                'message': '执行成功'
            }
            return result
        except Exception as e:
            duration = f"{time.time() - start_time:.1f}s"
            process_status[process_name] = {
                'status': 'error',
                'duration': duration,
                'message': '执行失败',
                'error': str(e)
            }
            raise
    
    try:
        # ======================================================================
        # 步骤 1: 获取多时间框架数据和信号
        # ======================================================================
        LOGGER.info("="*50 + "\n步骤 1: 获取多时间框架数据和信号")
        
        def collect_data():
            from btc_predictor.data import get_data
            from btc_predictor.config import DATA_CONFIG
            
            # 获取主要时间框架数据 (例如 1h)
            short_term_data = get_data(
                symbol=cast(str, DATA_CONFIG['symbol']), 
                timeframe=cast(str, DATA_CONFIG['timeframe'])
            )
            # 获取日线数据
            daily_data = get_data(symbol=cast(str, DATA_CONFIG['symbol']), timeframe='1d')
            # 获取周线数据
            weekly_data = get_data(symbol=cast(str, DATA_CONFIG['symbol']), timeframe='1w')

            price_data_for_ma = short_term_data.tail(200) if short_term_data is not None and not short_term_data.empty else None

            quant_signal_data = get_live_trade_signal(
                model_name=cast(str, config.DEFAULTS['model_name'])
            )
            if quant_signal_data is None:
                LOGGER.error("无法获取量化信号，将使用默认的HOLD信号继续。")
                quant_signal_data = {
                    "signal": "HOLD",
                    "predicted_return": 0.0,
                    "info": "信号获取失败"
                }
            LOGGER.info(f"获取到量化模型信号: {quant_signal_data}")
            
            return short_term_data, daily_data, weekly_data, price_data_for_ma, quant_signal_data
        
        short_term_data, daily_data, weekly_data, price_data_for_ma, quant_signal_data = track_process('data_collection', collect_data)

        # ======================================================================
        # 步骤 2: 初始化VLM分析器（仅用于K线图）
        # ======================================================================
        LOGGER.info("="*50 + "\n步骤 2: 初始化VLM分析器")
        vlm_analyzer = VLMAnalyzer()
        vlm_analyzer.cache.cleanup_expired_cache()
        cache_stats = vlm_analyzer.cache.get_cache_stats()
        LOGGER.info(f"当前VLM缓存状态 - K线图: {cache_stats.get('kline_cache_count', 0)} 条")

        # ======================================================================
        # 步骤 3: 生成并分析多时间框架的K线图
        # ======================================================================
        LOGGER.info("="*50 + "\n步骤 3: 生成并分析多时间框架的K线图")
        
        def perform_vlm_analysis():
            short_term_analysis, _ = _generate_and_analyze_kline(vlm_analyzer, price_data_for_ma, "Short-Term", "1h")
            daily_analysis, _ = _generate_and_analyze_kline(vlm_analyzer, daily_data.tail(200) if daily_data is not None else None, "Daily", "1d")
            weekly_analysis, _ = _generate_and_analyze_kline(vlm_analyzer, weekly_data.tail(100) if weekly_data is not None else None, "Weekly", "1w")
            return short_term_analysis, daily_analysis, weekly_analysis
        
        short_term_analysis, daily_analysis, weekly_analysis = track_process('vlm_analysis', perform_vlm_analysis)

        # ======================================================================
        # 步骤 4: 获取市场新闻情报
        # ======================================================================
        LOGGER.info("="*50 + "\n步骤 4: 获取市场新闻情报")
        market_news = track_process('news_intelligence', get_market_intelligence)

        # ======================================================================
        # 步骤 5: LLM决策引擎
        # ======================================================================
        LOGGER.info("="*50 + "\n步骤 5: 获取持仓并进行LLM决策")
        
        # 在决策前初始化交易器并获取当前仓位和余额
        trader = OKXTrader(demo_mode=config.DEMO_MODE)
        current_position = trader.get_position()
        current_balance = trader.get_balance('USDT')
        
        if skip_llm:
            LOGGER.warning("已设置--skip-llm，跳过LLM决策分析。")
            final_decision = {
                "decision": "HOLD",
                "reasoning": "Skipped LLM analysis",
                "trade_params": {},
                "suggested_trade_size": 0.95
            }
            process_status['llm_decision'] = {
                'status': 'info',
                'duration': '0.0s',
                'message': '跳过LLM分析'
            }
        else:
            def perform_llm_decision():
                analyzer = DeepSeekAnalyzer()
                return analyzer.get_trade_decision(
                    quant_signal=quant_signal_data,
                    twitter_data=market_news, 
                    kline_analysis={
                        "short_term": short_term_analysis,
                        "daily": daily_analysis,
                        "weekly": weekly_analysis
                    },
                    current_position=current_position,
                    current_balance=current_balance
                )
            
            final_decision = track_process('llm_decision', perform_llm_decision)

        # ======================================================================
        # 步骤 6: 保存并打印决策报告
        # ======================================================================
        LOGGER.info("="*50 + "\n步骤 6: 保存并打印决策报告")
        save_decision_report(final_decision)
        print_decision_report(final_decision)

        # ======================================================================
        # 步骤 7: 执行交易
        # ======================================================================
        LOGGER.info("="*50 + "\n步骤 7: 执行交易")
        # 深度类型校验与修正，防止类型污染
        if not isinstance(final_decision, dict):
            LOGGER.error("final_decision 不是字典，实际类型: {}，内容: {}", type(final_decision), final_decision)
            return
        # 强制类型修正
        final_decision["decision"] = str(final_decision.get("decision", "HOLD"))
        trade_params = final_decision.get("trade_params", {})
        if not isinstance(trade_params, dict):
            LOGGER.warning(f"trade_params 字段类型异常，已重置为空字典。实际值: {trade_params}")
            trade_params = {}
        final_decision["trade_params"] = trade_params
        suggested_trade_size = final_decision.get("suggested_trade_size", 0.95)
        try:
            final_decision["suggested_trade_size"] = float(suggested_trade_size)
        except Exception:
            LOGGER.warning(f"suggested_trade_size 字段类型异常，已重置为0.95。实际值: {suggested_trade_size}")
            final_decision["suggested_trade_size"] = 0.95
        # 其余字段可按需补全
        required_keys = ["decision", "trade_params", "suggested_trade_size"]
        for k in required_keys:
            if k not in final_decision:
                LOGGER.warning(f"final_decision 缺少关键字段: {k}，将使用默认值。")
                if k == "decision":
                    final_decision["decision"] = str("HOLD")
                elif k == "trade_params":
                    final_decision["trade_params"] = dict()
                elif k == "suggested_trade_size":
                    final_decision["suggested_trade_size"] = float(0.95)
        LOGGER.info(f"最终用于交易执行的决策数据: {final_decision}")
        
                # 执行交易决策
        try:
            def execute_trade():
                trader.execute_decision(final_decision)
            
            track_process('trade_execution', execute_trade)
            # 发送成功通知邮件
            email_notifier.send_decision_notification(final_decision, execution_success=True, process_status=process_status)
        except Exception as e:
            error_msg = f"交易执行失败: {str(e)}"
            LOGGER.error(error_msg)
            # 发送失败通知邮件
            email_notifier.send_decision_notification(final_decision, execution_success=False, error_msg=error_msg, process_status=process_status)
            # 同时发送错误通知邮件
            email_notifier.send_error_notification(
                "交易执行错误", 
                error_msg, 
                context={
                    "decision": final_decision.get("decision"),
                    "confidence": final_decision.get("confidence"),
                    "trade_params": str(final_decision.get("trade_params"))
                }
            )

    except Exception as e:
        import traceback
        error_msg = f"交易周期主循环发生严重错误: {repr(e)}"
        LOGGER.critical(f"{error_msg}\n详细traceback:\n{traceback.format_exc()}", exc_info=True)
        
        # 发送系统错误通知邮件
        email_notifier.send_error_notification(
            "系统错误", 
            error_msg, 
            context={
                "traceback": traceback.format_exc()[:500] + "..." if len(traceback.format_exc()) > 500 else traceback.format_exc()
            }
        )
    finally:
        LOGGER.info("========== 本轮决策周期结束 ==========\n")
        _save_last_run_timestamp()

def main():
    """主函数，用于设置和运行调度任务。"""
    parser = argparse.ArgumentParser(description="BTC智能决策系统主控程序。")
    parser.add_argument('--now', action='store_true', help='立即运行一次决策周期并退出。')
    parser.add_argument('--skip-llm', action='store_true', help='跳过LLM的API调用，用于调试。')
    args = parser.parse_args()

    job = partial(run_trading_cycle, skip_llm=args.skip_llm)

    if args.now:
        LOGGER.info("接收到 --now 参数，立即执行一次决策周期...")
        job()
        LOGGER.info("决策周期执行完毕，程序退出。")
        return

    LOGGER.info("BTC智能决策系统 v5.0 主控程序已启动（调度模式）。")
    
    # 只用schedule的每小时整点调度
    schedule.every().hour.at(":00").do(job)

    last_run_utc = _get_last_run_timestamp()
    if not last_run_utc or (datetime.now(timezone.utc) - last_run_utc).total_seconds() > 3600:
        LOGGER.info("首次运行或检测到错过的计划任务，立即执行一次决策周期...")
        job()

    LOGGER.info("系统正在等待下一个调度时间点... (按 Ctrl+C 退出)")
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main() 