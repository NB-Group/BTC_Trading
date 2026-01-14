import time
import json
import os
from typing import Dict, Any, List, cast
import argparse
from datetime import datetime, timezone
import schedule
from functools import partial
from tenacity import RetryError

import config
from btc_predictor.predict import get_live_trade_signal, get_rf4_signal, get_bollinger_breakout_signal, get_ma_crossover_signal
from btc_predictor.utils import LOGGER
from btc_predictor.kline_plot import create_kline_image
from data_ingestion.news_feeds import fetch_coindesk_news, fetch_truthsocial_news
from decision_engine.vlm_analyzer import VLMAnalyzer
from decision_engine.deepseek_analyzer import DeepSeekAnalyzer
from decision_engine.unified_analyzer import UnifiedGeminiAnalyzer
from decision_engine.gpt_reviewer import GPTReviewer
from execution_engine.okx_trader import OKXTrader
from utils.email_notifier import EmailNotifier
from market_scanner import scan_for_opportunities # 导入市场扫描器
from utils.autostart import ensure_windows_autostart

def save_decision_report(report: Dict[str, Any]):
    """将决策报告保存到文件。"""
    path = "decision_report.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    LOGGER.info(f"决策战报已保存到: {path}")

def print_decision_report(report: Dict[str, Any]):
    """在终端用美观的格式打印决策报告。"""
    print("\n" + "\033[38;5;102m" + "="*80 + "\033[0m")
    print("\033[38;5;180m           📊 BTC 期货智能决策系统 - 最终决策报告\033[0m")
    print("\033[38;5;102m" + "="*80 + "\033[0m")
    
    decision = report.get("decision", "N/A").upper()
    color_map = {
        "LONG": "\033[38;5;143m",      # 莫兰迪暖灰色
        "SHORT": "\033[38;5;174m",     # 莫兰迪粉橙色
        "CLOSE_LONG": "\033[38;5;180m", # 莫兰迪橄榄色
        "CLOSE_SHORT": "\033[38;5;180m",# 莫兰迪橄榄色
        "HOLD": "\033[38;5;152m"       # 莫兰迪薄荷绿
    }
    color = color_map.get(decision, "\033[0m") # 默认无颜色
    print(f"\033[38;5;109m  - 最终决策: {color}{decision}\033[0m")
        
    # 置信度字段已移除
    trade_params = report.get('trade_params')
    if trade_params:
        print("\033[38;5;109m  - 交易参数:\033[0m")
        print(f"\033[38;5;102m    - 杠杆: {trade_params.get('leverage', 'N/A')}x\033[0m")
        tp_pct = trade_params.get('take_profit_pct')
        sl_pct = trade_params.get('stop_loss_pct')
        print(f"\033[38;5;102m    - 止盈: {tp_pct}%\033[0m" if tp_pct is not None else "\033[38;5;102m    - 止盈: N/A\033[0m")
        print(f"\033[38;5;102m    - 止损: {sl_pct}%\033[0m" if sl_pct is not None else "\033[38;5;102m    - 止损: N/A\033[0m")

    print("\033[38;5;102m" + "-" * 80 + "\033[0m")
    print("\033[38;5;109m  - 决策理由:\033[0m")
    # 将长文本自动换行
    reasoning = report.get('reasoning', 'N/A')
    import textwrap
    wrapped_reasoning = "\n".join(["\033[38;5;145m    " + line + "\033[0m" for line in textwrap.wrap(reasoning, width=100)])
    print(wrapped_reasoning)
    
    # 新增打印交易对信息
    symbol = report.get("symbol", "N/A")
    print("\033[38;5;102m" + "-" * 80 + "\033[0m")
    print(f"\033[38;5;109m  - 交易对: {symbol}\033[0m")
    
    print("\033[38;5;102m" + "-" * 80 + "\033[0m")
    print("\033[38;5;109m  - 关键信号:\033[0m")
    print(f"\033[38;5;145m    {report.get('key_signals_detected', 'N/A')}\033[0m")
    print("\033[38;5;102m" + "-" * 80 + "\033[0m")
    print("\033[38;5;109m  - 风险评估:\033[0m")
    print(f"\033[38;5;145m    {report.get('risk_assessment', 'N/A')}\033[0m")

    # 若存在原始模型输出（通常是JSON解析失败时的诊断信息），完整输出以便排查
    raw_text = report.get('raw_response_text')
    parse_error = report.get('parse_error')
    if raw_text is not None or parse_error is not None:
        print("\033[38;5;102m" + "-" * 80 + "\033[0m")
        print("\033[38;5;174m  - 诊断: JSON解析失败，原始模型输出如下\033[0m")
        if parse_error:
            print(f"\033[38;5;145m    错误: {parse_error}\033[0m")
        if raw_text is not None:
            print("\033[38;5;102m" + "-" * 80 + "\033[0m")
            print("\033[38;5;109m  - 原始模型输出:\033[0m")
            print(raw_text)
    print("\033[38;5;102m" + "="*80 + "\033[0m" + "\n")

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

def get_market_intelligence(symbol: str) -> List[Dict[str, Any]]:
    """
    从CoinDesk和TruthSocial获取特定币种的市场情报。
    """
    all_news_items = []
    
    # 1. 获取CoinDesk新闻
    LOGGER.info(f"开始为 {symbol} 获取CoinDesk市场新闻情报...")
    coindesk_items = fetch_coindesk_news(symbol=symbol, limit=cast(int, config.SOCIAL_MEDIA.get('news_limit', 15)))
    if coindesk_items:
        all_news_items.extend(coindesk_items)
        LOGGER.success(f"从CoinDesk获取到 {len(coindesk_items)} 条新闻。")
    else:
        LOGGER.warning(f"未能从CoinDesk获取到 {symbol} 的新闻情报。")
    
    # 2. 获取TruthSocial帖子（特别关注特朗普等关键账号）
    LOGGER.info(f"开始为 {symbol} 获取TruthSocial帖子情报...")
    try:
        # 从配置中获取要监控的TruthSocial账号，默认为特朗普
        truthsocial_accounts = config.SOCIAL_MEDIA.get('truthsocial_accounts', ['realDonaldTrump'])
        truthsocial_limit = cast(int, config.SOCIAL_MEDIA.get('truthsocial_limit', 10))
        truthsocial_items = fetch_truthsocial_news(accounts=truthsocial_accounts, limit=truthsocial_limit)
        if truthsocial_items:
            all_news_items.extend(truthsocial_items)
            LOGGER.success(f"从TruthSocial获取到 {len(truthsocial_items)} 条帖子。")
        else:
            LOGGER.warning(f"未能从TruthSocial获取到帖子情报。")
    except Exception as e:
        LOGGER.error(f"获取TruthSocial帖子时发生错误: {e}")
    
    if not all_news_items:
        LOGGER.warning(f"未能获取到任何关于 {symbol} 的新闻情报。")
        return []
    
    LOGGER.success(f"情报整合完毕，共获取 {len(all_news_items)} 条关于 {symbol} 的新闻和帖子。")
    return all_news_items

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

    # 从价格数据中推断 symbol
    symbol_from_data = price_data.attrs.get('symbol', 'UNKNOWN')

    LOGGER.info(f"正在为 {symbol_from_data} 的 {timeframe_alias} 生成K线图...")
    # [修复] 移除多余的 'symbol' 参数，因为函数现在会从df.attrs中读取
    kline_result = create_kline_image(price_data, timeframe=timeframe or '1h')
    if not kline_result:
        return None, None
        
    kline_image_path, data_time_range = kline_result
    analysis = vlm_analyzer.analyze_kline_chart(kline_image_path, data_time_range, symbol_from_data, timeframe, price_data)
    LOGGER.info(f"{symbol_from_data} 的 {timeframe_alias} K线图VLM分析结果: {analysis}")
    return analysis, data_time_range

def analyze_and_trade_symbol(symbol: str, trader: OKXTrader, email_notifier: EmailNotifier, skip_llm: bool = False, is_primary_symbol: bool = False):
    """
    对单个币种执行完整的分析和交易流程。
    这是一个可重用的函数，包含了从数据收集到交易执行的完整逻辑。
    """
    print("\033[38;5;180m" + f"========== 开始处理币种: {symbol} ==========" + "\033[0m")
    LOGGER.info(f"========== 开始处理币种: {symbol} ==========")
    
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
        print(f"\033[38;5;152m[{symbol}] 步骤 1: 获取多时间框架数据和信号\033[0m")
        LOGGER.info(f"[{symbol}] 步骤 1: 获取多时间框架数据和信号")
        
        def collect_data(current_symbol):
            from btc_predictor.data import get_data
            
            # OKX symbol format for ccxt: 'BTC/USDT'
            ccxt_symbol = current_symbol.replace('-SWAP', '').replace('-', '/')

            # 获取主要时间框架数据 (1h) 约5天
            limit_5_days_hourly = 5 * 24
            short_term_data = get_data(symbol=ccxt_symbol, timeframe='1h', limit=limit_5_days_hourly)

            price_data_for_ma = short_term_data.tail(limit_5_days_hourly) if short_term_data is not None and not short_term_data.empty else None

            # --- 获取多个量化策略信号 ---
            quant_signals = []

            # 1. 获取RF4背离策略信号
            rf4_signal_data = get_rf4_signal(symbol=ccxt_symbol, period=15, order=5)
            if rf4_signal_data:
                # 转换RF4信号格式以兼容现有系统
                rf4_formatted = {
                    "signal": rf4_signal_data["signal"],
                    "predicted_return": 0.0,
                    "current_price": rf4_signal_data["current_price"],
                    "timestamp": rf4_signal_data["timestamp"],
                    "info": f"RF4背离策略信号 - {rf4_signal_data['action']}",
                    "strategy": "RF4_Divergence"
                }
                quant_signals.append(rf4_formatted)
                LOGGER.info(f"[{symbol}] 获取到RF4策略信号: {rf4_formatted}")
            else:
                LOGGER.warning(f"[{symbol}] 无法获取RF4信号。")
                quant_signals.append({
                    "signal": "ERROR", 
                    "info": "无法获取RF4信号", 
                    "strategy": "RF4_Divergence"
                })

            # 2. 获取布林带突破策略信号
            bb_signal_data = get_bollinger_breakout_signal(symbol=ccxt_symbol, window=20, std_dev=2.0)
            if bb_signal_data:
                bb_formatted = {
                    "signal": bb_signal_data["signal"],
                    "predicted_return": 0.0,
                    "current_price": bb_signal_data["current_price"],
                    "timestamp": bb_signal_data["timestamp"],
                    "info": f"布林带突破策略 - {bb_signal_data['action']}",
                    "strategy": "Bollinger_Breakout"
                }
                quant_signals.append(bb_formatted)
                LOGGER.info(f"[{symbol}] 获取到布林带突破策略信号: {bb_formatted}")
            else:
                LOGGER.warning(f"[{symbol}] 无法获取布林带突破信号。")
                quant_signals.append({
                    "signal": "ERROR", 
                    "info": "无法获取布林带突破信号", 
                    "strategy": "Bollinger_Breakout"
                })
            
            # 3. 新增：获取MA交叉策略信号
            ma_crossover_signal_data = get_ma_crossover_signal(symbol=ccxt_symbol, fast_period=5, slow_period=20)
            if ma_crossover_signal_data:
                ma_formatted = {
                    "signal": ma_crossover_signal_data["signal"],
                    "predicted_return": 0.0,
                    "current_price": ma_crossover_signal_data["current_price"],
                    "timestamp": ma_crossover_signal_data["timestamp"],
                    "info": f"MA交叉策略 - {ma_crossover_signal_data['action']}",
                    "strategy": "MA_Crossover"
                }
                quant_signals.append(ma_formatted)
                LOGGER.info(f"[{symbol}] 获取到MA交叉策略信号: {ma_formatted}")
            else:
                LOGGER.warning(f"[{symbol}] 无法获取MA交叉信号。")
                quant_signals.append({
                    "signal": "ERROR", 
                    "info": "无法获取MA交叉信号", 
                    "strategy": "MA_Crossover"
                })

            # (这个逻辑现在可能永远不会触发，因为上面已经处理了所有失败情况，但保留作为最终的保险)
            if not quant_signals:
                LOGGER.error(f"[{symbol}] 所有量化策略均未生成有效信号，将使用默认的HOLD信号继续。")
                default_signal = {
                    "signal": "HOLD",
                    "predicted_return": 0.0,
                    "info": "所有量化模型信号获取失败",
                    "strategy": "System_Default"
                }
                quant_signals.append(default_signal)

            return short_term_data, price_data_for_ma, quant_signals
        short_term_data, price_data_for_ma, quant_signal_data = track_process('data_collection', collect_data, current_symbol=symbol)

        # --- 智能分析门禁 (Smart Analysis Gate) ---
        # [重要修正] 此门禁仅对非主攻币种生效
        if not is_primary_symbol:
            is_all_hold = all(s.get('signal', 'HOLD').upper() == 'HOLD' for s in quant_signal_data)
            if is_all_hold:
                LOGGER.info(f"[{symbol}] 所有量化信号均为 'HOLD'，市场无明显交易机会，跳过昂贵的VLM和LLM分析。")
                print(f"\033[38;5;152m[{symbol}] 所有量化信号均为 'HOLD'，跳过深度分析...\033[0m")
                # 返回一个表示HOLD的特殊结果
                return {"decision": "HOLD", "reasoning": "Quant signals all HOLD"}, None
        else:
            LOGGER.info(f"[{symbol}] 是主攻币种，将执行完整的VLM和LLM深度分析，无论量化信号如何。")

        # ======================================================================
        # 步骤 2: 根据模式准备图表资源（统一Gemini模式将仅生成图像；传统模式走VLM分析）
        # ======================================================================
        use_unified = getattr(config, 'DECISION_RULES', {}).get('use_unified_gemini', False)
        vlm_analyzer = None
        kline_image_path_for_unified = None
        if use_unified:
            print(f"\033[38;5;152m[{symbol}] 步骤 2: 统一Gemini模式启用，仅生成1h K线图\033[0m")
            LOGGER.info(f"[{symbol}] 统一Gemini模式启用，仅生成1h K线图（跳过VLM文本分析）")
        else:
            print(f"\033[38;5;152m[{symbol}] 步骤 2: 初始化VLM分析器\033[0m")
            LOGGER.info(f"[{symbol}] 步骤 2: 初始化VLM分析器")
            vlm_analyzer = VLMAnalyzer()
            vlm_analyzer.cache.cleanup_expired_cache()
            cache_stats = vlm_analyzer.cache.get_cache_stats()
            LOGGER.info(f"[{symbol}] 当前VLM缓存状态 - K线图: {cache_stats.get('kline_cache_count', 0)} 条")

        # ======================================================================
        # 步骤 3: 生成并（按模式）分析1h K线图
        # ======================================================================
        print(f"\033[38;5;152m[{symbol}] 步骤 3: 生成1h K线图{'' if use_unified else '并进行VLM分析'}\033[0m")
        LOGGER.info(f"[{symbol}] 步骤 3: 生成1h K线图{'' if use_unified else '并进行VLM分析'}")

        import os
        def save_proxy_env():
            return {k: os.environ.get(k) for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']}
        def clear_proxy_env():
            for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
                if k in os.environ:
                    del os.environ[k]
        def restore_proxy_env(env):
            for k, v in env.items():
                if v is not None:
                    os.environ[k] = v
                elif k in os.environ:
                    del os.environ[k]

        _orig_proxy_env = save_proxy_env()
        clear_proxy_env()
        try:
            if use_unified:
                def generate_image_only():
                    if price_data_for_ma is None or price_data_for_ma.empty:
                        return None
                    result = create_kline_image(price_data_for_ma, timeframe='1h')
                    if not result:
                        return None
                    return result[0]  # image_path
                kline_image_path_for_unified = track_process('kline_image', generate_image_only)
                analysis_1h = None  # 统一模式下不生成文本分析
            else:
                def perform_vlm_analysis():
                    analysis_1h_val, _ = _generate_and_analyze_kline(vlm_analyzer, price_data_for_ma, "1H", "1h")
                    return analysis_1h_val
                analysis_1h = track_process('vlm_analysis', perform_vlm_analysis)
        finally:
            restore_proxy_env(_orig_proxy_env)

        # ======================================================================
        # 步骤 4: 获取市场新闻情报
        # ======================================================================
        print(f"\033[38;5;152m[{symbol}] 步骤 4: 获取市场新闻情报\033[0m")
        LOGGER.info(f"[{symbol}] 步骤 4: 获取市场新闻情报")
        market_news = track_process('news_intelligence', get_market_intelligence, symbol=symbol)

        # ======================================================================
        # 步骤 5: LLM决策引擎（支持统一Gemini路径）
        # ======================================================================
        print(f"\033[38;5;152m[{symbol}] 步骤 5: 获取持仓并进行LLM决策\033[0m")
        LOGGER.info(f"[{symbol}] 步骤 5: 获取持仓并进行LLM决策")
        
        current_position = trader.get_position(symbol)
        try:
            if current_position:
                LOGGER.info(f"[{symbol}] 当前持仓快照: posSide={current_position.get('posSide')}, pos={current_position.get('pos')}, avgPx={current_position.get('avgPx')}, upl={current_position.get('upl')}")
            else:
                LOGGER.info(f"[{symbol}] 当前无持仓或无法获取持仓信息")
        except Exception:
            pass
        current_balance = trader.get_balance('USDT')
        
        if skip_llm:
            print(f"\033[38;5;180m[{symbol}] 已设置--skip-llm，跳过LLM决策分析。\033[0m")
            LOGGER.warning(f"[{symbol}] 已设置--skip-llm，跳过LLM决策分析。")
            final_decision = {
                "decision": "HOLD",
                "reasoning": "Skipped LLM analysis",
                "trade_params": {},
                "suggested_trade_size": 0.95,
                "symbol": symbol
            }
            process_status['llm_decision'] = {
                'status': 'info',
                'duration': '0.0s',
                'message': '跳过LLM分析'
            }
        else:
            if use_unified and kline_image_path_for_unified:
                def perform_unified_decision():
                    try:
                        analyzer = UnifiedGeminiAnalyzer()
                        decision = analyzer.get_trade_decision_unified(
                            quant_signals=quant_signal_data,
                            twitter_data=market_news,
                            kline_image_path=kline_image_path_for_unified,
                            timeframe='1h',
                            current_position=current_position,
                            current_balance=current_balance,
                            symbol=symbol,
                        )
                        decision['symbol'] = symbol
                        return decision
                    except Exception as e:
                        # 优先展开重试错误的“最后一次尝试”的真实异常，便于定位问题
                        if isinstance(e, RetryError):
                            try:
                                last_exc = e.last_attempt.exception()  # type: ignore[attr-defined]
                            except Exception:
                                last_exc = None
                            if last_exc is not None:
                                # 避免格式化器解析异常消息中的花括号，改用 f-string
                                LOGGER.error(f"统一Gemini路径失败（最后一次尝试）: {type(last_exc).__name__}: {last_exc}")
                            else:
                                LOGGER.error(f"统一Gemini路径失败: {type(e).__name__}: {e}")
                        else:
                            LOGGER.error(f"统一Gemini路径失败: {type(e).__name__}: {e}")
                        if getattr(config, 'DECISION_RULES', {}).get('unified_fallback_enabled', True):
                            LOGGER.warning("触发回退：改用 VLM + DeepSeek 传统路径。")
                            ds = DeepSeekAnalyzer()
                            # 如统一路径失败且之前未跑VLM文本，则提供空分析
                            decision_fb = ds.get_trade_decision(
                                quant_signals=quant_signal_data,
                                twitter_data=market_news,
                                kline_analysis={"1h": analysis_1h or ""},
                                current_position=current_position,
                                current_balance=current_balance,
                                symbol=symbol,
                            )
                            decision_fb['symbol'] = symbol
                            return decision_fb
                        raise
                _orig_proxy_env_decision = save_proxy_env()
                clear_proxy_env()
                try:
                    final_decision = track_process('llm_decision', perform_unified_decision)
                finally:
                    restore_proxy_env(_orig_proxy_env_decision)
                # gpt-5.1 复核 Gemini 决策
                if config.DECISION_RULES.get('enable_gpt_reviewer', True):
                    def perform_reviewer():
                        reviewer = GPTReviewer()
                        return reviewer.review(
                            decision=final_decision,
                            kline_image_path=kline_image_path_for_unified,
                            quant_signals=quant_signal_data,
                            twitter_data=market_news,
                            current_position=current_position,
                            current_balance=current_balance,
                        )
                    try:
                        review_result = track_process('gpt_review', perform_reviewer)
                        final_decision['gpt_review'] = review_result
                    except Exception as e:
                        LOGGER.warning(f"[{symbol}] gpt-5.1 审核失败: {e}")
            else:
                def perform_llm_decision():
                    analyzer = DeepSeekAnalyzer()
                    decision = analyzer.get_trade_decision(
                        quant_signals=quant_signal_data,
                        twitter_data=market_news,
                        kline_analysis={"1h": analysis_1h},
                        current_position=current_position,
                        current_balance=current_balance,
                        symbol=symbol
                    )
                    decision['symbol'] = symbol
                    return decision
                _orig_proxy_env_decision = save_proxy_env()
                clear_proxy_env()
                try:
                    final_decision = track_process('llm_decision', perform_llm_decision)
                finally:
                    restore_proxy_env(_orig_proxy_env_decision)

        # ======================================================================
        # 步骤 6: 保存并打印决策报告
        # ======================================================================
        print(f"\033[38;5;152m[{symbol}] 步骤 6: 保存并打印决策报告\033[0m")
        LOGGER.info(f"[{symbol}] 步骤 6: 保存并打印决策报告")
        # 在保存与发送邮件前，注入持仓盈亏快照/状态卡片
        try:
            if current_position:
                asset_name = symbol.split('-')[0] if symbol else '资产'
                qty = current_position.get('pos') or current_position.get('posCcy') or '0'
                avg_price = current_position.get('avgPx') or current_position.get('basePx') or '0'

                # 尽量使用OKX原始未实现盈亏（USDT）
                upl_raw = current_position.get('upl')
                pnl_usd: float
                if upl_raw is not None:
                    try:
                        pnl_usd = float(upl_raw)
                    except Exception:
                        pnl_usd = 0.0
                else:
                    pnl_usd = 0.0

                pos_side = current_position.get('posSide')
                if pos_side == 'net':
                    try:
                        pos_val = float(current_position.get('pos', 0) or 0)
                    except Exception:
                        pos_val = 0.0
                    pos_side_display = "多头" if pos_val > 0 else ("空头" if pos_val < 0 else "无持仓")
                elif pos_side == 'long':
                    pos_side_display = "多头"
                elif pos_side == 'short':
                    pos_side_display = "空头"
                else:
                    pos_side_display = "未知"

                # 若识别为“无持仓”，也显示状态卡片
                if pos_side_display == "无持仓":
                    final_decision['position_snapshot'] = {
                        'status': 'no_position',
                        'desc': '当前无持仓',
                    }
                else:
                    desc = f"{pos_side_display} | 数量: {qty} {asset_name} | 开仓均价: ${avg_price}"
                    if isinstance(pnl_usd, (int, float)):
                        final_decision['position_snapshot'] = {
                            'pnl_usd': pnl_usd,
                            'desc': desc,
                        }
            else:
                # 无法获取持仓或确实无持仓时，也显示“无持仓”卡片
                final_decision['position_snapshot'] = {
                    'status': 'no_position',
                    'desc': '当前无持仓',
                }
        except Exception as e:
            LOGGER.warning(f"构建持仓盈亏快照失败: {e}")

        save_decision_report(final_decision)
        print_decision_report(final_decision)

        # ======================================================================
        # 步骤 7: 执行交易 (仅当币种在可交易列表时)
        # ======================================================================
        print(f"\033[38;5;152m[{symbol}] 步骤 7: 执行交易\033[0m")
        LOGGER.info(f"[{symbol}] 步骤 7: 执行交易")
        
        if symbol not in config.FUTURES['trade_symbols']:
            LOGGER.warning(f"币种 {symbol} 不在可交易列表中。跳过交易执行。")
        else:
            # 此处省略了大量的类型检查和修正代码，因为它们在新结构中是重复的
            try:
                def execute_trade():
                    trader.execute_decision(final_decision)
                
                track_process('trade_execution', execute_trade)
                email_notifier.send_decision_notification(final_decision, execution_success=True, process_status=process_status)
            except Exception as e:
                error_msg = f"交易执行失败: {str(e)}"
                LOGGER.error(error_msg)
                email_notifier.send_decision_notification(final_decision, execution_success=False, error_msg=error_msg, process_status=process_status)
                email_notifier.send_error_notification(
                    f"交易执行错误 ({symbol})", 
                    error_msg, 
                    context={
                        "decision": final_decision.get("decision"),
                        "confidence": final_decision.get("confidence"),
                        "trade_params": str(final_decision.get("trade_params"))
                    }
                )
        
        # 返回最终决策和仓位信息
        return final_decision, current_position

    except Exception as e:
        import traceback
        error_msg = f"[{symbol}] 交易周期发生严重错误: {repr(e)}"
        # 使用 opt(raw=True) 避免格式化问题，或者将 traceback 作为单独参数
        traceback_str = traceback.format_exc()
        # 转义花括号以避免格式化错误
        traceback_str_escaped = traceback_str.replace('{', '{{').replace('}', '}}')
        LOGGER.opt(raw=True).critical(f"{error_msg}\n详细traceback:\n{traceback_str}\n")
        
        email_notifier.send_error_notification(
            f"系统错误 ({symbol})", 
            error_msg, 
            context={
                "traceback": traceback.format_exc()[:500] + "..." if len(traceback.format_exc()) > 500 else traceback.format_exc()
            }
        )
        # 返回错误信息
        return {"decision": "ERROR", "reasoning": str(e)}, None
    finally:
        LOGGER.info(f"========== 币种 {symbol} 处理结束 ==========\n")


def run_trading_cycle(skip_llm: bool = False):
    """
    运行一个完整的交易决策周期。
    采用“机会驱动”策略：主攻BTC，闲时精选一个潜力币。
    """
    print("\033[38;5;109m========== 开始新一轮决策周期 (机会驱动策略) ==========\033[0m")
    LOGGER.info("========== 开始新一轮决策周期 (机会驱动策略) ==========")
    
    trader = OKXTrader(demo_mode=config.DEMO_MODE)
    email_notifier = EmailNotifier()

    # --- 步骤 1: 强制分析主攻币种 (BTC) ---
    main_symbols = config.FUTURES.get('trade_symbols', [])
    if not main_symbols:
        LOGGER.error("配置文件中未设置主攻交易对 'trade_symbols'，决策周期无法启动。")
        return
        
    primary_symbol = main_symbols[0]
    LOGGER.info(f"========== [阶段 1/2] 开始对主攻币种 {primary_symbol} 进行深度分析 ==========")
    
    primary_decision, primary_position = analyze_and_trade_symbol(primary_symbol, trader, email_notifier, skip_llm, is_primary_symbol=True)
    
    # --- 步骤 2: 设立“BTC检查点” ---
    decision_is_hold = primary_decision.get('decision', 'HOLD').upper() == 'HOLD'
    position_exists = primary_position is not None
    
    if not decision_is_hold or position_exists:
        LOGGER.info(f"检查点触发：主攻币种决策不为HOLD(是{primary_decision.get('decision')}) 或 已存在仓位(存在? {position_exists})。本轮决策周期结束。")
        print("\033[38;5;180m主攻币种已有明确信号或持仓，本轮决策周期结束。\033[0m")
        _save_last_run_timestamp()
        return

    # --- 步骤 3: 寻找“最佳替补” ---
    LOGGER.info(f"========== [阶段 2/2] 主攻币种无机会，开始扫描市场寻找最佳替补 ==========")
    print("\033[38;5;109m主攻币种无机会，开始扫描市场寻找最佳替补...\033[0m")
    
    try:
        # 只寻找一个最有潜力的币种
        discovered_symbols = scan_for_opportunities(top_n=1)
        if not discovered_symbols:
            LOGGER.info("市场扫描未发现高潜力币种。")
            print("\033[38;5;152m市场扫描未发现高潜力币种。\033[0m")
        else:
            alternative_symbol = discovered_symbols[0]
            # 确保不重复分析主攻币种
            if alternative_symbol != primary_symbol:
                LOGGER.success(f"市场扫描发现高潜力币种: {alternative_symbol}，开始分析...")
                print(f"\033[38;5;143m市场扫描发现高潜力币种: {alternative_symbol}，开始分析...\033[0m")
                analyze_and_trade_symbol(alternative_symbol, trader, email_notifier, skip_llm, is_primary_symbol=False)
            else:
                LOGGER.info(f"市场扫描发现的币种 {alternative_symbol} 与主攻币种相同，不再重复分析。")

    except Exception as e:
        LOGGER.error(f"市场扫描或替补币种分析过程中失败: {e}", exc_info=True)
        email_notifier.send_error_notification("市场扫描器错误", str(e))

    print("\033[38;5;109m========== 本轮所有币种决策周期结束 ==========\033[0m")
    LOGGER.info("========== 本轮所有币种决策周期结束 ==========\n")
    _save_last_run_timestamp()


def main():
    """主函数，用于设置和运行调度任务。"""
    parser = argparse.ArgumentParser(description="BTC智能决策系统主控程序。")
    parser.add_argument('--now', action='store_true', help='立即运行一次决策周期并退出。')
    parser.add_argument('--skip-llm', action='store_true', help='跳过LLM的API调用，用于调试。')
    args = parser.parse_args()

    # ====== 美观字符画 LOGO ======
    btc_logo = r"""
 ██████╗ ████████╗ ██████╗      ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗ 
 ██╔══██╗╚══██╔══╝██╔════╝      ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝ 
 ██████╔╝   ██║   ██║              ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗
 ██╔══██╗   ██║   ██║              ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║
 ██████╔╝   ██║   ╚██████╗         ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝
 ╚═════╝    ╚═╝    ╚═════╝         ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝ 
    """
    print("\033[38;5;109m" + btc_logo + "\033[0m")  # 莫兰迪蓝绿色
    print("\033[38;5;180m" + "★ BTC_TRADING 智能决策系统 v5.0 ★" + "\033[0m")  # 莫兰迪橄榄色
    print("\033[38;5;102m" + "──────────────────────────────────────────────────────────────────────────────" + "\033[0m")  # 莫兰迪灰绿色

    job = partial(run_trading_cycle, skip_llm=args.skip_llm)

    # ====== Windows 开机自启动（可选，默认开启，可通过 .env 控制） ======
    try:
        ensure_windows_autostart()
    except Exception as e:
        LOGGER.warning(f"AutoStart 注册失败: {e}")

    if args.now:
        print("\033[38;5;143m[启动] 接收到 --now 参数，立即执行一次决策周期...\033[0m")  # 莫兰迪暖灰色
        job()
        print("\033[38;5;143m[完成] 决策周期执行完毕，程序退出。\033[0m")
        return

    print("\033[38;5;152m[调度] BTC_TRADING 主控程序已启动（调度模式）\033[0m")  # 莫兰迪薄荷绿
    print("\033[38;5;102m[调度] 每小时整点自动运行决策周期。\033[0m")  # 莫兰迪灰绿色
    print("\033[38;5;102m[调度] 按 Ctrl+C 可随时退出。\033[0m")
    print("\033[38;5;102m" + "──────────────────────────────────────────────────────────────────────────────" + "\033[0m")

    # 只用schedule的每小时整点调度
    schedule.every().hour.at(":00").do(job)

    last_run_utc = _get_last_run_timestamp()
    if not last_run_utc or (datetime.now(timezone.utc) - last_run_utc).total_seconds() > 3600:
        print("\033[38;5;180m[调度] 首次运行或检测到错过的计划任务，立即执行一次决策周期...\033[0m")  # 莫兰迪橄榄色
        job()

    print("\033[38;5;152m[调度] 系统正在等待下一个调度时间点... (按 Ctrl+C 退出)\033[0m")  # 莫兰迪薄荷绿
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main() 