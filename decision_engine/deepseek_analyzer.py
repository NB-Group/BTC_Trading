import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from btc_predictor.utils import LOGGER

class DeepSeekAnalyzer:
    """
    使用DeepSeek LLM分析量化信号和社交媒体情报，生成交易决策。
    """

    def __init__(self):
        deepseek_config = config.API_KEYS.get('deepseek', {})
        self.base_url = deepseek_config.get('base_url')
        self.api_key = deepseek_config.get('api_key')
        self.model = deepseek_config.get('model', 'deepseek-chat')

        if not all([self.base_url, self.api_key, self.model]):
            raise ValueError("DeepSeek API的配置不完整 (base_url, api_key, model)。")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _parse_llm_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        健壮的JSON解析函数，处理LLM返回的各种格式问题。
        """
        if not response_text:
            raise ValueError("LLM返回了空内容。")
        
        # 清理响应文本
        cleaned_text = response_text.strip()
        
        # 移除可能的markdown代码块标记
        cleaned_text = re.sub(r'^```json\s*', '', cleaned_text)
        cleaned_text = re.sub(r'\s*```$', '', cleaned_text)
        
        # 尝试直接解析
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            LOGGER.warning("直接JSON解析失败，尝试修复格式...")
        
        # 查找所有可能的JSON对象
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, cleaned_text, re.DOTALL)
        
        if not json_matches:
            raise ValueError("在响应中未找到有效的JSON对象")
        
        # 尝试解析每个匹配的JSON对象，返回最后一个成功的
        for i, json_str in enumerate(reversed(json_matches)):
            try:
                # 清理JSON字符串中的多余空白字符
                cleaned_json = re.sub(r'\s+', ' ', json_str.strip())
                result = json.loads(cleaned_json)
                LOGGER.info(f"成功解析第 {len(json_matches) - i} 个JSON对象")
                return result
            except json.JSONDecodeError as e:
                LOGGER.debug(f"第 {len(json_matches) - i} 个JSON对象解析失败: {e}")
                continue
        
        # 如果所有尝试都失败，抛出最后一个错误
        raise ValueError(f"无法解析任何JSON对象。原始响应: {response_text[:200]}...")

    def get_trade_decision(
            self, 
            quant_signal: Dict[str, Any], 
            twitter_data: List[Dict[str, Any]],
            kline_analysis: Dict[str, Optional[str]],
            current_position: Optional[Dict[str, Any]] = None,
            current_balance: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        根据所有输入信息，请求DeepSeek LLM做出最终的交易决策。
        """
        prompt = self._construct_prompt(quant_signal, twitter_data, kline_analysis, current_position, current_balance)
        return self._make_api_call(prompt)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _make_api_call(self, prompt: str) -> Dict[str, Any]:
        """使用tenacity进行带重试的API调用。"""
        
        # 打印提示词（对Twitter部分进行截断）
        self._print_prompt_preview(prompt)
        
        LOGGER.info("向DeepSeek发送请求，进行最终决策分析...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1024,
                stream=False,
                response_format={"type": "json_object"},
            )
            message_content = response.choices[0].message.content
            LOGGER.debug(f"DeepSeek raw response: {message_content}")

            if not message_content:
                raise ValueError("LLM返回了空内容。")

            # 使用健壮的JSON解析
            parsed_result = self._parse_llm_json_response(message_content)
            
            # 确保新字段存在，如果不存在则添加默认值
            if 'key_signals_detected' not in parsed_result:
                parsed_result['key_signals_detected'] = '无关键风险信号'
                LOGGER.warning("模型响应中缺少 key_signals_detected 字段，已添加默认值")
            
            return parsed_result

        except json.JSONDecodeError as e:
            LOGGER.error(f"无法解析LLM返回的JSON: {e}\n响应内容: {message_content}")
            return self._error_response('LLM返回的JSON格式无效')
        except Exception as e:
            LOGGER.error(f"DeepSeek API 请求或解析时发生未知错误: {e}")
            # 重新抛出通用异常以允许tenacity重试
            raise

    def _print_prompt_preview(self, prompt: str):
        """打印提示词预览，对Twitter部分进行截断。"""
        print("\n" + "="*80)
        print("           📝 DeepSeek 提示词预览")
        print("="*80)
        
        lines = prompt.split('\n')
        in_twitter_section = False
        twitter_line_count = 0
        
        for line in lines:
            # 检测是否进入Twitter部分
            if '社交媒体与新闻情报' in line:
                in_twitter_section = True
                twitter_line_count = 0
            elif line.startswith('## ') and in_twitter_section:
                in_twitter_section = False
                
            # 如果在Twitter部分且超过20行，则截断
            if in_twitter_section:
                twitter_line_count += 1
                if twitter_line_count > 20:
                    print("    [... Twitter内容过长，已截断显示 ...]")
                    in_twitter_section = False
                    continue
                    
            print(line)
            
        print("="*80 + "\n")

    def _construct_prompt(
        self, 
        quant_signal: Dict[str, Any], 
        twitter_data: List[Dict[str, Any]],
        kline_analysis: Dict[str, Optional[str]],
        current_position: Optional[Dict[str, Any]] = None,
        current_balance: Optional[float] = None
    ) -> str:
        """
        构建一个更精细化的提示词，区分不同来源的推文并强调时效性。
        """
        signal_part = self._format_quant_signal(quant_signal)
        twitter_part = self._format_twitter_data(twitter_data)
        kline_part = self._format_kline_analysis(kline_analysis)
        position_part = self._format_position_info(current_position)
        # 持仓记忆：尝试读取 last_run.json，补充持仓上下文
        import os
        last_decision_info = ''
        last_json_path = os.path.join(os.path.dirname(__file__), '../last_run.json')
        if os.path.exists(last_json_path):
            try:
                with open(last_json_path, 'r', encoding='utf-8') as f:
                    last_data = json.load(f)
                if last_data.get('decision') in ['LONG', 'SHORT']:
                    last_decision_info = f"\n# 持仓记忆\n- 上次开仓方向: {last_data.get('decision')}\n- 上次开仓原因: {last_data.get('reasoning', '')}\n- 上次关键信号: {last_data.get('key_signals_detected', '')}\n- 上次置信度: {last_data.get('confidence', '')}\n"
            except Exception as e:
                last_decision_info = f"\n# 持仓记忆读取失败: {e}\n"
        balance_part = self._format_balance_info(current_balance)
        
        # 获取当前UTC时间和市场价格
        current_time_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        current_price = quant_signal.get('current_price')
        market_context_part = f"当前BTC/USDT市场价格: **${current_price:.2f}**" if current_price else "无法获取当前市场价格。"

        system_prompt = f"""
# 角色
你是一名顶级的加密货币**期货**短线交易策略师，专注于小时级别（1H）K线的短线快进快出操作。你必须在大量信息中精准识别关键信号，尤其关注能在1-6小时内带来收益的短线机会，忽略长线趋势。

# 约束与要求
- **短线优先 (硬性要求)**: 你的所有分析和决策必须以小时级别（1H）K线和短线信号为主，优先考虑1-6小时内的盈利机会。日线、周线仅作为背景参考，不能主导决策。
- **快进快出**: 你的目标是捕捉短线波动，快进快出，避免长时间持仓。
- **资金优先 (硬性要求)**: 你的首要任务是确保能成功下单。如果决策是开仓（LONG/SHORT），你必须根据当前余额和市场价格计算能够满足交易所 **0.01 BTC** 最小开仓量的最低杠杆。
  - **计算公式**: `所需杠杆 = (0.01 * 当前市场价格) / (当前账户余额 * 0.95)`
  - **决策逻辑**:
    - 计算出`所需杠杆`后，向上取整（例如，2.1倍计算为3倍）。
    - 如果`所需杠杆` > 5 (最大允许杠杆)，则最终决策必须是 `HOLD`，并在 `reasoning` 中明确指出“因资金不足，即使5倍杠杆也无法满足最小开仓量，故放弃交易”。
    - 否则，在 `trade_params` 中必须使用计算出的`所需杠杆`。
- **持仓优先**: 你的最终决策必须与当前持仓状态逻辑一致。
- **操作含义说明**:
  - `LONG`：开多仓（买入做多）
  - `SHORT`：开空仓（卖出做空）
  - `HOLD`：观望，不操作
  - `CLOSE_LONG`：平掉多仓（卖出已持有的多仓）
  - `CLOSE_SHORT`：平掉空仓（买入已持有的空仓）
- **持仓方向判断规则**:
  - 如果 `posSide` 字段为 `net`，则根据 `pos` 数值判断方向：`pos > 0` 为多仓，`pos < 0` 为空仓。
  - 只有持有多仓时才允许 `CLOSE_LONG`，只有持有空仓时才允许 `CLOSE_SHORT`。
  - 如果当前无持仓，则不允许返回 `CLOSE_LONG` 或 `CLOSE_SHORT`，只能返回 `HOLD`。
  - 如果建议的平仓方向与实际持仓方向不符，必须返回 `HOLD`，并在 reasoning 里说明原因。
- **关键信号优先**: 如果发现任何"大海捞针"类型的关键信号，必须在reasoning中明确提及并重点分析。
- **时效性**: 信息具有极强的时效性，优先考虑最新发布的情报。
- **信号冲突处理**: 如果关键风险信号与量化模型信号冲突，关键风险信号具有优先权。
- **风险管理**: 如果市场信息极度混乱或出现重大不确定性，且有持仓，首选是平仓(`CLOSE_LONG`/`CLOSE_SHORT`)；如果空仓，则决策为`HOLD`。
- **JSON输出**: 必须严格按照指定的JSON格式输出，不要包含任何额外说明或```json```标记。
- **内部量化模型信号**: 其依赖于MA60与k线的交叉信号，如果没有，均输出HOLD，因此其输出HOLD时，不得作为任何决策的依据，在其做出除HOLD以外的决策时，你要加大其在判断中的权重。
- **短线盈利优先**: 只要VLM技术分析显示短线盈利机会（1H K线），就优先考虑短线盈利机会，忽略长期金融市场趋势。
- **如果持有仓位与长期方向相同，但与短期方向相反，在盈利时可以平仓，但不盈利时建议继续持有，总之就是慎防亏损最多的时候卖出。**

        # 当前市场状态与持仓记忆
        - **分析时间**: {current_time_utc}
        - {market_context_part}
        {last_decision_info}

# 信息输入
{position_part}
{balance_part}
{kline_part}
{twitter_part}
{signal_part}

# 分析框架与优先级 (必须遵守)
1.  **短线机会识别 (最高优先级)**: 以1H K线和短线信号为主，寻找1-6小时内的具体交易机会。
2.  **宏观趋势判断 (仅作背景参考)**: 结合周线和日线分析，仅用于判断大方向，不主导短线决策。
3.  **当前持仓状态**: 评估当前是空仓、持有多仓还是持有空仓。
4.  **关键风险信号扫描**: 快速扫描所有信息，识别可能改变趋势的关键信号。
5.  **内部量化模型与新闻**: 作为辅助信息，验证或调整交易决策。

# 交易参数设置指南
**短线交易策略**:
- 以小时级别波动为主，止损建议4%（常规）或5%（信号强烈时），止盈建议8%（常规）或10%（信号强烈时）。
- 仓位大小95%（全仓操作，但用杠杆控制风险）。

# JSON输出格式
{{
  "decision": "LONG/SHORT/HOLD/CLOSE_LONG/CLOSE_SHORT",
  "reasoning": "详细说明你做出决策的完整逻辑链，必须首先陈述当前持仓状态，然后报告是否发现了关键风险信号，最后体现你对'分析框架与优先级'的遵守，解释你如何权衡不同来源的数据。务必参考上次开仓原因、信号、置信度等持仓记忆，只有反向信号足够强或达到止损/止盈条件时才建议平仓。",
  "key_signals_detected": "如果发现了'大海捞针'类型的关键信号，请明确列出；如果没有发现，请说明'无关键风险信号'",
  "confidence": "请根据当前与历史信号自动判断置信度，范围0-1，不要固定。",
  "suggested_trade_size": 0.95,
  "trade_params": {{
    "leverage": 2,
    "take_profit_pct": 8.0,
    "stop_loss_pct": 4.0
  }},
  "risk_assessment": "对此次交易潜在风险的简要评估，特别注意是否存在黑天鹅风险。"
}}
"""
        return system_prompt.strip()

    def _format_quant_signal(self, quant_signal: Dict[str, Any]) -> str:
        """格式化量化信号部分。"""
        signal_info = quant_signal.get('info', '当前无明确金叉/死叉信号。')
        signal_type = quant_signal.get('signal', 'HOLD')
        predicted_return = quant_signal.get('predicted_return', 0.0)
        current_price = quant_signal.get('current_price')

        quant_part = "### 内部量化模型信号\n"
        if signal_type in ['BUY', 'SELL']:
            quant_part += f"- **信号类型**: **{signal_type}**\n"
            quant_part += f"- **模型预测回报率**: **{predicted_return:.4f}%**\n"
        else:
            quant_part += "- **信号类型**: **HOLD (无明确交易信号)**\n"
        quant_part += f"- **模型分析**: {signal_info}\n"
        if current_price:
            quant_part += f"- **参考价格**: ${current_price:.2f}\n"
        return quant_part

    def _format_twitter_data(self, twitter_data: List[Dict[str, Any]]) -> str:
        """格式化新闻数据。"""
        
        twitter_part = "## 1. 社交媒体与新闻情报\n\n"
        if not twitter_data:
            return twitter_part + "无有效的社交媒体或新闻情报。\n"

        # 由于数据源已更改为CoinDesk，我们不再进行细分
        twitter_part += "### 1.1 CoinDesk 最新市场新闻 (高优先级)\n"
        twitter_part += "分析以下来自CoinDesk的最新市场新闻：\n"
        for news_item in twitter_data:
            # 复用 _format_tweet 方法来格式化新闻，因为它结构相似
            twitter_part += self._format_tweet(news_item)
        twitter_part += "\n"
        
        return twitter_part

    def _format_kline_analysis(self, kline_analysis_dict: Dict[str, Optional[str]]) -> str:
        """格式化K线分析部分，确保即使部分分析失败也能展示有效的部分。"""
        kline_part = "## 3. K线图技术分析 (VLM模型提供)\n"
        
        valid_analyses = []
        
        # 按指定顺序添加分析，并检查有效性
        order = {'weekly': '周线 (Weekly) K线分析', 'daily': '日线 (Daily) K线分析', 'short_term': '短期 (1H) K线分析'}
        
        for key, title in order.items():
            analysis = kline_analysis_dict.get(key)
            if analysis and isinstance(analysis, str) and analysis.strip():
                valid_analyses.append(f"### {title}\n{analysis.strip()}\n")
                
        if not valid_analyses:
            return kline_part + "\n无有效的K线图分析结果。"
            
        return kline_part + "\n".join(valid_analyses)

    def _format_position_info(self, position_data: Optional[Dict[str, Any]]) -> str:
        """格式化当前持仓信息。"""
        position_part = "## 1. 当前持仓状态\n\n"
        if not position_data or not position_data.get('posSide'):
            position_part += "当前 **空仓**。\n"
            return position_part

        side = position_data.get('posSide')
        qty = position_data.get('posCcy') or position_data.get('pos')  # 兼容性
        avg_price = position_data.get('avgPx')
        unrealized_pnl = position_data.get('upl')
        leverage = position_data.get('lever')

        # 新增net模式判断
        if side == 'net':
            try:
                pos_val = float(position_data.get('pos', 0))
            except Exception:
                pos_val = 0
            if pos_val > 0:
                position_part += f"- **持仓方向**: **做多 (LONG, net模式, pos={pos_val})**\n"
            elif pos_val < 0:
                position_part += f"- **持仓方向**: **做空 (SHORT, net模式, pos={pos_val})**\n"
            else:
                position_part += "当前 **空仓**。\n"
                return position_part
        elif side == 'long':
            position_part += f"- **持仓方向**: **做多 (LONG)**\n"
        elif side == 'short':
            position_part += f"- **持仓方向**: **做空 (SHORT)**\n"

        position_part += f"- **持仓数量**: {qty} BTC\n"
        position_part += f"- **开仓均价**: ${avg_price}\n"
        position_part += f"- **杠杆倍数**: {leverage}x\n"
        position_part += f"- **未实现盈亏**: **${unrealized_pnl}**\n"
        return position_part

    def _format_balance_info(self, balance: Optional[float]) -> str:
        """格式化当前账户余额信息。"""
        balance_part = "## 2. 当前账户余额\n\n"
        if balance is not None:
            balance_part += f"当前账户余额: **${balance:.2f} USDT**\n"
            balance_part += f"可用于交易的资金: **${balance * 0.95:.2f} USDT** (95%)\n"
        else:
            balance_part += "无法获取当前账户余额信息。\n"
        return balance_part

    def _format_tweet(self, tweet: Dict[str, Any]) -> str:
        """格式化单条推文或新闻，包含来源、时间戳、标题和摘要。"""
        source = tweet.get('source', 'Unknown')
        
        user = tweet.get('username', 'N/A')
        
        title = tweet.get('text', '无标题').replace('\n', ' ').strip()
        description = tweet.get('description', '无摘要').replace('\n', ' ').strip()
        
        created_at = tweet.get('created_at', '未知时间')
        
        vlm_analysis = tweet.get('vlm_analysis')

        source_display = f"来自: {source}"
        if user != 'N/A':
            source_display += f" (@{user})"
        
        formatted_tweet = f"\n**{source_display}** | {created_at}\n"
        formatted_tweet += f"**标题**: {title}\n"
        if description and description != '无摘要':
            formatted_tweet += f"**摘要**: {description}\n"
        if vlm_analysis:
            formatted_tweet += f"**VLM分析**: {vlm_analysis}\n"
        formatted_tweet += "-" * 50 + "\n"
        
        return formatted_tweet

    def _error_response(self, reason: str) -> Dict[str, Any]:
        """返回错误响应。"""
        return {
            "decision": "HOLD",
            "reasoning": f"由于内部错误，无法进行决策分析: {reason}",
            "key_signals_detected": "由于内部错误，无法检测关键信号",
            "confidence": 0.0,
            "trade_params": {
                "leverage": 0,
                "take_profit_price": None,
                "stop_loss_price": None
            },
            "risk_assessment": "由于内部错误，无法评估风险。"
        }

if __name__ == '__main__':
    from btc_predictor.utils import setup_logger
    setup_logger()
    
    LOGGER.info("--- 运行 DeepSeekAnalyzer 独立测试 ---")
    
    # 模拟测试数据
    test_quant_signal = {
        "signal": "HOLD",
        "predicted_return": 0.0,
        "info": "测试量化信号",
        "current_price": 30000.0
    }
    
    test_twitter_data = [
        {
            "source": "CoinDesk",
            "username": "test_user",
            "text": "测试新闻标题",
            "description": "测试新闻摘要",
            "created_at": "2025-01-01 12:00:00"
        }
    ]
    
    test_kline_analysis: Dict[str, Optional[str]] = {
        "short_term": "Short-term analysis shows a potential breakout.",
        "daily": "Daily chart confirms the uptrend.",
        "weekly": "Weekly trend is bullish."
    }
    
    try:
        analyzer = DeepSeekAnalyzer()
        result = analyzer.get_trade_decision(
            quant_signal=test_quant_signal,
            twitter_data=test_twitter_data,
            kline_analysis=test_kline_analysis,
            current_position=None,
            current_balance=500.0
        )
        print("\n" + "="*25 + " 测试结果 " + "="*25)
        print(f"决策结果: {result}")
        print("="*64 + "\n")
    except Exception as e:
        LOGGER.error(f"独立测试期间发生错误: {e}") 