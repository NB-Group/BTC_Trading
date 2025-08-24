import json
import os
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from btc_predictor.utils import LOGGER

TRADE_LOG_FILE = 'trade_log.json'


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
            quant_signals: List[Dict[str, Any]], 
            twitter_data: List[Dict[str, Any]],
            kline_analysis: Dict[str, Optional[str]],
            current_position: Optional[Dict[str, Any]] = None,
            current_balance: Optional[float] = None,
            symbol: str = 'BTC-USDT-SWAP'  # 新增 symbol 参数
    ) -> Dict[str, Any]:
        """
        根据所有输入信息，请求DeepSeek LLM做出最终的交易决策。
        """
        prompt = self._construct_prompt(quant_signals, twitter_data, kline_analysis, current_position, current_balance, symbol)
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

    def _get_lessons_from_log(self, limit: int = 3) -> str:
        """从交易日志中读取最近的亏损交易，生成学习教训。"""
        log_path = os.path.join(os.path.dirname(__file__), '..', TRADE_LOG_FILE)
        if not os.path.exists(log_path):
            return ""

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                trade_logs = json.load(f)
        except (json.JSONDecodeError, IOError):
            return ""

        losing_trades = [t for t in trade_logs if t.get('pnl', 0) < 0]
        
        # 按时间倒序排序，获取最近的亏损
        losing_trades.sort(key=lambda x: x.get('exit_timestamp_utc', ''), reverse=True)
        
        recent_losses = losing_trades[:limit]
        
        if not recent_losses:
            return ""

        lessons_part = "\n# 历史教训 (复盘最近的亏损交易)\n"
        lessons_part += "--- \n"
        lessons_part += "在做本次决策前，请务必回顾并吸取以下最近几次失败交易的教训，避免重蹈覆辙。\n"

        for i, trade in enumerate(recent_losses):
            entry_report = trade.get('entry_report', {})
            lessons_part += f"\n### 复盘案例 {i+1}: 一笔 {trade.get('symbol')} 的亏损交易\n"
            lessons_part += f"- **方向**: {entry_report.get('decision', 'N/A')}\n"
            lessons_part += f"- **最终盈亏**: **{trade.get('pnl', 0):.2f} USDT**\n"
            lessons_part += f"- **当时的决策理由**: \"{entry_report.get('reasoning', '无记录')}\"\n"
            lessons_part += f"- **当时的关键信号**: \"{entry_report.get('key_signals_detected', '无记录')}\"\n"
            
            # [学习闭环增强] 新增：展示行情复盘和核心教训
            outcome = trade.get('outcome_analysis')
            if outcome:
                lessons_part += f"- **行情复盘与教训**: {outcome} **核心教训**: 当市场处于下跌趋势后的弱反弹时，即使VLM给出看涨信号，也应警惕‘多头陷阱’，优先选择观望。\n"

        lessons_part += "\n---\n"
        return lessons_part

    def _construct_prompt(
        self, 
        quant_signals: List[Dict[str, Any]], 
        twitter_data: List[Dict[str, Any]],
        kline_analysis: Dict[str, Optional[str]],
        current_position: Optional[Dict[str, Any]] = None,
        current_balance: Optional[float] = None,
        symbol: str = 'BTC-USDT-SWAP' # 新增 symbol 参数
    ) -> str:
        """
        构建一个更精细化的提示词，区分不同来源的推文并强调时效性。
        """
        # --- [学习闭环] 新增：获取历史教训 ---
        lessons_part = self._get_lessons_from_log()

        signal_part = self._format_quant_signals(quant_signals) # 修改调用
        twitter_part = self._format_twitter_data(twitter_data)
        kline_part = self._format_kline_analysis(kline_analysis)
        position_part = self._format_position_info(current_position, symbol) # 传递symbol
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
        
        # 从信号列表中获取当前价格 (选择第一个信号的价格作为代表)
        current_price = None
        if quant_signals and isinstance(quant_signals, list) and quant_signals[0].get('current_price'):
            current_price = quant_signals[0].get('current_price')

        # 使用传入的 symbol 格式化交易对名称
        asset_name = symbol.split('-')[0]
        market_context_part = f"当前 {asset_name}/USDT 市场价格: **${current_price:.2f}**" if current_price else "无法获取当前市场价格。"

        system_prompt = f"""
# 角色
你是一名顶级的加密货币**期货**短线交易策略师，专注于小时级别（1H）K线的短线快进快出操作。你必须在大量信息中精准识别关键信号，尤其关注能在1-6小时内带来收益的短线机会。你正在为 **{asset_name}** 这个币种做决策。

{lessons_part}
# 核心原则
1.  **顺势与机会捕捉 (最高优先级)**:
    *   **默认中性**: 在证据不足或信号冲突时，首选 `HOLD`，避免勉强进场。
    *   **震荡行情处理**: 当VLM分析识别出市场处于横盘震荡（例如布林带收窄，价格在区间内波动）时，主要策略应为 `HOLD`，耐心等待明确的突破信号，避免在区间内被反复止损。
    *   **牛市思维**: 在市场整体上涨或有强烈看涨信号时（如VLM分析显示多头排列、关键支撑位反弹），优先考虑 **LONG** 机会；在无明确利空的情况下，不因局部回撤而做空。
    *   **陷阱识别 (Trap Detection)**:
        *   **弱反弹陷阱 (Weak Bounce Trap)**: 如果VLM分析的看涨信号，是出现在一段**下跌趋势之后**的**首次**弱势反弹（例如，价格刚刚重新站上短期均线，但K线实体小，缺乏力量），那么**必须**将其视为潜在的“多头陷阱”。此时，最终决策应**优先选择 `HOLD`**，而不是 `LONG`，并在`reasoning`中说明“正在观察反弹的有效性，警惕多头陷阱”。
    *   **空头门槛（更高要求）**: 只有在以下至少两项同时满足时，才可考虑 `SHORT`：
        1) VLM 1H 结论为明确看跌并给出关键位被跌破；
        2) 内部量化模型矩阵中至少一个模型为 `SELL` 或出现强烈的空头动量信号；
        3) 新闻情报出现突发性实质性利空（可验证来源）；
        4) 更高周期（日线）不处于明确的多头趋势，或多头趋势出现关键位失守。
    *   **果断出击**: 当多个来源（VLM、量化信号、新闻）指向同一方向时，应果断决策，提高置信度。
    *   **短线盈利优先**: 只要VLM技术分析显示明确的短线盈利机会（1H K线），就优先考虑短线盈利机会，忽略长期金融市场趋势。

2.  **风险管理与资金保护**:
    *   **资金优先 (硬性要求)**: 你的首要任务是确保能成功下单。如果决策是开仓（LONG/SHORT），你必须根据当前余额和市场价格计算能够满足交易所 **最小开仓量** 的最低杠杆。
      *   **计算公式**: `所需杠杆 = (最小开仓名义价值) / (当前账户余额 * 0.95)` (例如BTC最小开仓量为0.01, ETH为0.1)
      *   **决策逻辑**:
        *   计算出`所需杠杆`后，向上取整（例如，2.1倍计算为3倍）。
        *   如果`所需杠杆` > 5 (最大允许杠杆)，则最终决策必须是 `HOLD`，并在 `reasoning` 中明确指出“因资金不足，即使5倍杠杆也无法满足最小开仓量，故放弃交易”。
        *   否则，在 `trade_params` 中必须使用计算出的`所需杠杆`。
    *   **趋势一致性**: 若更高周期（日线）呈现明确上升趋势，则避免逆势做空，除非满足“空头门槛”。若呈现明确下降趋势，空头亦需满足1H关键位与动量共振。
    *   **信号冲突处理**: 当信号冲突时，综合评估风险与机遇。风险信号不再具有绝对优先权，而是作为调整仓位和置信度的依据；若仍无法形成清晰优势，选择 `HOLD`。
    *   **极端行情处理**: 在市场极端恐慌或狂热时，若无持仓，首选 `HOLD`；若有持仓，则根据盈利情况和短期趋势决定是否平仓。

# 操作指南
- **操作定义**:
  - `LONG`: 开多仓 (买入做多)
  - `SHORT`: 开空仓 (卖出做空)
  - `HOLD`: 观望，不操作
  - `CLOSE_LONG`: 平掉多仓
  - `CLOSE_SHORT`: 平掉空仓
- **持仓一致性**:
  - 决策必须与当前持仓状态逻辑一致（例如，不能在没有多仓时 `CLOSE_LONG`）。
  - `posSide` 为 `net` 时，`pos > 0` 为多仓，`pos < 0` 为空仓。
- **平仓逻辑**:
  - **止盈**: 当VLM分析或关键新闻表明短期趋势可能反转，且当前有盈利时，应果断平仓。
  - **止损**: 当价格触及预设的止损点，或出现强烈的反向关键信号时，必须平仓。
  - **持有逻辑**: 如果持有仓位与短期趋势方向一致，即使有小幅回调，也应继续持有，以捕捉更大的波动。

# 信息解读
- **内部量化模型矩阵**: 你会收到一个包含多个策略信号的列表。每个信号都有独立的来源和逻辑。
  - **信号一致性**: 如果多个策略（如RF4背离和MA交叉）发出相同方向的信号（都是BUY或都是SELL），这是一个强烈的共振信号，应显著提高决策置信度。
  - **信号冲突**: 如果策略信号相互矛盾（一个BUY，一个SELL），这表明市场方向不明朗，风险较高。在这种情况下，应优先考虑 `HOLD`，或至少降低仓位和杠杆，并更加依赖VLM对当前K线形态的直接解读来打破僵局。
  - **单一信号**: 如果只有一个策略发出明确信号，应将其视为重要参考，但不是决定性因素，需要与其他信息源（VLM K线分析、新闻）进行交叉验证。
- **VLM K线分析**: 这是短线决策的核心依据。你现在会收到三个时间周期的分析：`日线 (Daily)`，`1小时线 (1H)` 和 `15分钟线 (15_min)`。
  - **分析逻辑**: 遵循“日线定方向，小时找区间，分钟定买卖”的原则。
    1.  **日线 (Daily)**: 首先看日线分析，确定市场当前的大方向（上涨、下跌、震荡）。日线是你的战略地图。
    2.  **1小时线 (1H)**: 在日线确定的方向上，使用1小时线寻找具体的交易区间和关键的支撑/阻力位。1小时线是你的战术部署。
    3.  **15分钟线 (15_min)**: 这是执行层面的核心依据。它用于确定精准的入场和出场点。**15分钟线的操作建议权重最高**，但必须与日线和1小时线的大方向不冲突。
  - **结构化指令**: `15_min` 和 `1H` 的分析会提供结构化的`操作建议`。**你必须严格遵循这些指令**。例如，如果 `15_min` 信号是`做空`，条件是`价格低于65000`，并且1H和日线没有强烈的看涨冲突，那么只有当**当前市场价格**低于65000时，你的决策才能是`SHORT`。如果条件不满足，即使VLM看跌，你的决策也应该是`HOLD`，并在`reasoning`中说明你在等待条件触发。

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

# 分析框架 (必须遵守)
1.  **识别核心机会**: 结合VLM分析（特别是1H图）和关键新闻，判断是否存在明确的做多或做空机会。若证据不足或冲突，则为 `HOLD`。若为 `SHORT`，请在推理中明确列出满足的“空头门槛”条目。
2.  **评估持仓状态**: 确定当前是空仓、持有多仓还是空仓，并计算盈亏。
3.  **整合辅助信号**: 使用内部量化模型和宏观K线分析（日线）来验证或微调决策。
4.  **风险评估与参数设定**: 基于市场波动性和信号强度，设定合理的杠杆、止盈和止损。

# JSON输出格式
{{
  "decision": "LONG/SHORT/HOLD/CLOSE_LONG/CLOSE_SHORT",
  "reasoning": "详细说明你做出决策的完整逻辑链。首先陈述当前持仓和盈亏状况，然后分析核心机会（VLM和新闻），接着整合辅助信号，最后基于风险评估得出结论。务必体现你对'核心原则'和'分析框架'的遵守。",
  "key_signals_detected": "明确列出本次决策所依据的最关键信号（多/空信号或风险信号）；如果没有，请填写'无特别关键信号'。",
  "confidence": "请根据当前与历史信号自动判断置信度，范围0-1，不要固定。",
  "suggested_trade_size": 0.95,
  "trade_params": {{
    "leverage": 2,
    "take_profit_pct": 8.0,
    "stop_loss_pct": 4.0
  }},
  "risk_assessment": "对此次交易潜在风险的简要评估，并说明你的风控措施（如止损点设置）。"
}}
"""
        return system_prompt.strip()

    def _format_quant_signals(self, quant_signals: List[Dict[str, Any]]) -> str:
        """格式化多个量化策略的信号部分。"""
        if not quant_signals:
            return "### 内部量化模型矩阵\n- 未提供任何量化信号。\n"

        quant_part = "### 内部量化模型矩阵\n"
        for signal_data in quant_signals:
            # 确保即使没有strategy字段也不会报错
            strategy = signal_data.get('strategy', '未知策略') 
            signal_type = signal_data.get('signal', 'HOLD')
            
            # 优先使用 action 字段，如果不存在则使用 info
            info = signal_data.get('action') or signal_data.get('info', '无详细信息。')
            current_price = signal_data.get('current_price')

            quant_part += f"\n#### 策略: {strategy}\n"
            quant_part += f"- **信号类型**: **{signal_type}**\n"
            quant_part += f"- **策略分析**: {info}\n"
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
        
        # 按指定顺序添加分析，并检查有效性（优先展示1H，减少高周期锚定偏差）
        order = {'short_term': '短期 (1H) K线分析', '15_min': '精细 (15分钟) K线分析', 'daily': '日线 (Daily) K线分析'}
        
        for key, title in order.items():
            analysis = kline_analysis_dict.get(key)
            if analysis and isinstance(analysis, str) and analysis.strip():
                valid_analyses.append(f"### {title}\n{analysis.strip()}\n")
                
        if not valid_analyses:
            return kline_part + "\n无有效的K线图分析结果。"
            
        return kline_part + "\n".join(valid_analyses)

    def _format_position_info(self, position_data: Optional[Dict[str, Any]], symbol: str) -> str:
        """格式化当前持仓信息。"""
        position_part = "## 1. 当前持仓状态\n\n"
        if not position_data or not position_data.get('posSide'):
            position_part += "当前**无持仓**。\n"
            return position_part

        side = position_data.get('posSide')
        qty = position_data.get('posCcy') or position_data.get('pos')  # 兼容性
        avg_price = position_data.get('avgPx')
        unrealized_pnl = position_data.get('upl')
        leverage = position_data.get('lever')

        # 从 symbol 中提取资产名称 (e.g., 'BTC' from 'BTC-USDT-SWAP')
        asset_name = symbol.split('-')[0] if symbol else '资产'

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
                position_part += "当前**无持仓**。\n"
                return position_part
        elif side == 'long':
            position_part += f"- **持仓方向**: **做多 (LONG)**\n"
        elif side == 'short':
            position_part += f"- **持仓方向**: **做空 (SHORT)**\n"

        position_part += f"- **持仓数量**: {qty} {asset_name}\n"
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
    test_quant_signals = [
        {
            "strategy": "RF4_Divergence",
            "signal": "BUY",
            "action": "看涨背离",
            "info": "测试RF4背离信号",
            "current_price": 30000.0
        },
        {
            "strategy": "Bollinger_Breakout",
            "signal": "SELL",
            "action": "跌破布林带下轨",
            "info": "测试布林带突破信号",
            "current_price": 30500.0
        }
    ]
    
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
        "15_min": "15-min chart shows an entry point."
    }
    
    try:
        analyzer = DeepSeekAnalyzer()
        result = analyzer.get_trade_decision(
            quant_signals=test_quant_signals,
            twitter_data=test_twitter_data,
            kline_analysis=test_kline_analysis,
            current_position=None,
            current_balance=500.0,
            symbol='BTC-USDT-SWAP' # 添加测试symbol
        )
        print("\n" + "="*25 + " 测试结果 " + "="*25)
        print(f"决策结果: {result}")
        print("="*64 + "\n")
    except Exception as e:
        LOGGER.error(f"独立测试期间发生错误: {e}") 