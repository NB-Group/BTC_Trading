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

        # 初始化日志：打印“仅VLM模式”与关键决策参数
        try:
            rules = getattr(config, 'DECISION_RULES', {})
            vlm_solo = rules.get('vlm_solo_trade', True)
            probe_ratio = rules.get('probe_position_ratio', 0.3)
            strict_long = rules.get('strict_long_trigger', True)
            LOGGER.info(f"DeepSeekAnalyzer 初始化: VLM单源模式={'启用' if vlm_solo else '禁用'}, 试探仓比例={probe_ratio:.2f}, 严格做多触发={'是' if strict_long else '否'}")
        except Exception:
            pass

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
        decision = self._make_api_call(prompt)

        # 若是横盘（或决策为HOLD），尝试做突破区间预测
        try:
            one_h_analysis = kline_analysis.get('1h') if kline_analysis else None
            if one_h_analysis and isinstance(one_h_analysis, str):
                if decision.get('decision', '').upper() == 'HOLD' or '震荡' in one_h_analysis or '横盘' in one_h_analysis:
                    decision['range_forecast'] = self._predict_range_breakout(one_h_analysis, symbol)
        except Exception as e:
            LOGGER.warning(f"横盘突破预测失败: {e}")
        return decision

    def _predict_range_breakout(self, one_h_text: str, symbol: str) -> Dict[str, Any]:
        """基于1H分析文本，调用LLM估计：横盘上下沿、概率、预计持续时间、触发条件与挂单计划。"""
        range_prompt = f"""
你现在的任务：从下面的 1H 技术分析文本中，如果存在震荡/横盘/区间特征，提取其区间上沿与下沿，并做出突破预测计划。
要求：
1. 严格输出 JSON，不要多余文字；数字用浮点。
2. 如果未能识别有效区间，输出 reason 并置 is_range=false。
3. 若存在区间：
   - 估计上下沿价位 (support, resistance)；若文本无明确价位，可基于语义推测一个合理整数价（保留到整数或50/100步长），并标记 inferred=true。
   - 估计未来 2-8 小时内向上/向下突破概率 (0-1)。两者和不必为1，但需合理。
   - 给出预计横盘剩余持续时间（分钟），以及若突破向上/向下的初步目标价 (target_up / target_down)。
   - 给出一个简单执行计划：
       * plan.long_trigger: 上破触发条件 (如 price > resistance * 1.002)
       * plan.short_trigger: 下破触发条件 (如 price < support * 0.998)
       * plan.long_cancel / short_cancel: 何种情况取消等待。
       * plan.initial_sl_pct / tp_pct: 推荐止损/止盈百分比。
4. 若当前结构不适合提前挂单（例如波动剧烈或无效区间），给出 is_range=false 和 reason。

【1H分析原文】\n{one_h_text}\n
请直接返回 JSON：
{{
  "is_range": true,
  "support": 0,
  "resistance": 0,
  "inferred": false,
  "prob_break_up": 0.0,
  "prob_break_down": 0.0,
  "expected_remaining_minutes": 0,
  "target_up": 0,
  "target_down": 0,
  "plan": {{
     "long_trigger": "",
     "short_trigger": "",
     "long_cancel": "",
     "short_cancel": "",
     "initial_sl_pct": 0.0,
     "initial_tp_pct": 0.0
  }},
  "reason": ""
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": range_prompt}],
                temperature=0.3,
                max_tokens=600,
                stream=False,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            parsed = self._parse_llm_json_response(raw)
            # 最低字段保障
            for k, v in {"is_range": False, "reason": "模型未给出说明"}.items():
                parsed.setdefault(k, v)
            return parsed
        except Exception as e:
            LOGGER.warning(f"范围预测调用失败: {e}")
            return {"is_range": False, "reason": f"range_forecast_error: {e}"}

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

        default_leverage = int(getattr(config, 'FUTURES', {}).get('leverage', 3))

        system_prompt = f"""
# 角色
你是一名顶级的加密货币**期货**短线交易策略师，当前策略已精简为单一 **1小时 (1H)** 时间框的快进快出操作。你必须在有限信息中精准识别 1H 级别可在 1-6 小时内实现的收益机会。你正在为 **{asset_name}** 这个币种做决策。

{lessons_part}
# 核心原则
1.  **顺势与机会捕捉 (最高优先级)**:
    *   **默认中性**: 在证据不足或信号冲突时，首选 `HOLD`，避免勉强进场。
    *   **震荡行情处理**: 当VLM分析识别出市场处于横盘震荡（例如布林带收窄，价格在区间内波动）时，主要策略应为 `HOLD`，耐心等待明确的突破信号，避免在区间内被反复止损。
    *   **做多优先**: 当出现多头排列、关键支撑位反弹或放量突破等看涨共振信号时，优先考虑 **LONG**；不要因为局部回撤而轻易做空。
    *   **陷阱识别 (Trap Detection)**: 由 VLM 技术分析负责识别弱反弹/多头陷阱；若VLM提示“弱反弹/量能不足”，可降低仓位或等待确认。
    *   **空头门槛（动态要求）**:
        *   理想情况下，下列信号至少有两项共振，再执行 `SHORT`；
        *   若目前仅有其中一项成立，但其余来源（量化/新闻/VLM）均为中性或明确“无冲突”，也可以执行 `SHORT`，同时要在 reasoning 中备注“单一空头信号 + 无冲突”并适度降低置信度或仓位；
        *   一旦任意来源给出明确的反向看多信号，则必须回到 `HOLD`，等待更多证据。
        *   候选空头证据：
            1) 1H 级别结论明确看跌且关键位被有效跌破；
            2) 内部量化模型矩阵中至少一个模型为 `SELL` 或出现强烈空头动量；
            3) 新闻情报出现突发、可信的实质性利空。
    *   **果断出击**: 当多个来源（VLM、量化信号、新闻）指向同一方向时，提高置信度并果断执行。
    *   **短线盈利优先**: 只基于 1H 结构与动量判断机会；分钟级噪声忽略。

2.  **风险管理与资金保护**:
    *   **资金优先 (硬性要求)**: 若决策为开仓（LONG/SHORT），必须计算满足交易所**最小开仓量**所需的最低杠杆。
      *   **计算公式**: `所需杠杆 = (最小开仓名义价值) / (当前可用保证金 * 0.95)`
      *   **注意**: 这里使用的是"可用保证金"（availEq），不是"账户总资产"。可用保证金为0时无法开新仓。
      *   **决策逻辑**:
        *   计算出`所需杠杆`后，向上取整（例如 2.1→3）。
        *   若`所需杠杆` > 5，则最终决策为 `HOLD`，并在 `reasoning` 明确“资金不足，5倍杠杆亦无法满足最小开仓量”。
        *   否则，设定 `最终杠杆 = max(向上取整后的所需杠杆, {default_leverage})`，并在 `trade_params` 中使用该杠杆值。
    *   **信号冲突处理**: 当内部策略信号冲突时，参考 1H 趋势结构；若无明确方向则 `HOLD`。
    *   **仓位规模**: 默认使用 **100% 可用资金** 进行开仓，除非风险评估明确要求减仓；需在 reasoning 中说明任何减仓理由。
    *   **极端行情处理**: 极端恐慌/狂热时，无持仓→`HOLD`；有持仓→结合盈利与 1H 趋势是否衰减决定平仓。

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
  - **重要澄清**: 若“内部量化模型矩阵”的信号为 `HOLD` 且当前为“无持仓”，这表示“没有入场信号（保持空仓）”，而不是“继续持有已有仓位”。严禁将“无入场信号”解读为“维持仓位”。

# 信息解读
- **内部量化模型矩阵**: 多策略共振提升置信度；冲突→`HOLD` 或减弱杠杆。
- **VLM K线分析 (1H)**: 唯一技术图形来源；用于趋势、结构、关键位、动量与潜在入/出场窗口。
    - **操作建议解释**: 若 1H 图给出"做多"且条件为"价格高于X"，需验证当前价是否已满足；否则应 `HOLD` 并等待。
    - **HOLD 语义**: 若当前无持仓，`HOLD` 表示空仓观望；若当前有持仓，`HOLD` 表示不加减仓但保留止盈止损计划。
- **社交媒体与新闻情报**: 
    - **TruthSocial 帖子（极高优先级）**: 来自特朗普(@realDonaldTrump)等关键政治人物的政策声明可能对BTC价格产生重大且快速的影响。请特别关注加密货币相关政策、经济政策或监管态度。如果出现重大政策声明，应优先考虑其对市场的潜在影响。
    - **CoinDesk 新闻**: 专业市场新闻，提供市场动态和行业分析。

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
1.  **识别核心机会**: 结合 1H VLM 分析与新闻判断是否存在明确多空机会；证据不足或冲突→`HOLD`。若为 `SHORT`，列出满足的“空头门槛”条目。
2.  **评估持仓状态**: 确定当前是空仓、持有多仓还是空仓，并计算盈亏。
3.  **整合辅助信号**: 使用内部量化模型和宏观K线分析（日线）来验证或微调决策。
4.  **风险评估与参数设定**: 基于市场波动性和信号强度，设定合理的杠杆、止盈和止损。

# JSON输出格式
{{
  "decision": "LONG/SHORT/HOLD/CLOSE_LONG/CLOSE_SHORT",
  "reasoning": "详细说明你做出决策的完整逻辑链。首先陈述当前持仓和盈亏状况，然后分析核心机会（VLM和新闻），接着整合辅助信号，最后基于风险评估得出结论。务必体现你对'核心原则'和'分析框架'的遵守。",
  "key_signals_detected": "明确列出本次决策所依据的最关键信号（多/空信号或风险信号）；如果没有，请填写'无特别关键信号'。",
  "confidence": "请根据当前与历史信号自动判断置信度，范围0-1，不要固定。",
  "suggested_trade_size": 1.0,
  "trade_params": {{
    "leverage": {default_leverage},
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
            # 对 HOLD 进行更明确的语义说明，避免与“继续持有”混淆
            if str(signal_type).upper() == 'HOLD':
                quant_part += f"- **信号类型**: **{signal_type}** （无入场信号：若当前无持仓则保持空仓；若已有持仓则维持不加不减）\n"
            else:
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

        # 分离 TruthSocial 和 CoinDesk 数据
        truthsocial_items = [item for item in twitter_data if 'TruthSocial' in item.get('source', '')]
        coindesk_items = [item for item in twitter_data if 'TruthSocial' not in item.get('source', '')]
        
        # 1. TruthSocial 帖子（特别重要，尤其是特朗普的政策声明）
        if truthsocial_items:
            twitter_part += "### 1.1 TruthSocial 关键账号帖子 (极高优先级)\n"
            twitter_part += "**重要提示**: TruthSocial帖子，特别是来自特朗普(@realDonaldTrump)等关键政治人物的政策声明，可能对BTC价格产生重大影响。请仔细分析这些帖子的内容，特别关注：\n"
            twitter_part += "- 加密货币相关政策声明\n"
            twitter_part += "- 经济政策或监管态度\n"
            twitter_part += "- 可能影响市场情绪的重大声明\n\n"
            for item in truthsocial_items:
                twitter_part += self._format_tweet(item)
        
        # 2. CoinDesk 新闻
        if coindesk_items:
            twitter_part += "\n### 1.2 CoinDesk 最新市场新闻 (高优先级)\n"
            twitter_part += "分析以下来自CoinDesk的最新市场新闻：\n"
            for news_item in coindesk_items:
                twitter_part += self._format_tweet(news_item)
        
        twitter_part += "\n"
        return twitter_part

    def _format_kline_analysis(self, kline_analysis_dict: Dict[str, Optional[str]]) -> str:
        """格式化K线分析部分（仅1H）。"""
        kline_part = "## 3. K线图技术分析 (1H)\n"
        analysis = kline_analysis_dict.get('1h') or kline_analysis_dict.get('short_term')
        if analysis and isinstance(analysis, str) and analysis.strip():
            return kline_part + f"\n### 1H 技术分析\n{analysis.strip()}\n"
        return kline_part + "\n无有效的1H K线图分析结果。"

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
        """格式化当前可用保证金信息。"""
        balance_part = "## 2. 当前可用保证金\n\n"
        if balance is not None:
            balance_part += f"当前可用保证金: **${balance:.2f} USDT**\n"
            balance_part += f"可用于交易的资金: **${balance * 0.95:.2f} USDT** (95%)\n"
            balance_part += "\n**重要概念澄清**：\n"
            balance_part += "- **可用保证金**（availEq）：可用于开新仓的资金\n"
            balance_part += "- **账户总资产**（equity）：账户总价值（包含持仓价值）\n"
            balance_part += "- **关键区别**：可用保证金为0 ≠ 账户总资产为0\n"
            balance_part += "- 如果所有资金都被用于持仓，可用保证金可能为0，但账户总资产（包含持仓价值）可能不为0\n"
            if balance <= 0:
                balance_part += "\n**当前状态**：可用保证金为0，**开仓操作（LONG/SHORT）无法执行**，但**平仓操作（CLOSE_LONG/CLOSE_SHORT）不受影响，可以正常执行**。平仓是减少持仓，不需要可用保证金。\n"
        else:
            balance_part += "无法获取当前可用保证金信息。\n"
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