import base64
import json
import mimetypes
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from btc_predictor.utils import LOGGER


class UnifiedGeminiAnalyzer:
    """
    使用 Gemini (OpenAI兼容) 的多模态能力，合并 VLM 与 LLM 决策：
    - 直接输入 1H K线图（图片）+ 量化信号 + 新闻情报 + 持仓与余额
    - 输出与 DeepSeekAnalyzer 相同结构的交易决策 JSON

    基于 OpenAI SDK，base_url 配置为 `https://jeniya.cn/v1`，model 为 `gemini-2.5-pro`（可经环境变量覆盖）。
    若调用失败，可在上层启用回退逻辑：走“VLM分析 + DeepSeek决策”老路径。
    """

    def __init__(self):
        gemini_config = config.API_KEYS.get('gemini', {})
        self.base_url = gemini_config.get('base_url')
        self.api_key = gemini_config.get('api_key')
        self.model = gemini_config.get('model', 'gemini-3-pro-preview-thinking')

        if not all([self.base_url, self.api_key, self.model]):
            raise ValueError("Gemini API的配置不完整 (base_url, api_key, model)。")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _parse_llm_json_response(self, response_text: str) -> Dict[str, Any]:
        if not response_text:
            raise ValueError("LLM返回了空内容。")

        cleaned_text = response_text.strip()
        cleaned_text = re.sub(r'^```json\s*', '', cleaned_text)
        cleaned_text = re.sub(r'\s*```$', '', cleaned_text)

        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass

        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, cleaned_text, re.DOTALL)
        for json_str in reversed(matches):
            try:
                obj = json.loads(re.sub(r'\s+', ' ', json_str.strip()))
                return obj
            except json.JSONDecodeError:
                continue
        
        print(f"--- 完整的原始响应 ---\n{response_text}\n--------------------")
        raise ValueError(f"无法解析JSON。原始响应片段: {response_text[:200]}...")

    def _encode_image(self, image_path: str) -> Tuple[str, str]:
        with open(image_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        mime = mimetypes.guess_type(image_path)[0] or 'image/png'
        return b64, mime

    def _format_quant_signals(self, quant_signals: List[Dict[str, Any]]) -> str:
        if not quant_signals:
            return "### 内部量化模型矩阵\n- 未提供任何量化信号。\n"
        text = "### 内部量化模型矩阵\n"
        for s in quant_signals:
            strategy = s.get('strategy', '未知策略')
            signal = s.get('signal', 'HOLD')
            info = s.get('action') or s.get('info', '无详细信息。')
            current_price = s.get('current_price')
            text += f"\n#### 策略: {strategy}\n"
            if str(signal).upper() == 'HOLD':
                text += "- **信号类型**: **HOLD** （无入场信号：无持仓→保持空仓；有持仓→不加不减）\n"
            else:
                text += f"- **信号类型**: **{signal}**\n"
            text += f"- **策略分析**: {info}\n"
            if current_price:
                try:
                    text += f"- **参考价格**: ${float(current_price):.2f}\n"
                except Exception:
                    text += f"- **参考价格**: {current_price}\n"
        return text

    def _format_news(self, twitter_data: List[Dict[str, Any]]) -> str:
        part = "## 1. 社交媒体与新闻情报\n\n"
        if not twitter_data:
            return part + "无有效的社交媒体或新闻情报。\n"
        part += "### 1.1 CoinDesk 最新市场新闻 (高优先级)\n分析以下来自CoinDesk的最新市场新闻：\n"
        for item in twitter_data:
            source = item.get('source', 'CoinDesk')
            title = (item.get('text') or '').replace('\n', ' ').strip() or '无标题'
            desc = (item.get('description') or '').replace('\n', ' ').strip()
            created_at = item.get('created_at', '未知时间')
            part += f"\n**来自: {source}** | {created_at}\n"
            part += f"**标题**: {title}\n"
            if desc and desc != '无摘要':
                part += f"**摘要**: {desc}\n"
            part += "-" * 50 + "\n"
        return part

    def _build_prompt(self, quant_signals: List[Dict[str, Any]], twitter_data: List[Dict[str, Any]], current_position: Optional[Dict[str, Any]], current_balance: Optional[float], symbol: str) -> str:
        from datetime import datetime, timezone
        current_time_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        asset_name = symbol.split('-')[0] if symbol else '资产'
        market_price = None
        if quant_signals and quant_signals[0].get('current_price') is not None:
            market_price = quant_signals[0].get('current_price')

        position_text = "## 1. 当前持仓状态\n\n"
        if not current_position or not current_position.get('posSide'):
            position_text += "当前**无持仓**。\n"
        else:
            side = current_position.get('posSide')
            qty = current_position.get('posCcy') or current_position.get('pos')
            avg_price = current_position.get('avgPx')
            leverage = current_position.get('lever')
            unrealized_pnl = current_position.get('upl')
            if side == 'net':
                try:
                    pos_val = float(current_position.get('pos', 0))
                except Exception:
                    pos_val = 0
                if pos_val > 0:
                    position_text += f"- **持仓方向**: **做多 (LONG, net模式, pos={pos_val})**\n"
                elif pos_val < 0:
                    position_text += f"- **持仓方向**: **做空 (SHORT, net模式, pos={pos_val})**\n"
                else:
                    position_text += "当前**无持仓**。\n"
            elif side == 'long':
                position_text += f"- **持仓方向**: **做多 (LONG)**\n"
            elif side == 'short':
                position_text += f"- **持仓方向**: **做空 (SHORT)**\n"
            position_text += f"- **持仓数量**: {qty} {asset_name}\n"
            position_text += f"- **开仓均价**: ${avg_price}\n"
            position_text += f"- **杠杆倍数**: {leverage}x\n"
            if unrealized_pnl is not None:
                try:
                    pnl_value = float(unrealized_pnl)
                    pnl_sign = "+" if pnl_value >= 0 else ""
                    position_text += f"- **未实现盈亏**: **{pnl_sign}${pnl_value:.2f} USDT**\n"
                except Exception:
                    position_text += f"- **未实现盈亏**: ${unrealized_pnl} USDT\n"

        balance_text = "## 2. 当前可用保证金\n\n"
        if current_balance is not None:
            balance_text += f"当前可用保证金: **${current_balance:.2f} USDT**\n可用于交易的资金: **${current_balance * 0.95:.2f} USDT** (95%)\n"
            balance_text += "\n**重要概念澄清**：\n"
            balance_text += "- **可用保证金**（availEq）：可用于开新仓的资金\n"
            balance_text += "- **账户总资产**（equity）：账户总价值（包含持仓价值）\n"
            balance_text += "- **关键区别**：可用保证金为0 ≠ 账户总资产为0\n"
            balance_text += "- 如果所有资金都被用于持仓，可用保证金可能为0，但账户总资产（包含持仓价值）可能不为0\n"
            if current_balance <= 0:
                balance_text += "\n**当前状态**：可用保证金为0，**开仓操作（LONG/SHORT）无法执行**，但**平仓操作（CLOSE_LONG/CLOSE_SHORT）不受影响，可以正常执行**。平仓是减少持仓，不需要可用保证金。\n"
        else:
            balance_text += "无法获取当前可用保证金信息。\n"

        qs_text = self._format_quant_signals(quant_signals)
        news_text = self._format_news(twitter_data)

        market_price_text = f"${market_price:.2f}" if isinstance(market_price, (int, float)) else "未知"

        # 使用程序化方式构造 JSON 模板，避免 f-string 花括号转义问题
        default_leverage = int(getattr(config, 'FUTURES', {}).get('leverage', 3))

        json_template = {
            "decision": "LONG/SHORT/HOLD/CLOSE_LONG/CLOSE_SHORT",
            "reasoning": "详述完整逻辑链。**必须首先明确说明当前持仓状态和未实现盈亏情况**（如果有持仓），然后分析图像结构、量化信号与新闻，最后给出风险与参数。",
            "key_signals_detected": "列出本次决策的关键多/空/风险信号。没有则写无。",
            "confidence": 0.0,
            "suggested_trade_size": 1.0,
            "trade_params": {
                "leverage": default_leverage,
                "take_profit_pct": 8.0,
                "stop_loss_pct": 4.0
            },
            "risk_assessment": "对本次交易潜在风险的简要评估与风控措施"
        }
        json_template_text = json.dumps(json_template, ensure_ascii=False, indent=2)

        prompt = f"""
# 角色
你是一名顶级的加密货币**期货**短线交易策略师，当前策略为单一 **1小时 (1H)** 时间框的快进快出操作。你必须结合图像与文本做出可执行决策。你正在为 **{asset_name}** 做决策。

- **当前时间(UTC)**: {current_time_utc}
- **市场价格**: {market_price_text}

# 输入
（1）一张 1H K线图（在本消息中作为图片内容附带）。
（2）以下结构化信息：
{position_text}
{balance_text}
{news_text}
{qs_text}

# 分析与约束
1. 仅基于 1H 结构与动量把握 1-6 小时机会；证据不足或冲突→`HOLD`。
2. **资金优先 (硬性要求)**：若开仓（LONG/SHORT），计算满足交易所**最小开仓名义价值**所需最低杠杆：
   所需杠杆 = (最小开仓名义价值) / (当前可用保证金 * 0.95)。向上取整；若 >5 则改为 `HOLD` 并在 reasoning 标注原因。否则，设定最终杠杆 = max(向上取整后的所需杠杆, {default_leverage})，写入 trade_params。
   注意：这里使用的是"可用保证金"（availEq），不是"账户总资产"。可用保证金为0时无法开新仓。
3. **仓位规模**：默认使用 **100% 可用资金**（即 suggested_trade_size = 1.0）进行开仓，除非风险评估明确要求减仓；任何减仓需在 reasoning 中说明原因。

# 输出JSON（严格JSON，不要多余文字）
{json_template_text}
""".strip()
        return prompt

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_trade_decision_unified(
        self,
        quant_signals: List[Dict[str, Any]],
        twitter_data: List[Dict[str, Any]],
        kline_image_path: str,
        timeframe: str = '1h',
        current_position: Optional[Dict[str, Any]] = None,
        current_balance: Optional[float] = None,
        symbol: str = 'BTC-USDT-SWAP',
    ) -> Dict[str, Any]:
        """
        使用单次多模态请求（图像+文本）得到最终决策。输出结构与 DeepSeekAnalyzer 保持一致。
        """
        LOGGER.info(f"[UnifiedGemini] 启动统一多模态决策，请求模型: {self.model}")

        # 1. 构建系统提示
        system_prompt = f"""
你是一个专业的加密货币（特别是BTC比特币）期货交易决策系统。你的核心任务是分析K线图、技术指标、市场新闻和内部量化模型信号，为用户提供精确、审慎的交易决策。

**请严格遵守以下指令:**

1.  **决策逻辑**:
    *   **分析框架**: 你的分析应着眼于未来1-2天的中线趋势，并结合短线信号来确定精确的入场和出场点。**你的主要决策依据应该是K线图形态、关键技术指标（如MA, MACD, RSI）和市场新闻情绪。量化模型信号只是参考信息之一，不应过度依赖。**
    *   **量化模型的作用（重要）**: 量化模型信号仅作为**辅助参考**，不是决策的主要依据。**如果你基于K线形态、技术指标和市场情绪分析后，认为应该开仓（LONG或SHORT），即使量化模型显示HOLD，你也应该给出开仓决策。**量化模型可能滞后或无法捕捉某些市场机会，你的图像识别和技术分析能力是更重要的决策工具。
    *   **持仓与盈亏评估（重要）**: **在做出任何决策前，你必须首先明确当前持仓状态和盈亏情况**。如果当前有持仓，请仔细查看"未实现盈亏"字段：
        - **盈利持仓**: 当持仓处于盈利状态时，应结合技术分析判断是否应该止盈离场，或继续持有以获取更大利润。
        - **亏损持仓**: 当持仓处于亏损状态时，应评估是否触及止损位，或是否出现反向信号需要及时止损。**特别注意：对于空头持仓，价格上涨意味着亏损；对于多头持仓，价格下跌意味着亏损。**
        - **盈亏计算**: 系统已提供准确的未实现盈亏（USDT），你无需自己计算，直接使用即可。
    *   **风险与机遇**: "风险第一"是核心原则。在信号模糊时，保持空仓（HOLD）是明智的。**然而，当图表形态、技术指标和市场情绪等信息高度一致，形成共振时，你应该更果断地给出 'LONG' 或 'SHORT' 决策，即使量化模型显示HOLD。**
    *   **顺势与逆势**: 基本原则是顺势而为。**但你也要学会识别潜在的趋势反转点。当价格在关键支撑位出现放量企稳、或出现看涨背离等强力反转信号时，可以谨慎地尝试"抄底"（LONG）。反之，在关键阻力位出现滞涨、看跌背离时，可以尝试"摸顶"（SHORT）。所有逆势交易都必须有明确的信号支持，并设置更严格的止损。**
    *   **模型自主性**: 在遵守上述核心原则的基础上，你可以运用你强大的分析能力，在规则之外寻找并评估独特的交易机会。如果数据和模式强烈支持一个非传统策略，你可以提出该决策，但必须在理由中详细阐述你的逻辑和风险控制方案。**记住：你的图像识别和技术分析能力是核心优势，不要被量化模型的HOLD信号束缚。**
    *   **具体参数**: 如果决策是 'LONG' 或 'SHORT'，必须提供明确的止盈（take_profit）和止损（stop_loss）价格。

2.  **输出格式**:
    *   **必须严格遵循下面的JSON格式**，不得添加任何额外解释或注释。
    *   **所有输出内容，特别是 `reasoning`, `key_signals`, 和 `risk_assessment` 字段，必须使用【中文】进行说明。**

    ```json
    {{
        "decision": "...", // 必须是 'LONG', 'SHORT', 'HOLD' 或 'CLOSE' 其中之一
        "trade_parameters": {{
            "leverage": "...", // 例如 '10x', 如果是 'HOLD' 或 'CLOSE' 则为 'None'
            "take_profit": "...", // 价格, 如果是 'HOLD' 或 'CLOSE' 则为 'N/A'
            "stop_loss": "..." // 价格, 如果是 'HOLD' 或 'CLOSE' 则为 'N/A'
        }},
        "reasoning": "...", // 详细的中文决策逻辑和市场分析
        "key_signals": [
            "..." // 关键信号列表（中文）
        ],
        "risk_assessment": "..." // 风险评估（中文）
    }}
    ```
"""

        # 2. 构建用户提示
        user_prompt = self._build_prompt(quant_signals, twitter_data, current_position, current_balance, symbol)
        img_b64, mime = self._encode_image(kline_image_path)
        img_url = f"data:{mime};base64,{img_b64}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": img_url, "detail": "high"}},
                    ],
                }
            ],
            temperature=0.4,
            max_tokens=1048576,
            stream=False,
            response_format={"type": "json_object"},
        )

        # 优先使用 SDK 在 json_object 模式下可能提供的已解析结果；否则回退到文本解析
        choice = response.choices[0].message
        try:
            if hasattr(choice, 'parsed') and getattr(choice, 'parsed') is not None:  # type: ignore[attr-defined]
                parsed = getattr(choice, 'parsed')  # type: ignore[assignment]
            else:
                raw = choice.content
                if isinstance(raw, (dict, list)):
                    parsed = raw  # type: ignore[assignment]
                else:
                    parsed = self._parse_llm_json_response(raw or "")
        except Exception as parse_err:
            # 解析失败时，返回包含完整原始响应的回退决策，便于定位问题
            try:
                raw_text = choice.content if isinstance(choice.content, str) else str(choice.content)
            except Exception:
                raw_text = None
            LOGGER.warning(
                f"[UnifiedGemini] JSON解析失败: {type(parse_err).__name__}: {parse_err}. 将返回原始响应文本以便诊断。",
                exc_info=True,
            )
            parsed = {
                "decision": "HOLD",
                "reasoning": "LLM JSON解析失败，已返回原始响应以便诊断。",
                "key_signals_detected": "无关键风险信号",
                "confidence": 0.0,
                "suggested_trade_size": 0.95,
                "trade_params": {},
                "raw_response_text": raw_text,
                "parse_error": f"{type(parse_err).__name__}: {parse_err}",
            }
        # 最低字段保障
        parsed.setdefault('key_signals_detected', '无关键风险信号')
        parsed.setdefault('trade_params', {})
        return parsed


