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

    基于 OpenAI SDK，base_url 配置为 `https://jeniya.cn/v1`，默认 model 为 `gemini-3-pro-preview`（可在配置中覆盖）。
    若调用失败，可在上层启用回退逻辑：走“VLM分析 + DeepSeek决策”老路径。
    """

    def __init__(self):
        gemini_config = config.API_KEYS.get('gemini', {})
        self.base_url = gemini_config.get('base_url')
        self.api_key = gemini_config.get('api_key')
        # 默认模型改为 gemini-3-pro-preview，可通过环境变量覆盖
        self.model = gemini_config.get('model', 'gemini-3-pro-preview')

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
        
        # 分离 TruthSocial 和 CoinDesk 数据
        truthsocial_items = [item for item in twitter_data if 'TruthSocial' in item.get('source', '')]
        coindesk_items = [item for item in twitter_data if 'TruthSocial' not in item.get('source', '')]
        
        # 1. TruthSocial 帖子（特别重要，尤其是特朗普的政策声明）
        if truthsocial_items:
            part += "### 1.1 TruthSocial 关键账号帖子 (极高优先级)\n"
            part += "**重要提示**: TruthSocial帖子，特别是来自特朗普(@realDonaldTrump)等关键政治人物的政策声明，可能对BTC价格产生重大影响。请仔细分析这些帖子的内容，特别关注：\n"
            part += "- 加密货币相关政策声明\n"
            part += "- 经济政策或监管态度\n"
            part += "- 可能影响市场情绪的重大声明\n\n"
            for item in truthsocial_items:
                source = item.get('source', 'TruthSocial')
                username = item.get('username', '')
                title = (item.get('text') or '').replace('\n', ' ').strip() or '无标题'
                desc = (item.get('description') or '').replace('\n', ' ').strip()
                created_at = item.get('created_at', '未知时间')
                url = item.get('url', '')
                part += f"\n**来自: {source}**"
                if username:
                    part += f" (@{username})"
                part += f" | {created_at}\n"
                part += f"**内容**: {title}\n"
                if desc and desc != '无摘要':
                    part += f"**摘要**: {desc}\n"
                if url:
                    part += f"**链接**: {url}\n"
                part += "-" * 50 + "\n"
        
        # 2. CoinDesk 新闻
        if coindesk_items:
            part += "\n### 1.2 CoinDesk 最新市场新闻 (高优先级)\n"
            part += "分析以下来自CoinDesk的最新市场新闻：\n"
            for item in coindesk_items:
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
            # 标注市场形态：TREND 或 RANGE（震荡）
            "market_regime": "TREND/RANGE",
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

**特别提醒**：如果新闻情报中包含 TruthSocial 帖子（特别是来自特朗普等关键政治人物的声明），请给予极高优先级。这些政策声明可能对BTC价格产生重大且快速的影响。请仔细分析这些帖子的内容，特别关注加密货币相关政策、经济政策或监管态度。

# 分析与约束
0. **市场形态判定（强制）**：在给出任何交易建议前，先判断当前市场是 **趋势（TREND）** 还是 **震荡/区间（RANGE）**。若判断为 `RANGE`（震荡），**优先采用区间反转策略（在支撑/阻力附近做反转）而不是突破策略**；若判断为 `TREND`，以顺势突破/追随趋势为主。请在输出 JSON 中返回 `market_regime` 字段（取值：\"TREND\" 或 \"RANGE\"），并在 `reasoning` 中明确说明判定依据与位置（支撑/阻力/均线）。
1. 仅基于 1H 结构与动量把握 1-6 小时机会；证据不足或冲突→`HOLD`。
2. **资金优先 (硬性要求)**：若开仓（LONG/SHORT），计算满足交易所**最小开仓名义价值**所需最低杠杆：
   所需杠杆 = (最小开仓名义价值) / (当前可用保证金 * 0.95)。向上取整；若 >5 则改为 `HOLD` 并在 reasoning 标注原因。否则，设定最终杠杆 = max(向上取整后的所需杠杆, {default_leverage})，写入 trade_params。
   注意：这里使用的是"可用保证金"（availEq），不是"账户总资产"。可用保证金为0时无法开新仓。
3. **仓位规模**：默认使用 **100% 可用资金**（即 suggested_trade_size = 1.0）进行开仓，除非风险评估明确要求减仓；任何减仓需在 reasoning 中说明原因。

4. **盈利锁定与动能衰竭（最高优先级硬规则）**：
   - 该规则只用于“已有持仓”的**离场/减仓**决策，优先级高于任何“等待突破/等待确认/再观察一根K线”的续持逻辑。
   - **严禁**在触发动能衰竭后输出“等待下一次突破/破位确认”。
   - 若当前持有 **多单（LONG）且处于浮盈**：一旦你判断“上涨动能已明显衰竭/冲高受阻”，必须直接输出 `decision = CLOSE_LONG`（或至少在 reasoning 中说明应立即减仓锁盈，但默认应 CLOSE_LONG）。
     典型衰竭证据（满足任意两项或你认为证据足够的组合即可）：RSI顶背离或高位回落、量价背离（创新高但量能不增）、连续上冲失败与长上影/冲高回落扩大、布林上轨附近走弱并回落到上轨/中轨、均线斜率明显变平。
   - 若当前持有 **空单（SHORT）且处于浮盈**：一旦你判断“下跌动能已明显衰竭/砸盘不延续”，必须直接输出 `decision = CLOSE_SHORT`（或至少在 reasoning 中说明应立即减仓锁盈，但默认应 CLOSE_SHORT）。
     典型衰竭证据（满足任意两项或你认为证据足够的组合即可）：RSI底背离或低位回升、下破不延续/跌无量、连续下探失败与长下影/急跌后快速反抽、触及/跌破布林下轨后收回、重新站回关键均线/中轨附近。

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
    *   **分析框架**: 未来1-2天中线 + 1-6 小时短线；证据不充分或信号冲突→`HOLD`。
    *   **量化模型的作用（重要）**: 量化模型是辅助。**当量化为 HOLD 时，只有在图表与技术指标出现“强一致、多重共振”的明确信号才可逆势给出开仓；若图表信号也不强或存在矛盾，应保持 HOLD。** 量化发出 BUY/SELL 时可作为加分。
    *   **持仓与盈亏评估（重要）**: **在做出任何决策前，你必须首先明确当前持仓状态和盈亏情况**。如果当前有持仓，请仔细查看"未实现盈亏"字段：
        - **盈利持仓**: 当持仓处于盈利状态时，应结合技术分析判断是否应该止盈离场，或继续持有以获取更大利润。
        - **亏损持仓**: 当持仓处于亏损状态时，应评估是否触及止损位，或是否出现反向信号需要及时止损。**特别注意：对于空头持仓，价格上涨意味着亏损；对于多头持仓，价格下跌意味着亏损。**
        - **盈亏计算**: 系统已提供准确的未实现盈亏（USDT），你无需自己计算，直接使用即可。
    *   **风险与机遇**: 风险优先，信号模糊/矛盾→`HOLD`；只有形成共振才开仓。
    *   **顺势与逆势**: 首选顺势。逆势（抄底/摸顶）仅在出现放量长影+收回关键均线或明显背离时考虑，并使用更紧的止损。
    *   **极端位置禁忌（避免追涨杀跌/陷阱）**：
        - 价格处于布林带上轨外或 RSI>70 且远离 MA20 时，禁止追多；等待回踩中轨/MA 或二次确认。
        - 价格处于布林带下轨外或 RSI<30 且远离 MA20 时，禁止追空；等待反抽失败或二次跌破确认。
        - 大幅单边后第一次急拉/急砸形成 V 形时，不在尾部追单，需等回踩/二次信号。
    *   **确认要求**: 突破类交易需看到收盘站稳+量能放大；逆势单需放量长影并收复均线；否则 `HOLD`。
    *   **盈利锁定（硬性要求）**: 如果当前存在持仓且处于浮盈，一旦你判断出现“趋势动能衰竭/冲高受阻/下破不延续”等离场信号，应**立即**选择平仓锁定盈利：
        - 多单浮盈 + 上涨动能衰竭 → 输出 `CLOSE_LONG`（不得等待下一次突破确认）。
        - 空单浮盈 + 下跌动能衰竭 → 输出 `CLOSE_SHORT`（不得等待下一次破位确认）。
        - 该规则优先级高于“等待确认/再观察”的续持逻辑。
    *   **模型自主性**: 在遵守上述核心原则的基础上，你可以运用你强大的分析能力，在规则之外寻找并评估独特的交易机会。如果数据和模式强烈支持一个非传统策略，你可以提出该决策，但必须在理由中详细阐述你的逻辑和风险控制方案。**记住：你的图像识别和技术分析能力是核心优势，不要被量化模型的HOLD信号束缚。**
    *   **具体参数**: 如果决策是 'LONG' 或 'SHORT'，必须提供明确的止盈（take_profit）和止损（stop_loss）价格。

2.  **输出格式**:
    *   **必须严格遵循下面的JSON格式**，不得添加任何额外解释或注释。
    *   **所有输出内容，特别是 `reasoning`, `key_signals`, 和 `risk_assessment` 字段，必须使用【中文】进行说明。**

    ```json
    {{
        "decision": "...", // 必须是 'LONG', 'SHORT', 'HOLD', 'CLOSE_LONG' 或 'CLOSE_SHORT' 其中之一
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

        # 目前上游 Gemini 端对该模型的最大 completion tokens 约为 128k，
        # 这里设置一个足够大的安全上限（如 8000），避免出现 invalid_value 错误。
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": img_url, "detail": "high"}},
                    ],
                }
            ],
            temperature=0.4,
            max_tokens=8000,
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


