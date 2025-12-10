import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from btc_predictor.utils import LOGGER


class GPTReviewer:
    """
    使用 gpt-5.1 对统一 Gemini 决策进行二次审核。
    """

    def __init__(self) -> None:
        api_key = (
            os.getenv("GPT_REVIEW_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("GPT reviewer 缺少 API key (GPT_REVIEW_API_KEY / OPENAI_API_KEY)。")

        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("GPT_REVIEW_BASE_URL") or None,
        )
        # 审核模型改为 Gemini 3 预览（可通过环境变量覆盖）
        self.model = os.getenv("GPT_REVIEW_MODEL", "gemini-3-pro-preview")

    def review(
        self,
        decision: Dict[str, Any],
        kline_image_path: str,
        quant_signals: List[Dict[str, Any]],
        twitter_data: List[Dict[str, Any]],
        current_position: Optional[Dict[str, Any]],
        current_balance: Optional[float],
    ) -> Dict[str, Any]:
        """
        返回审核结果 JSON：{verdict, issues, recommendation}
        """
        prompt = f"""
你是风险审核员，对上一轮 Gemini 多模态决策进行复核。

要求：
- 仅基于提供的决策/量化信号/新闻/持仓与余额，检视是否存在追涨杀跌、逆势抄底、缺少确认、止损过宽/缺失等问题。
- 重点检查：1) 是否在布林带上轨外或 RSI 高位追多；2) 下轨外或 RSI 低位追空；3) 未等回踩/二次确认的突破追单；4) 逆势抄底/摸顶缺少放量长影+收复均线；5) 止损是否设置且不合理过宽。
- 若发现明显风险，给出简洁中文要点。
- 输出严格 JSON：
{{
  "verdict": "PASS" | "WARN" | "BLOCK",
  "issues": ["..."],
  "recommendation": "..."
}}

当前决策：{decision}
量化信号：{quant_signals}
新闻情报：{twitter_data}
持仓：{current_position}
可用保证金：{current_balance}
        """.strip()

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=800,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            choice = resp.choices[0].message
            if isinstance(choice.content, dict):
                return choice.content  # type: ignore[return-value]
            if isinstance(choice.content, str):
                import json

                return json.loads(choice.content)
            raise ValueError("未知的审核返回格式")
        except Exception as e:
            LOGGER.warning(f"[GPTReviewer] 审核失败: {type(e).__name__}: {e}")
            raise

