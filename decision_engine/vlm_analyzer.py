# decision_engine/vlm_analyzer.py
import os
import requests
import base64
import mimetypes
from typing import List, Dict, Any, Optional
import json # 添加用于详细错误日志记录
import hashlib # 添加用于K线图缓存哈希
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

import config
from btc_predictor.utils import LOGGER
from .vlm_cache import VLMCache # 导入缓存管理器

# --- Sanity Check Helper ---
def _is_sane(text: str) -> bool:
    """
    对VLM的输出进行基本的理智检查，防止乱码污染下游。
    """
    if not isinstance(text, str) or not text.strip():
        return False
    
    # 检查是否包含多种语言的特征词 (这是一个启发式方法)
    forbidden_keywords = [
        'architects', 'settle', 'gracias', 'спасибо', 'ありがとうございました', '안녕하세요'
    ]
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in forbidden_keywords):
        return False
        
    # 检查是否有过多不常见的非ASCII字符（排除中文）
    import re
    non_ascii_chars = re.findall(r'[^\u0000-\u007F\u4e00-\u9fa5\u3000-\u303F\uff00-\uffef]', text)
    if len(non_ascii_chars) > 10: # 允许少量特殊符号
        return False

    return True

class VLMAnalyzer:
    """
    一个封装了视觉语言模型（VLM）分析功能的类。
    支持为不同任务使用不同的模型，并具有智能缓存机制。
    """
    def __init__(self):
        deepseek_config = config.API_KEYS.get('deepseek', {})
        self.api_url = (deepseek_config.get('base_url') or "https://api.deepseek.com/v1").rstrip('/') + "/chat/completions"
        self.api_key = deepseek_config.get('api_key')
        
        # 超时设置（秒）
        self.request_timeout_seconds = 300  # VLM请求超时，默认5分钟
        self.download_timeout_seconds = 60   # 媒体下载超时，默认60秒
        # 连接与流式读取超时（更细粒度控制）
        self.request_connect_timeout_seconds = 30  # 连接超时
        self.stream_read_timeout_seconds = 120     # 流式读取超时：无数据超过此值即超时
        # 流式重试与首字超时
        self.stream_max_retries = 3               # 流式整体重试次数（首字/中途卡住都会重试）
        self.first_token_timeout_seconds = 60     # 首字/中途最长等待时间（单次流式读），超过则重试
        
        # 是否启用流式输出（SSE），默认关闭。开启后将边到边打印到控制台
        self.stream: bool = True
        
        # 为不同任务定义不同的模型
        self.kline_model = "stepfun-ai/step3"  # K线图分析模型
        self.tweet_model = "stepfun-ai/step3"  # 推文图片分析模型
        
        if not self.api_key or 'YOUR' in self.api_key:
            LOGGER.warning("VLM (DeepSeek) API key 未配置，VLM分析功能将被跳过。")
            self.api_key = None

        self.session = requests.Session()
        # proxy_url = config.DEFAULTS.get('proxy_url')
        # if proxy_url:
        #     self.session.proxies = {'http': proxy_url, 'https': proxy_url}
        #     LOGGER.info(f"VLMAnalyzer 已配置代理: {proxy_url}")

        # 初始化缓存管理器（使用配置文件设置）
        self.cache = VLMCache()
        
        # 启动时清理过期缓存
        self.cache.cleanup_expired_cache()
        
        # 显示缓存统计
        stats = self.cache.get_cache_stats()
        LOGGER.info(f"VLM缓存统计: 推文缓存 {stats['tweet_cache_count']} 条, K线图缓存 {stats['kline_cache_count']} 条")
        # 打印 VLM 单源模式开关日志
        try:
            rules = getattr(config, 'DECISION_RULES', {})
            vlm_solo = rules.get('vlm_solo_trade', True)
            LOGGER.info(f"VLM 单源模式: {'启用' if vlm_solo else '禁用'}")
        except Exception:
            pass

    def _download_media(self, url: str) -> Optional[Dict[str, Any]]:
        """下载媒体文件并返回字节和MIME类型。"""
        if not url:
            return None
        try:
            LOGGER.info(f"正在下载媒体文件: {url}")
            response = self.session.get(url, timeout=self.download_timeout_seconds, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type') or mimetypes.guess_type(url)[0]
            if not content_type or not (content_type.startswith('image/') or content_type.startswith('video/')):
                LOGGER.warning(f"下载的内容不是可识别的媒体类型: {content_type}")
                return None
            
            media_bytes = response.content
            LOGGER.success(f"媒体文件下载成功 ({len(media_bytes) / 1024:.2f} KB).")
            return {"bytes": media_bytes, "mime_type": content_type}
        except requests.exceptions.RequestException as e:
            LOGGER.error(f"下载媒体文件失败: {e}")
            return None

    def _analyze_with_vlm(self, base64_media: str, mime_type: str, prompt_text: str, model_name: str) -> Optional[str]:
        """通用的VLM分析函数，现在支持指定模型。"""
        if not self.api_key:
            return "VLM分析被跳过（API Key未配置）。"

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_name,  # 使用传入的模型名称
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_media}", "detail": "high"}}
            ]}],
            "max_tokens": 4096, # 提高最大生成长度，减少过早截断
            "temperature": 0.2 # 降低温度，减少幻觉和乱码
        }
        
        # 流式输出开关（兼容OpenAI/DeepSeek风格），若为True则返回SSE
        if self.stream:
            payload["stream"] = True
            # SSE常用头
            headers["Accept"] = "text/event-stream"
            headers["Cache-Control"] = "no-cache"
            headers["Connection"] = "keep-alive"

        # --- 代理设置处理 ---
        # 记录原始代理设置
        orig_http_proxy = os.environ.get('http_proxy')
        orig_https_proxy = os.environ.get('https_proxy')
        orig_HTTP_PROXY = os.environ.get('HTTP_PROXY')
        orig_HTTPS_PROXY = os.environ.get('HTTPS_PROXY')
        orig_session_proxies = getattr(self.session, 'proxies', None)

        # 清除所有代理设置
        for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            if key in os.environ:
                del os.environ[key]
        self.session.proxies = {}
        # 关闭从环境继承（如代理）
        orig_trust_env = getattr(self.session, 'trust_env', True)
        self.session.trust_env = False

        try:
            LOGGER.info(f"正在使用模型 {model_name} 进行VLM分析...")
            LOGGER.debug(f"VLM API URL: {self.api_url}")
            LOGGER.debug(f"Payload size: {len(json.dumps(payload))} chars")
            
            # 非流式/流式分别处理
            if not self.stream:
                # 使用(连接超时, 读取超时)的形式，避免整体阻塞
                response = self.post_with_retry(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=(self.request_connect_timeout_seconds, self.request_timeout_seconds),
                    stream_response=False,
                )
                LOGGER.info(f"VLM API响应状态码: {response.status_code}")
                response.raise_for_status()
                result = response.json()
                analysis = result['choices'][0]['message']['content'].strip()
                LOGGER.success(f"VLM分析成功接收（模型: {model_name}）。")
                LOGGER.info(f"VLM模型输出: {analysis}")
                return analysis
            else:
                # 流式：边接收边打印；加入首字和中途卡顿的自动重试
                for attempt_idx in range(1, self.stream_max_retries + 1):
                    got_any_token = False
                    try:
                        # 统一用较短的读取超时保障首字与中途卡顿都能尽快超时重试
                        response = self.post_with_retry(
                            self.api_url,
                            headers=headers,
                            json=payload,
                            timeout=(self.request_connect_timeout_seconds, self.first_token_timeout_seconds),
                            stream_response=True,
                        )
                        LOGGER.info(f"[try {attempt_idx}/{self.stream_max_retries}] VLM API流式响应状态码: {response.status_code}")
                        response.raise_for_status()
                        # 强制按UTF-8解析SSE，避免在Windows下出现乱码
                        advertised_encoding = getattr(response, 'encoding', None)
                        if advertised_encoding and advertised_encoding.lower() != 'utf-8':
                            LOGGER.debug(f"服务端宣称编码为: {advertised_encoding}，将强制使用UTF-8解析SSE。")
                        response.encoding = 'utf-8'

                        # 记录进度
                        import time
                        start_time = time.time()
                        first_token_time: Optional[float] = None
                        last_report_time = start_time
                        received_chars = 0
                        LOGGER.info("VLM流式开始接收...（将定期报告进度）")
                        final_text_chunks = []
                        # 有些模型（如 Thinking 系列）会把主体内容放在 reasoning_content
                        final_reasoning_chunks = []
                        print("[VLM Streaming] ", end="", flush=True)
                        reasoning_open = False
                        for line_bytes in response.iter_lines(decode_unicode=False):
                            if not line_bytes:  # 心跳/空行
                                continue
                            # 显式使用UTF-8解码
                            try:
                                line = line_bytes.decode('utf-8', errors='ignore')
                            except Exception:
                                # 兜底：按ISO-8859-1避免异常
                                line = line_bytes.decode('latin-1', errors='ignore')
                            if line.startswith('data:'):
                                data_str = line[len('data:'):].strip()
                                if data_str == "[DONE]":
                                    # 在流式结束时，若思考段仍未关闭，则补齐关闭标签
                                    if reasoning_open:
                                        print("</think>", end="", flush=True)
                                        reasoning_open = False
                                    break
                                try:
                                    evt = json.loads(data_str)
                                    # 兼容多种流式字段：choices[].delta.content 或 choices[].message.content
                                    content_delta = None
                                    reasoning_delta = None
                                    choices = evt.get('choices') or []
                                    if choices:
                                        choice0 = choices[0]
                                        delta = choice0.get('delta') or {}
                                        content_delta = delta.get('content')
                                        reasoning_delta = delta.get('reasoning_content')
                                        if content_delta is None:
                                            # 一些实现直接用message
                                            message = choice0.get('message') or {}
                                            content_delta = message.get('content')
                                    if content_delta:
                                        got_any_token = True
                                        # 若之前处于<think>段中，则先闭合
                                        if reasoning_open:
                                            print("</think>", end="", flush=True)
                                            reasoning_open = False
                                        print(content_delta, end="", flush=True)
                                        final_text_chunks.append(content_delta)
                                        received_chars += len(content_delta)
                                    if reasoning_delta:
                                        got_any_token = True
                                        # 推理内容以<think>标签包裹，首段打开，后续持续输出直到出现content或结束
                                        if not reasoning_open:
                                            print("<think>", end="", flush=True)
                                            reasoning_open = True
                                        print(reasoning_delta, end="", flush=True)
                                        final_reasoning_chunks.append(reasoning_delta)
                                        received_chars += len(reasoning_delta)
                                    # 首字延迟与周期性进度日志
                                    now = time.time()
                                    if first_token_time is None and received_chars > 0:
                                        first_token_time = now
                                        LOGGER.info(f"VLM首字延迟: {first_token_time - start_time:.1f}s")
                                    if now - last_report_time >= 15:
                                        elapsed = now - start_time
                                        LOGGER.info(f"VLM流式进度: 已接收约{received_chars}字符，耗时{elapsed:.0f}s")
                                        last_report_time = now
                                except Exception as parse_err:
                                    LOGGER.debug(f"SSE行解析失败，忽略: {parse_err}; 行: {line[:200]}")
                        # 若结束时仍在<think>中，补齐关闭标签后换行
                        if reasoning_open:
                            print("</think>", end="", flush=True)
                            reasoning_open = False
                        print()  # 换行
                        final_text = ''.join(final_text_chunks).strip()
                        # 若可见内容极少，尝试使用 reasoning_content 作为回退
                        if len(final_text) < 5:
                            alt = ''.join(final_reasoning_chunks).strip()
                            if alt:
                                final_text = alt
                        if final_text:
                            total_elapsed = time.time() - start_time
                            LOGGER.success(f"VLM流式分析完成（模型: {model_name}），总耗时{total_elapsed:.1f}s，内容长度{len(final_text)}字符。")
                            return final_text
                        else:
                            LOGGER.warning("VLM流式无内容返回，可能被中断。")
                            # 当作中途卡住处理，进入重试
                            raise requests.exceptions.ReadTimeout("VLM流式无内容")
                    except requests.exceptions.ReadTimeout as e_timeout:
                        if not got_any_token:
                            LOGGER.warning(f"[try {attempt_idx}] 首字超时 {self.first_token_timeout_seconds}s，准备重试...")
                        else:
                            LOGGER.warning(f"[try {attempt_idx}] 流式中途卡住达到超时 {self.first_token_timeout_seconds}s，准备重试...")
                        if attempt_idx >= self.stream_max_retries:
                            LOGGER.error("VLM流式重试耗尽（超时）。")
                            return "VLM分析暂时不可用（网络问题），建议稍后重试。"
                        continue
                    except requests.exceptions.RequestException as e_req:
                        LOGGER.warning(f"[try {attempt_idx}] 流式请求异常: {e_req}，准备重试...")
                        if attempt_idx >= self.stream_max_retries:
                            LOGGER.error("VLM流式重试耗尽（请求异常）。")
                            return "VLM分析暂时不可用（网络问题），建议稍后重试。"
                        continue
                    except Exception as e_any:
                        LOGGER.warning(f"[try {attempt_idx}] 流式处理异常: {e_any}，准备重试...")
                        if attempt_idx >= self.stream_max_retries:
                            LOGGER.error("VLM流式重试耗尽（处理异常）。")
                            return "VLM分析暂时不可用（网络问题），建议稍后重试。"
                        continue
        except requests.exceptions.RequestException as e:
            LOGGER.error(f"VLM API请求失败: {e}")
            
            # 打印详细的错误信息
            if hasattr(e, 'response') and e.response is not None:
                LOGGER.error(f"HTTP状态码: {e.response.status_code}")
                LOGGER.error(f"响应头: {dict(e.response.headers)}")
                LOGGER.error(f"响应内容: {e.response.text}")
                
                # 如果是400错误，还要打印我们发送的请求内容
                if e.response.status_code == 400:
                    LOGGER.error("=== 400错误详细调试信息 ===")
                    LOGGER.error(f"请求URL: {self.api_url}")
                    LOGGER.error(f"使用模型: {model_name}")
                    LOGGER.error(f"请求头: {headers}")
                    
                    # 打印payload，但截断base64内容以避免过长
                    debug_payload = payload.copy()
                    if 'messages' in debug_payload and debug_payload['messages']:
                        for msg in debug_payload['messages']:
                            if 'content' in msg:
                                for content_item in msg['content']:
                                    if content_item.get('type') == 'image_url':
                                        original_url = content_item['image_url']['url']
                                        if len(original_url) > 100:
                                            content_item['image_url']['url'] = original_url[:100] + "...[截断]"
                    
                    LOGGER.error(f"请求payload: {json.dumps(debug_payload, indent=2, ensure_ascii=False)}")
                    LOGGER.error("=== 调试信息结束 ===")
            
            # 返回一个默认的分析结果，而不是抛出异常
            return f"VLM分析暂时不可用（网络问题），将基于K线图形态进行基础分析。建议观察价格在关键支撑阻力位的表现。"
        except RetryError as e:
            LOGGER.error(f"VLM API重试耗尽，彻底失败: {e}")
            return f"VLM分析暂时不可用（网络问题），将基于K线图形态进行基础分析。建议观察价格在关键支撑阻力位的表现。"
        except (KeyError, IndexError) as e:
            LOGGER.error(f"解析VLM API响应失败: {e}")
            return "VLM分析响应解析失败，建议使用技术指标辅助判断。"
        finally:
            # --- 恢复代理设置 ---
            if orig_http_proxy is not None:
                os.environ['http_proxy'] = orig_http_proxy
            if orig_https_proxy is not None:
                os.environ['https_proxy'] = orig_https_proxy
            if orig_HTTP_PROXY is not None:
                os.environ['HTTP_PROXY'] = orig_HTTP_PROXY
            if orig_HTTPS_PROXY is not None:
                os.environ['HTTPS_PROXY'] = orig_HTTPS_PROXY
            if orig_session_proxies is not None:
                self.session.proxies = orig_session_proxies
            # 恢复trust_env
            try:
                self.session.trust_env = orig_trust_env
            except Exception:
                pass

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=15))
    def post_with_retry(self, url, headers, json, timeout, stream_response: bool = False):
        # 每次重试前都彻底清理代理设置
        import os
        for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            if key in os.environ:
                del os.environ[key]
        
        # 确保session代理也清空
        self.session.proxies = {}
        # 同时关闭环境继承，避免代理再次被读取
        orig_trust_env = getattr(self.session, 'trust_env', True)
        self.session.trust_env = False
        
        # 增加间隔避免频繁请求
        import time
        time.sleep(2)
        
        # timeout 可以是单值或 (连接超时, 读取超时)，这里统一设置为 (30秒连接, 剩余读取)
        if isinstance(timeout, (int, float)):
            timeout = (30, timeout)
        try:
            return self.session.post(url, headers=headers, json=json, timeout=timeout, stream=stream_response)
        finally:
            # 恢复trust_env
            try:
                self.session.trust_env = orig_trust_env
            except Exception:
                pass

    def analyze_media(self, media_url: str, tweet_text: str, is_video: bool = False) -> Optional[str]:
        """分析来自推文的在线媒体（图片/视频），使用7B模型，支持缓存。"""
        
        # 首先检查缓存
        cached_result = self.cache.get_tweet_analysis(tweet_text, media_url)
        if cached_result:
            return cached_result
        
        # 缓存中没有，进行实际分析
        media_data = self._download_media(media_url)
        if not media_data:
            return "媒体文件下载失败。"

        base64_media = base64.b64encode(media_data["bytes"]).decode('utf-8')
        mime_type = media_data["mime_type"]
        media_type = "视频" if is_video else "图片"
        prompt_text = f"你是一位专注于加密货币的金融情感分析专家。请分析这个{media_type}在比特币（BTC）背景下的情绪。附带的推文是：\"{tweet_text}\"。请仅根据视觉内容判断对比特币价格的情绪是积极、消极还是中性？请用一句话总结视觉内容和你的情绪结论。"
        
        analysis_result = self._analyze_with_vlm(base64_media, mime_type, prompt_text, self.tweet_model)
        
        # 将结果缓存起来（如果分析成功）
        if analysis_result and not analysis_result.startswith("VLM API请求失败"):
            self.cache.set_tweet_analysis(tweet_text, media_url, analysis_result)
        
        return analysis_result

    def analyze_kline_chart(self, image_path: str, data_time_range: Optional[str] = None, symbol: str = 'BTC/USDT', timeframe: Optional[str] = None, price_data=None) -> Optional[str]:
        """分析本地的K线图图片，使用72B模型进行更精准的技术分析，支持基于时间范围的智能缓存。
        
        Args:
            image_path: K线图图片路径
            data_time_range: 数据时间范围标识
            symbol: 交易对，例如 'BTC/USDT'
            timeframe: 时间周期（如'1h', '1d', '1w'等）
            price_data: 价格数据DataFrame，用于提取价格范围信息
        """
        
        # 生成缓存hash标识
        if data_time_range:
            # 使用数据时间范围作为主要缓存标识（确保同一小时内的数据共享缓存）
            data_hash = hashlib.md5(data_time_range.encode()).hexdigest()
            LOGGER.info(f"使用数据时间范围生成缓存hash: {data_time_range}")
        else:
            # 回退到文件属性（兼容旧调用方式）
            try:
                stat_info = os.stat(image_path)
                file_size = stat_info.st_size
                file_mtime = stat_info.st_mtime
                data_hash = hashlib.md5(f"{image_path}_{file_size}_{file_mtime}".encode()).hexdigest()
                LOGGER.info("使用文件属性生成缓存hash（回退模式）")
            except Exception as e:
                LOGGER.warning(f"无法生成K线图文件hash: {e}")
                data_hash = hashlib.md5(image_path.encode()).hexdigest()
        
        # 检查缓存
        cached_result = self.cache.get_kline_analysis(data_hash)
        if cached_result:
            return cached_result
        
        # 缓存中没有，进行实际分析
        try:
            with open(image_path, "rb") as image_file:
                base64_media = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        except Exception as e:
            LOGGER.error(f"读取或编码K线图失败: {e}")
            return "读取K线图文件失败。"
            
        # 根据时间周期生成不同的提示词
        timeframe_info = self._get_timeframe_info(timeframe)
        
        # 提取价格范围信息（如果有价格数据）
        price_range_info = ""
        if price_data is not None and not price_data.empty:
            try:
                current_price = price_data['close'].iloc[-1]
                high_price = price_data['high'].max()
                low_price = price_data['low'].min()
                price_range_info = f"""

**重要价格参考信息:**
*   **当前价格**: 约 {current_price:,.0f} USDT
*   **图表价格范围**: {low_price:,.0f} - {high_price:,.0f} USDT
*   **注意**: 请基于这个价格范围来识别图表中的支撑阻力位，避免价格读取错误。
"""
            except Exception as e:
                LOGGER.warning(f"提取价格范围信息失败: {e}")
        
        prompt_text = f"""
你是一名精通技术分析的资深量化交易员。请仔细分析这张 **{symbol}** 的{timeframe_info['name']}K线图。
你的分析必须只使用简体中文。
{price_range_info}
**图表指标说明:**
*   **K线 (Candlestick)**: 绿色(#26A69A)为阳线, 红色(#EF5350)为阴线。
*   **移动平均线 (MA)**:
    *   MA5 (5周期, 蓝色)
    *   MA10 (10周期, 亮绿色)
    *   MA20 (20周期, 黄色)
    *   MA60 (60周期, 粉红色)
*   **布林带 (Bollinger Bands)**: 由MA20生成的上下两条灰色轨道线。
*   **底部面板1 - 成交量 (Volume)**: 柱状图，颜色与K线对应。
*   **底部面板2 - RSI (相对强弱指数)**:
    *   紫色线 (14周期)

**判断原则（必须遵守）:**
*   **默认中性**: 若证据不足或信号冲突，请结论为中性/震荡，不要过度解读噪声。
*   **做空门槛（更高要求）**: 仅当以下条件至少满足两项，方可给出明确看跌：
    1) 关键支撑位被有效跌破（收盘价确认），且价格在均线系统下方运行；
    2) 动量与量能共振（如RSI跌破关键阈值且放量下跌）；
    3) 更高周期趋势未呈现明确多头，或多头关键位失守；
    4) 形态上有明确的反转/延续空头形态并得到成交量确认。
*   **顺大级别而行**: 如更高周期呈现显著多头结构，避免因短期回撤得出看跌结论，除非满足“做空门槛”。

**弱反弹陷阱识别（必须执行）:**
* 若处于明显下跌后的首次弱反弹阶段，且满足以下至少两项，则将“做多”建议降级为“观望”或“仅小仓位试探”，并在理由中标注“弱反弹陷阱风险”:
  1) 上涨过程中成交量未放大，或量能背离；
  2) 价格仅略高于短期均线（如MA5/MA10），但未站稳MA20/布林中轨；
  3) RSI处于高位但无明确底背离信号；
  4) 上方存在近端密集阻力，且最近高点未有效突破（收盘确认）。

**你的分析任务:**
1.  **当前趋势与动能**:
    *   结合MA5, MA10, MA20, MA60的排列（多头/空头排列）和价格位置，判断当前主要趋势（上升/下降/盘整）。
    *   价格与布林带三轨的关系如何？（例如：在中轨上方运行，触及上轨，跌破下轨等），这揭示了什么趋势强度和波动性？ **特别注意布林带是否收窄，这是市场进入横盘震荡的重要信号。**
2.  **关键形态与价位**:
    *   是否存在头肩、双顶/底、三角形、旗形等经典技术形态？
    *   图中的关键支撑位和阻力位在哪里？（可结合均线、布林带轨道和前期高低点判断）
3.  **成交量与RSI验证**:
    *   成交量在关键价格行为（如突破、反转）时是否配合？（例如：放量突破阻力位，缩量回调）
    *   RSI指标处于什么区域（超买/超卖/中性）？是否与价格走势形成背离（顶背离/底背离）？
4.  **综合结论与策略**:
    *   **核心结论**: 综合以上所有信息，对{timeframe_info['forecast_period']}的价格走势给出一个明确的 **看涨 (Bullish)**、**看跌 (Bearish)** 或 **中性/震荡 (Neutral/Sideways)** 的判断；若为看跌，请在结论中用条目列出满足的“做空门槛”。
    *   **主要理由**: 简明扼要地列出支持你结论的核心技术信号（例如：MA多头排列，RSI底背离，放量突破上轨）。
    *   **操作建议**: 基于你的结论，**必须** 使用以下模板提供明确的操作信号。请务必使用指定的关键词（`信号`、`条件`、`价格`、`理由`），不要有任何偏差。
        *   **信号**: [做多/做空/观望]
        *   **条件**: [立即执行 / 价格高于 / 价格低于]
        *   **价格**: [触发条件的价格，例如 65000 USDT。如果“条件”是“立即执行”，则填“N/A”]
        *   **理由**: [简明扼要地说明给出此建议的原因]

请以结构化、逻辑清晰的方式提供你的专业分析，并在报告标题中明确标注这是{timeframe_info['name']}技术分析报告。
"""
        try:
            analysis_result = self._analyze_with_vlm(base64_media, mime_type, prompt_text, self.kline_model)

            # --- 对VLM输出进行理智检查 ---
            if not _is_sane(analysis_result):
                LOGGER.warning(f"VLM输出未通过理智检查，内容可能为乱码。原始输出: '{analysis_result[:200]}...'")
                analysis_result = "VLM分析结果异常（可能为乱码），本次分析已忽略。"

            # 将结果缓存起来（如果分析成功）
            if analysis_result and "VLM分析" not in analysis_result: # 仅在非错误/警告信息时缓存
                info_text = data_time_range or f"K线图文件: {os.path.basename(image_path)}"
                self.cache.set_kline_analysis(data_hash, info_text, analysis_result)
            
            return analysis_result
            
        except Exception as e:
            # VLM分析失败时的降级处理，不让系统崩溃
            LOGGER.warning(f"VLM分析失败 ({timeframe_info['name']}): {e}")
            fallback_analysis = f"""### BTC/USDT {timeframe_info['name']}技术分析报告（降级模式）

---

VLM分析暂时不可用（网络问题），将基于K线图形态进行基础分析。

#### 技术分析建议
- **当前状态**: 系统无法完成详细的图表分析
- **操作建议**: 建议观察价格在关键支撑阻力位的表现
- **风险提示**: 在网络连接恢复前，建议谨慎操作或暂停自动交易

#### 降级策略
- 密切关注价格突破关键技术位的确认
- 结合成交量变化判断趋势强度
- 重点关注主要移动平均线的支撑阻力作用

---

**注意**: 本报告为降级模式，建议等待网络恢复后重新获取完整分析。
"""
            return fallback_analysis

    def _get_timeframe_info(self, timeframe: Optional[str]) -> Dict[str, str]:
        """根据时间周期获取相关信息"""
        timeframe_mapping = {
            '1h': {'name': '1小时', 'forecast_period': '未来4-12小时'},
            '1d': {'name': '日线', 'forecast_period': '未来1-3天'},
            '1w': {'name': '周线', 'forecast_period': '未来1-2周'},
            '4h': {'name': '4小时', 'forecast_period': '未来1-2天'},
            '15m': {'name': '15分钟', 'forecast_period': '未来2-6小时'},
            '30m': {'name': '30分钟', 'forecast_period': '未来4-8小时'},
            '5m': {'name': '5分钟', 'forecast_period': '未来30-120分钟'},
        }
        
        # 默认使用1小时
        default_info = {'name': '1小时', 'forecast_period': '未来4-12小时'}
        
        if timeframe is None:
            return default_info
        
        return timeframe_mapping.get(timeframe, default_info)

if __name__ == '__main__':
    from btc_predictor.utils import setup_logger
    setup_logger()
    
    LOGGER.info("--- 运行 VLMAnalyzer 独立测试 ---")
    
    # 使用一个真实的、公开的图片URL进行测试
    test_media_url = "https://i.insider.com/606dd4b56183e200195e0493?width=1136&format=jpeg"
    test_tweet_text = "看看我本地咖啡馆里新的比特币ATM！#比特币涨到月球"

    try:
        analyzer = VLMAnalyzer()
        if analyzer.api_key:
            analysis = analyzer.analyze_media(test_media_url, test_tweet_text)
            print("\n" + "="*25 + " 测试结果 " + "="*25)
            print(f"媒体URL: {test_media_url}")
            print(f"分析结果: {analysis}")
            print("="*64 + "\n")
        else:
            print("\n测试跳过，因为未配置VLM (DeepSeek) API key。")
    except Exception as e:
        LOGGER.error(f"独立测试期间发生错误: {e}") 