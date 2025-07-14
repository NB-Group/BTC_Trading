# decision_engine/vlm_analyzer.py
import os
import requests
import base64
import mimetypes
from typing import List, Dict, Any, Optional
import json # 添加用于详细错误日志记录
import hashlib # 添加用于K线图缓存哈希

import config
from btc_predictor.utils import LOGGER
from .vlm_cache import VLMCache # 导入缓存管理器

class VLMAnalyzer:
    """
    一个封装了视觉语言模型（VLM）分析功能的类。
    支持为不同任务使用不同的模型，并具有智能缓存机制。
    """
    def __init__(self):
        deepseek_config = config.API_KEYS.get('deepseek', {})
        self.api_url = (deepseek_config.get('base_url') or "https://api.deepseek.com/v1").rstrip('/') + "/chat/completions"
        self.api_key = deepseek_config.get('api_key')
        
        # 为不同任务定义不同的模型
        self.kline_model = "Qwen/Qwen2.5-VL-72B-Instruct"  # K线图分析使用72B模型
        self.tweet_model = "Pro/Qwen/Qwen2.5-VL-7B-Instruct"  # 推文图片分析使用7B模型
        
        if not self.api_key or 'YOUR' in self.api_key:
            LOGGER.warning("VLM (DeepSeek) API key 未配置，VLM分析功能将被跳过。")
            self.api_key = None

        self.session = requests.Session()
        proxy_url = config.DEFAULTS.get('proxy_url')
        if proxy_url:
            self.session.proxies = {'http': proxy_url, 'https': proxy_url}
            LOGGER.info(f"VLMAnalyzer 已配置代理: {proxy_url}")

        # 初始化缓存管理器（使用配置文件设置）
        self.cache = VLMCache()
        
        # 启动时清理过期缓存
        self.cache.cleanup_expired_cache()
        
        # 显示缓存统计
        stats = self.cache.get_cache_stats()
        LOGGER.info(f"VLM缓存统计: 推文缓存 {stats['tweet_cache_count']} 条, K线图缓存 {stats['kline_cache_count']} 条")

    def _download_media(self, url: str) -> Optional[Dict[str, Any]]:
        """下载媒体文件并返回字节和MIME类型。"""
        if not url:
            return None
        try:
            LOGGER.info(f"正在下载媒体文件: {url}")
            response = self.session.get(url, timeout=20, stream=True)
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
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_media}"}}
            ]}],
            "max_tokens": 500 # 为K线图分析增加token上限
        }

        try:
            LOGGER.info(f"正在使用模型 {model_name} 进行VLM分析...")
            LOGGER.debug(f"VLM API URL: {self.api_url}")
            LOGGER.debug(f"Payload size: {len(json.dumps(payload))} chars")
            
            response = self.session.post(self.api_url, headers=headers, json=payload, timeout=120)
            
            LOGGER.info(f"VLM API响应状态码: {response.status_code}")
            
            response.raise_for_status()
            
            result = response.json()
            analysis = result['choices'][0]['message']['content'].strip()
            LOGGER.success(f"VLM分析成功接收（模型: {model_name}）。")
            LOGGER.info(f"VLM模型输出: {analysis}")
            return analysis
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
            
            return f"VLM API请求失败: {e}"
        except (KeyError, IndexError) as e:
            LOGGER.error(f"解析VLM API响应失败: {e}")
            return "解析VLM API响应失败。"

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

    def analyze_kline_chart(self, image_path: str, data_time_range: Optional[str] = None) -> Optional[str]:
        """分析本地的K线图图片，使用72B模型进行更精准的技术分析，支持基于时间范围的智能缓存。"""
        
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
            
        prompt_text = """
你是一名精通技术分析的资深量化交易员。请仔细分析这张BTC/USDT的K线图。

**图表指标说明:**
*   **移动平均线 (MA)**:
    *   白色线: MA7 (7周期)
    *   黄色线: MA25 (25周期)
    *   紫色线: MA99 (99周期)
*   **布林带 (Bollinger Bands)**:
    *   青色线 (上下两条): 布林带上下轨 (20周期, 2倍标准差)
    *   橙色虚线: 布林带中轨 (MA20)
*   **底部面板 - RSI (相对强弱指数)**:
    *   洋红色线: RSI (14周期)
    *   红色虚线: 超买线 (RSI=70)
    *   绿色虚线: 超卖线 (RSI=30)
*   **成交量**: 柱状图颜色与对应K线颜色一致 (绿涨红跌)。

**你的分析任务:**
1.  **当前趋势与动能**:
    *   结合MA7, MA25, MA99的排列（多头/空头排列）和价格位置，判断当前主要趋势（上升/下降/盘整）。
    *   价格与布林带三轨的关系如何？（例如：在中轨上方运行，触及上轨，跌破下轨等），这揭示了什么趋势强度和波动性？
2.  **关键形态与价位**:
    *   是否存在头肩、双顶/底、三角形、旗形等经典技术形态？
    *   图中的关键支撑位和阻力位在哪里？（可结合均线、布林带轨道和前期高低点判断）
3.  **成交量与RSI验证**:
    *   成交量在关键价格行为（如突破、反转）时是否配合？（例如：放量突破阻力位，缩量回调）
    *   RSI指标处于什么区域（超买/超卖/中性）？是否与价格走势形成背离（顶背离/底背离）？
4.  **综合结论与策略**:
    *   **核心结论**: 综合以上所有信息，对未来4-12小时的价格走势给出一个明确的 **看涨 (Bullish)**、**看跌 (Bearish)** 或 **中性/震荡 (Neutral/Sideways)** 的判断。
    *   **主要理由**: 简明扼要地列出支持你结论的核心技术信号（例如：MA多头排列，RSI底背离，放量突破上轨）。
    *   **操作建议**: 基于结论，提出具体的交易策略（例如：若看涨，可在XX价位附近入场，止损设于YY，目标看至ZZ）。

请以结构化、逻辑清晰的方式提供你的专业分析。
"""
        analysis_result = self._analyze_with_vlm(base64_media, mime_type, prompt_text, self.kline_model)
        
        # 将结果缓存起来（如果分析成功）
        if analysis_result and not analysis_result.startswith("VLM API请求失败"):
            info_text = data_time_range or f"K线图文件: {os.path.basename(image_path)}"
            self.cache.set_kline_analysis(data_hash, info_text, analysis_result)
        
        return analysis_result

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