import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

import config
from btc_predictor.utils import LOGGER

class VLMCache:
    """
    VLM分析结果的缓存管理器，避免重复分析相同内容。
    """
    
    def __init__(self, cache_dir: Optional[str] = None, cache_hours: Optional[int] = None):
        """
        初始化缓存管理器。
        
        Args:
            cache_dir: 缓存目录（如果为None则使用配置文件中的值）
            cache_hours: 缓存有效期（小时，如果为None则使用配置文件中的值）
        """
        # 从配置文件获取默认值
        cache_config = getattr(config, 'VLM_CACHE', {})
        self.enabled = cache_config.get('enabled', True)
        
        if not self.enabled:
            LOGGER.info("VLM缓存已禁用，自动清理所有缓存文件。")
            for f in [self.tweet_cache_file, self.kline_cache_file]:
                if f.exists():
                    f.unlink()
            self.tweet_cache = {}
            self.kline_cache = {}
            return
        
        self.cache_dir = Path(cache_dir or cache_config.get('cache_dir', 'cache'))
        self.cache_dir.mkdir(exist_ok=True)
        
        self.tweet_cache_file = self.cache_dir / "vlm_tweet_cache.json"
        self.kline_cache_file = self.cache_dir / "vlm_kline_cache.json"
        self.cache_hours = cache_hours or cache_config.get('cache_hours', 4)
        
        # 加载现有缓存
        self.tweet_cache = self._load_cache(self.tweet_cache_file)
        self.kline_cache = self._load_cache(self.kline_cache_file)
        # 初始化时自动清理过期缓存
        self.cleanup_expired_cache()
        
        LOGGER.info(f"VLM缓存管理器已初始化，缓存有效期: {self.cache_hours}小时")

    def _load_cache(self, cache_file: Path) -> Dict[str, Any]:
        """从文件加载缓存数据。"""
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                LOGGER.info(f"已加载缓存文件: {cache_file} ({len(cache_data)} 条记录)")
                return cache_data
            except Exception as e:
                LOGGER.warning(f"加载缓存文件失败 {cache_file}: {e}")
        return {}

    def _save_cache(self, cache_data: Dict[str, Any], cache_file: Path):
        """将缓存数据保存到文件。"""
        if not self.enabled:
            LOGGER.info(f"缓存禁用，未写入缓存文件 {cache_file}")
            return
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            LOGGER.error(f"保存缓存文件失败 {cache_file}: {e}")

    def _generate_content_hash(self, content: str) -> str:
        """为内容生成唯一的hash标识。"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _is_cache_valid(self, timestamp_str: str) -> bool:
        """检查缓存是否在有效期内。"""
        try:
            cache_time = datetime.fromisoformat(timestamp_str)
            expiry_time = cache_time + timedelta(hours=self.cache_hours)
            return datetime.now() < expiry_time
        except Exception:
            return False

    def get_tweet_analysis(self, tweet_text: str, media_url: str) -> Optional[str]:
        """
        获取推文的VLM分析结果（如果缓存中有的话）。
        
        Args:
            tweet_text: 推文文本
            media_url: 媒体URL
            
        Returns:
            缓存的分析结果，如果没有有效缓存则返回None
        """
        if not self.enabled:
            return None
        self.cleanup_expired_cache()  # 每次访问前自动清理过期缓存
            
        # 生成内容标识：推文文本 + 媒体URL
        content_key = f"{tweet_text}|{media_url}"
        content_hash = self._generate_content_hash(content_key)
        
        if content_hash in self.tweet_cache:
            cache_entry = self.tweet_cache[content_hash]
            if self._is_cache_valid(cache_entry['timestamp']):
                LOGGER.info(f"使用缓存的推文VLM分析结果 (hash: {content_hash[:8]}...)")
                return cache_entry['analysis']
            else:
                # 缓存过期，删除旧记录
                del self.tweet_cache[content_hash]
                LOGGER.info(f"推文缓存已过期，删除旧记录 (hash: {content_hash[:8]}...)")
        
        return None

    def set_tweet_analysis(self, tweet_text: str, media_url: str, analysis: str):
        """
        缓存推文的VLM分析结果。
        
        Args:
            tweet_text: 推文文本
            media_url: 媒体URL
            analysis: VLM分析结果
        """
        if not self.enabled:
            LOGGER.info("缓存禁用，未写入推文分析缓存。"); return
        self.cleanup_expired_cache()  # 每次写入前自动清理过期缓存
            
        content_key = f"{tweet_text}|{media_url}"
        content_hash = self._generate_content_hash(content_key)
        
        self.tweet_cache[content_hash] = {
            'content_preview': tweet_text[:50] + "..." if len(tweet_text) > 50 else tweet_text,
            'media_url': media_url,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        self._save_cache(self.tweet_cache, self.tweet_cache_file)
        LOGGER.info(f"已缓存推文VLM分析结果 (hash: {content_hash[:8]}...)")

    def get_kline_analysis(self, data_hash: str) -> Optional[str]:
        """
        获取K线图的VLM分析结果（如果缓存中有的话）。
        
        Args:
            data_hash: K线数据的hash标识
            
        Returns:
            缓存的分析结果，如果没有有效缓存则返回None
        """
        if not self.enabled:
            return None
        self.cleanup_expired_cache()  # 每次访问前自动清理过期缓存
            
        if data_hash in self.kline_cache:
            cache_entry = self.kline_cache[data_hash]
            if self._is_cache_valid(cache_entry['timestamp']):
                LOGGER.info(f"使用缓存的K线图VLM分析结果 (hash: {data_hash[:8]}...)")
                return cache_entry['analysis']
            else:
                # 缓存过期，删除旧记录
                del self.kline_cache[data_hash]
                LOGGER.info(f"K线图缓存已过期，删除旧记录 (hash: {data_hash[:8]}...)")
        
        return None

    def set_kline_analysis(self, data_hash: str, data_info: str, analysis: str):
        """
        缓存K线图的VLM分析结果。
        
        Args:
            data_hash: K线数据的hash标识
            data_info: 数据描述信息
            analysis: VLM分析结果
        """
        if not self.enabled:
            LOGGER.info("缓存禁用，未写入K线分析缓存。"); return
        self.cleanup_expired_cache()  # 每次写入前自动清理过期缓存
            
        self.kline_cache[data_hash] = {
            'data_info': data_info,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        self._save_cache(self.kline_cache, self.kline_cache_file)
        LOGGER.info(f"已缓存K线图VLM分析结果 (hash: {data_hash[:8]}...)")

    def cleanup_expired_cache(self):
        """清理所有过期的缓存记录。"""
        if not self.enabled:
            return
            
        # 清理推文缓存
        expired_tweet_keys = []
        for key, entry in self.tweet_cache.items():
            if not self._is_cache_valid(entry['timestamp']):
                expired_tweet_keys.append(key)
        
        for key in expired_tweet_keys:
            del self.tweet_cache[key]
        
        # 清理K线图缓存
        expired_kline_keys = []
        for key, entry in self.kline_cache.items():
            if not self._is_cache_valid(entry['timestamp']):
                expired_kline_keys.append(key)
        
        for key in expired_kline_keys:
            del self.kline_cache[key]
        
        if expired_tweet_keys or expired_kline_keys:
            LOGGER.info(f"清理过期缓存: {len(expired_tweet_keys)} 个推文缓存, {len(expired_kline_keys)} 个K线图缓存")
            self._save_cache(self.tweet_cache, self.tweet_cache_file)
            self._save_cache(self.kline_cache, self.kline_cache_file)

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息。"""
        return {
            'tweet_cache_count': len(self.tweet_cache),
            'kline_cache_count': len(self.kline_cache)
        } 