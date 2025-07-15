import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

from btc_predictor.utils import LOGGER
import config
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def requests_get_with_retry(*args, **kwargs):
    return requests.get(*args, **kwargs)

def fetch_coindesk_news(limit: int = 15) -> List[Dict[str, Any]]:
    """
    从CoinDesk的RSS源获取最新的加密货币新闻。

    Args:
        limit (int): 要返回的最大新闻条目数。

    Returns:
        List[Dict[str, Any]]: 新闻条目列表，每条新闻是一个字典。
    """
    coindesk_rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    LOGGER.info(f"正在从CoinDesk RSS源获取新闻: {coindesk_rss_url}")

    try:
        # 使用requests库，并设置代理
        proxy_url = config.DEFAULTS.get('proxy_url')
        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
        
        response = requests_get_with_retry(coindesk_rss_url, timeout=20, proxies=proxies)
        response.raise_for_status()  # 如果请求失败 (如 404, 500)，则抛出异常

        root = ET.fromstring(response.content)
        news_items = []

        # RSS源中的命名空间
        ns = {'dc': 'http://purl.org/dc/elements/1.1/'}

        for item in root.findall('.//item'):
            creator = item.find('dc:creator', ns)
            title_elem = item.find('title')
            desc_elem = item.find('description')
            pub_elem = item.find('pubDate')
            link_elem = item.find('link')
            news = {
                'text': title_elem.text if title_elem is not None and title_elem.text is not None else '',
                'description': desc_elem.text if desc_elem is not None and desc_elem.text is not None else '',
                'created_at': pub_elem.text if pub_elem is not None and pub_elem.text is not None else '',
                'url': link_elem.text if link_elem is not None and link_elem.text is not None else '',
                'source': f"CoinDesk News by {creator.text if creator is not None and creator.text is not None else 'Unknown'}",
                'media_url': [],
                'video_url': []
            }
            news_items.append(news)
        
        if not news_items:
            LOGGER.warning("成功获取CoinDesk RSS源，但未能解析出任何新闻条目。")
            return []
            
        LOGGER.success(f"成功从CoinDesk获取并解析了 {len(news_items)} 条新闻。")
        
        if limit and len(news_items) > limit:
            return news_items[:limit]
            
        return news_items

    except requests.exceptions.RequestException as e:
        LOGGER.error(f"请求CoinDesk RSS源时发生网络错误: {e}")
        return []
    except ET.ParseError as e:
        LOGGER.error(f"解析CoinDesk RSS XML时出错: {e}")
        return []
    except Exception as e:
        LOGGER.critical(f"获取CoinDesk新闻时发生未知错误: {e}", exc_info=True)
        return [] 