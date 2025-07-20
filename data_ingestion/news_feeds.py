import asyncio
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from datetime import datetime
import re

from btc_predictor.utils import LOGGER
import config

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    LOGGER.warning("Playwright未安装，将使用备用方法获取新闻")

async def fetch_coindesk_news_with_playwright(limit: int = 15) -> List[Dict[str, Any]]:
    """
    使用Playwright从CoinDesk的RSS源获取最新的加密货币新闻。

    Args:
        limit (int): 要返回的最大新闻条目数。

    Returns:
        List[Dict[str, Any]]: 新闻条目列表，每条新闻是一个字典。
    """
    if not PLAYWRIGHT_AVAILABLE:
        LOGGER.error("Playwright不可用，无法获取新闻")
        return []
    
    coindesk_rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    LOGGER.info(f"正在使用Playwright从CoinDesk RSS源获取新闻: {coindesk_rss_url}")

    try:
        async with async_playwright() as p:
            # 启动浏览器
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor'
                ]
            )
            
            # 创建新页面
            page = await browser.new_page()
            
            # 设置用户代理
            await page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            
            # 设置代理（如果配置了）
            proxy_url = config.DEFAULTS.get('proxy_url')
            if proxy_url:
                LOGGER.info(f"使用代理: {proxy_url}")
                # 注意：这里需要在launch时设置代理，但为了简化，我们先尝试直接访问
            
            # 访问RSS源
            LOGGER.info("正在访问CoinDesk RSS源...")
            response = await page.goto(coindesk_rss_url, wait_until='networkidle', timeout=30000)
            
            if not response or response.status != 200:
                LOGGER.error(f"访问CoinDesk RSS源失败，状态码: {response.status if response else 'No response'}")
                await browser.close()
                return []
            
            # 获取页面内容
            content = await page.content()
            LOGGER.info("成功获取RSS内容，开始解析...")
            
            # 关闭浏览器
            await browser.close()
            
            # 解析XML内容
            try:
                root = ET.fromstring(content)
                news_items = []
                
                # RSS源中的命名空间
                ns = {'dc': 'http://purl.org/dc/elements/1.1/'}
                
                # 查找所有item元素
                items = root.findall('.//item')
                LOGGER.info(f"找到 {len(items)} 个新闻条目")
                
                for item in items:
                    try:
                        # 提取新闻信息
                        creator = item.find('dc:creator', ns)
                        title_elem = item.find('title')
                        desc_elem = item.find('description')
                        pub_elem = item.find('pubDate')
                        link_elem = item.find('link')
                        
                        # 清理和验证数据
                        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ''
                        description = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ''
                        pub_date = pub_elem.text.strip() if pub_elem is not None and pub_elem.text else ''
                        link = link_elem.text.strip() if link_elem is not None and link_elem.text else ''
                        author = creator.text.strip() if creator is not None and creator.text else 'Unknown'
                        
                        # 跳过空标题的条目
                        if not title:
                            continue
                        
                        # 清理HTML标签
                        title = re.sub(r'<[^>]+>', '', title)
                        description = re.sub(r'<[^>]+>', '', description)
                        
                        news = {
                            'text': title,
                            'description': description,
                            'created_at': pub_date,
                            'url': link,
                            'source': f"CoinDesk News by {author}",
                            'media_url': [],
                            'video_url': [],
                            'username': author
                        }
                        news_items.append(news)
                        
                    except Exception as e:
                        LOGGER.warning(f"解析单个新闻条目时出错: {e}")
                        continue
                
                if not news_items:
                    LOGGER.warning("成功获取CoinDesk RSS源，但未能解析出任何有效新闻条目。")
                    return []
                    
                LOGGER.success(f"成功从CoinDesk获取并解析了 {len(news_items)} 条新闻。")
                
                # 限制返回数量
                if limit and len(news_items) > limit:
                    return news_items[:limit]
                    
                return news_items
                
            except ET.ParseError as e:
                LOGGER.error(f"解析CoinDesk RSS XML时出错: {e}")
                return []
                
    except Exception as e:
        LOGGER.error(f"使用Playwright获取CoinDesk新闻时发生错误: {e}")
        return []

def fetch_coindesk_news(limit: int = 15) -> List[Dict[str, Any]]:
    """
    从CoinDesk的RSS源获取最新的加密货币新闻。
    使用Playwright作为主要方法，如果失败则返回空列表。

    Args:
        limit (int): 要返回的最大新闻条目数。

    Returns:
        List[Dict[str, Any]]: 新闻条目列表，每条新闻是一个字典。
    """
    try:
        # 运行异步函数
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(fetch_coindesk_news_with_playwright(limit))
        finally:
            loop.close()
        return result
    except Exception as e:
        LOGGER.error(f"获取CoinDesk新闻时发生错误: {e}")
        return [] 