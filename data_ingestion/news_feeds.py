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

async def fetch_coindesk_news_with_playwright(symbol: str, limit: int = 15) -> List[Dict[str, Any]]:
    """
    使用Playwright从CoinDesk的RSS源获取最新的加密货币新闻。
    现在支持根据symbol获取特定币种的新闻。
    """
    if not PLAYWRIGHT_AVAILABLE:
        LOGGER.error("Playwright不可用，无法获取新闻")
        return []

    # 根据symbol构建RSS URL
    asset = symbol.split('-')[0].lower() # e.g., 'btc' from 'BTC-USDT-SWAP'
    
    # 基础URL和特定币种URL的映射
    url_map = {
        'btc': "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
        'eth': "https://www.coindesk.com/arc/outboundfeeds/rss/category/markets/ethereum/?outputType=xml",
        'sol': "https://www.coindesk.com/arc/outboundfeeds/rss/category/web3/solana/?outputType=xml",
    }
    # 默认使用BTC的URL
    coindesk_rss_url = url_map.get(asset, url_map['btc'])
    
    LOGGER.info(f"正在使用Playwright从CoinDesk RSS源 ({symbol}) 获取新闻: {coindesk_rss_url}")

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

def fetch_coindesk_news(symbol: str = 'BTC-USDT-SWAP', limit: int = 15) -> List[Dict[str, Any]]:
    """
    从CoinDesk的RSS源获取最新的加密货币新闻。
    使用Playwright作为主要方法，如果失败则返回空列表。

    Args:
        symbol (str): 交易对，例如 'BTC-USDT-SWAP'
        limit (int): 要返回的最大新闻条目数。

    Returns:
        List[Dict[str, Any]]: 新闻条目列表，每条新闻是一个字典。
    """
    try:
        # 运行异步函数
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(fetch_coindesk_news_with_playwright(symbol, limit))
        finally:
            loop.close()
        return result
    except Exception as e:
        LOGGER.error(f"获取CoinDesk新闻时发生错误: {e}")
        return []

async def fetch_truthsocial_news_with_playwright(accounts: List[str] = None, limit: int = 15) -> List[Dict[str, Any]]:
    """
    使用Playwright从TruthSocial获取指定账号的帖子。
    特别关注特朗普的账号，因为他的政策声明可能影响BTC价格。

    Args:
        accounts (List[str]): 要获取的账号列表，例如 ['realDonaldTrump']。默认为None时只获取特朗普的。
        limit (int): 每个账号要返回的最大帖子数。
    
    Returns:
        List[Dict[str, Any]]: 帖子列表，每个帖子是一个字典。
    """
    if not PLAYWRIGHT_AVAILABLE:
        LOGGER.error("Playwright不可用，无法获取TruthSocial帖子")
        return []

    if accounts is None:
        accounts = ['realDonaldTrump']  # 默认关注特朗普

    all_posts = []
    
    try:
        async with async_playwright() as p:
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
            
            page = await browser.new_page()
            
            # 设置用户代理
            await page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            
            # 设置代理（如果配置了）
            proxy_url = config.DEFAULTS.get('proxy_url')
            if proxy_url:
                LOGGER.info(f"使用代理: {proxy_url}")

            for account in accounts:
                try:
                    LOGGER.info(f"正在获取 @{account} 的TruthSocial帖子...")
                    url = f"https://truthsocial.com/@{account}"
                    
                    # 访问页面
                    response = await page.goto(url, wait_until='networkidle', timeout=30000)
                    
                    if not response or response.status != 200:
                        LOGGER.warning(f"访问 @{account} 的页面失败，状态码: {response.status if response else 'No response'}")
                        continue

                    # 等待页面加载
                    await asyncio.sleep(3)

                    # 查找并点击关闭按钮（如果存在）
                    try:
                        close_button = page.locator('button[data-testid="close-modal"]')
                        if await close_button.count() > 0:
                            LOGGER.info("找到关闭按钮，正在点击...")
                            await close_button.click()
                            await asyncio.sleep(2)
                    except Exception as e:
                        LOGGER.debug(f"点击关闭按钮时出错（可能不存在）: {e}")

                    # 等待内容加载
                    await asyncio.sleep(2)

                    # 滚动页面以加载更多内容
                    for scroll in range(5):  # 增加滚动次数
                        await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                        await asyncio.sleep(3)  # 增加等待时间

                    # 尝试多种可能的选择器来查找帖子
                    posts = []
                    
                    # 方法1: 优先使用 data-testid*="status" 选择器（根据结构探索结果）
                    status_elements = await page.locator('[data-testid*="status"]').all()
                    if len(status_elements) > 0:
                        LOGGER.info(f"找到 {len(status_elements)} 个 [data-testid*='status'] 元素")
                        for idx, status_elem in enumerate(status_elements[:limit * 2]):
                            try:
                                post_data = await _extract_post_data(status_elem, account, page)
                                if post_data and post_data.get('text'):
                                    LOGGER.debug(f"从status元素 {idx+1} 成功提取帖子: {post_data.get('text', '')[:50]}...")
                                    posts.append(post_data)
                                else:
                                    LOGGER.debug(f"从status元素 {idx+1} 提取失败: 无文本内容")
                            except Exception as e:
                                LOGGER.debug(f"提取帖子数据时出错: {e}")
                                continue
                    
                    # 方法2: 使用 class*="status" 选择器（根据结构探索结果）
                    if len(posts) < limit:
                        class_status_elements = await page.locator('[class*="status"]').all()
                        if len(class_status_elements) > 0:
                            LOGGER.info(f"找到 {len(class_status_elements)} 个 [class*='status'] 元素")
                            for idx, status_elem in enumerate(class_status_elements[:limit * 2]):
                                try:
                                    post_data = await _extract_post_data(status_elem, account, page)
                                    if post_data and post_data.get('text'):
                                        # 检查是否已存在（基于URL或文本）
                                        if not any(p.get('url') == post_data.get('url') or 
                                                  (p.get('text', '').strip() == post_data.get('text', '').strip() and len(post_data.get('text', '')) > 20)
                                                  for p in posts):
                                            LOGGER.debug(f"从class status元素 {idx+1} 成功提取帖子: {post_data.get('text', '')[:50]}...")
                                            posts.append(post_data)
                                except Exception as e:
                                    LOGGER.debug(f"提取帖子数据时出错: {e}")
                                    continue
                    
                    # 方法3: 尝试查找article元素
                    if len(posts) < limit:
                        article_elements = await page.locator('article').all()
                        if len(article_elements) > 0:
                            LOGGER.info(f"找到 {len(article_elements)} 个article元素")
                            for idx, article in enumerate(article_elements[:limit * 2]):
                                try:
                                    post_data = await _extract_post_data(article, account, page)
                                    if post_data and post_data.get('text'):
                                        # 检查是否已存在
                                        if not any(p.get('url') == post_data.get('url') or 
                                                  (p.get('text', '').strip() == post_data.get('text', '').strip() and len(post_data.get('text', '')) > 20)
                                                  for p in posts):
                                            LOGGER.debug(f"从article {idx+1} 成功提取帖子: {post_data.get('text', '')[:50]}...")
                                            posts.append(post_data)
                                except Exception as e:
                                    LOGGER.debug(f"提取帖子数据时出错: {e}")
                                    continue
                        else:
                            LOGGER.warning("未找到article元素，尝试其他方法...")
                    
                    # 方法2: 如果article方法没找到足够的内容，尝试查找包含帖子链接的元素
                    if len(posts) < limit:
                        LOGGER.info("尝试通过链接查找更多帖子...")
                        # 尝试查找包含帖子链接的元素
                        post_links = await page.locator(f'a[href*="/@{account}/"]').all()
                        if len(post_links) > 0:
                            LOGGER.info(f"找到 {len(post_links)} 个帖子链接")
                            valid_links_count = 0
                            for link in post_links[:limit * 3]:
                                try:
                                    # 获取链接的href
                                    href = await link.get_attribute('href')
                                    if not href:
                                        continue
                                    
                                    # 放宽条件：只要包含用户名和看起来像帖子链接的就接受
                                    # 不强制要求'/statuses/'，因为TruthSocial的链接格式可能不同
                                    if f'/@{account}/' not in href:
                                        continue
                                    
                                    # 跳过明显不是帖子的链接（如主页、设置等）
                                    if any(skip in href.lower() for skip in ['/settings', '/followers', '/following', '/media', '/likes']):
                                        continue
                                    
                                    valid_links_count += 1
                                    LOGGER.info(f"处理链接 {valid_links_count}: {href}")
                                    
                                    # 获取链接的父元素文本
                                    parent_text = await link.evaluate('''el => {
                                        let parent = el.closest("article") || el.parentElement;
                                        let depth = 0;
                                        while (parent && depth < 5 && (!parent.textContent || parent.textContent.trim().length < 10)) {
                                            parent = parent.parentElement;
                                            depth++;
                                        }
                                        return parent ? parent.textContent : null;
                                    }''')
                                    
                                    if not parent_text:
                                        LOGGER.debug(f"链接 {href} 的父元素文本为空，跳过")
                                        continue
                                    
                                    text_length = len(parent_text.strip())
                                    LOGGER.info(f"链接 {href} 的父元素文本长度: {text_length}")
                                    
                                    # 降低最小长度要求，从20降到10
                                    if text_length < 10:
                                        LOGGER.debug(f"父元素文本太短（{text_length}字符），跳过")
                                        continue
                                    
                                    # 构建完整URL
                                    if href.startswith('/'):
                                        full_url = f"https://truthsocial.com{href}"
                                    else:
                                        full_url = href
                                    
                                    # 尝试获取时间
                                    time_text = None
                                    try:
                                        time_elem = await link.evaluate_handle('''el => {
                                            let parent = el.closest("article") || el.parentElement;
                                            let depth = 0;
                                            while (parent && depth < 5) {
                                                const timeEl = parent.querySelector("time");
                                                if (timeEl) return timeEl;
                                                parent = parent.parentElement;
                                                depth++;
                                            }
                                            return null;
                                        }''')
                                        if time_elem:
                                            time_text = await time_elem.get_attribute('datetime') or await time_elem.text_content()
                                    except Exception as e:
                                        LOGGER.debug(f"获取时间时出错: {e}")
                                        pass
                                    
                                    post_data = {
                                        'text': parent_text.strip(),
                                        'description': '',
                                        'created_at': time_text.strip() if time_text else datetime.utcnow().isoformat(),
                                        'url': full_url,
                                        'source': f"TruthSocial @{account}",
                                        'media_url': [],
                                        'video_url': [],
                                        'username': account
                                    }
                                    
                                    # 检查是否已存在（基于URL）
                                    if not any(p.get('url') == full_url for p in posts):
                                        posts.append(post_data)
                                        LOGGER.info(f"成功添加帖子，当前总数: {len(posts)}")
                                    else:
                                        LOGGER.debug(f"帖子已存在，跳过: {full_url}")
                                    
                                    if len(posts) >= limit * 2:
                                        break
                                except Exception as e:
                                    LOGGER.warning(f"从链接提取帖子数据时出错: {e}")
                                    import traceback
                                    LOGGER.debug(traceback.format_exc())
                                    continue

                    # 去重（基于URL）
                    seen_urls = set()
                    unique_posts = []
                    for post in posts:
                        if post.get('url') and post['url'] not in seen_urls:
                            seen_urls.add(post['url'])
                            unique_posts.append(post)

                    # 限制数量
                    if limit and len(unique_posts) > limit:
                        unique_posts = unique_posts[:limit]

                    LOGGER.success(f"从 @{account} 获取了 {len(unique_posts)} 条帖子")
                    all_posts.extend(unique_posts)
                
                except Exception as e:
                    LOGGER.error(f"获取 TruthSocial 帖子时发生错误: {e}")
                    return []

            # 按时间排序（最新的在前）
            all_posts.sort(key=lambda x: x.get('created_at', ''), reverse=True)

            return all_posts
                
    except Exception as e:
        LOGGER.error(f"使用Playwright获取TruthSocial帖子时发生错误: {e}")
        return []
    
async def _extract_post_data(element, account: str, page) -> Dict[str, Any]:
    """从元素中提取帖子数据"""
    try:
        # 尝试提取更准确的帖子文本内容
        # 优先查找帖子正文区域，而不是整个元素的文本
        text = None
        try:
            # 方法1: 尝试查找常见的帖子文本容器
            text_elem = await element.evaluate_handle('''el => {
                // 查找包含帖子正文的元素
                // 常见的帖子文本容器选择器
                const selectors = [
                    '[data-testid*="text"]',
                    '[class*="text"]',
                    '[class*="content"]',
                    '[class*="post-text"]',
                    'p',
                    'div[dir="auto"]',
                ];
                
                for (const selector of selectors) {
                    const elem = el.querySelector(selector);
                    if (elem && elem.textContent && elem.textContent.trim().length > 20) {
                        return elem;
                    }
                }
                
                // 如果没找到，返回整个元素
                return el;
            }''')
            
            if text_elem:
                text_content = await text_elem.text_content()
                if text_content:
                    text = text_content.strip()
        except Exception as e:
            LOGGER.debug(f"提取帖子文本时出错: {e}")
            pass
        
        # 如果上面的方法失败，使用整个元素的文本
        if not text:
            text_content = await element.text_content()
            if text_content:
                text = text_content.strip()
        
        if not text:
            return None

        # 清理文本：移除多余的空格和换行
        text = ' '.join(text.split())
        
        # 过滤掉太短的文本（但如果有有效的帖子链接，仍然保留）
        if len(text) < 10:
            # 检查是否有有效的帖子链接，如果有则保留
            # 这个检查会在后面进行
            pass

        # 查找链接 - 使用更灵活的方法
        url = None
        try:
            # 方法1: 查找包含用户名的链接
            link_elem = element.locator(f'a[href*="/@{account}/"]').first
            if await link_elem.count() > 0:
                href = await link_elem.get_attribute('href')
                if href:
                    # 跳过明显不是帖子的链接
                    if not any(skip in href.lower() for skip in ['/settings', '/followers', '/following', '/media', '/likes']):
                        if href.startswith('/'):
                            url = f"https://truthsocial.com{href}"
                        else:
                            url = href
        except:
            pass

        # 方法2: 如果没有找到链接，尝试从元素本身或父元素获取
        if not url:
            try:
                href = await element.evaluate('''el => {
                    // 先查找包含用户名的链接
                    let link = el.querySelector('a[href*="/@"]');
                    if (link) {
                        let href = link.getAttribute('href');
                        // 跳过明显不是帖子的链接
                        if (href && !['/settings', '/followers', '/following', '/media', '/likes'].some(skip => href.toLowerCase().includes(skip))) {
                            return href;
                        }
                    }
                    // 如果没找到，尝试查找任何包含/的链接
                    link = el.querySelector('a[href*="/"]');
                    if (link) {
                        let href = link.getAttribute('href');
                        if (href && href.includes('/') && !href.startsWith('#')) {
                            return href;
                        }
                    }
                    return null;
                }''')
                if href:
                    if f'/@{account}/' in href or '/statuses/' in href or (href.count('/') >= 3 and not any(skip in href.lower() for skip in ['/settings', '/followers', '/following', '/media', '/likes'])):
                        if href.startswith('/'):
                            url = f"https://truthsocial.com{href}"
                        elif href.startswith('http'):
                            url = href
            except Exception as e:
                LOGGER.debug(f"通过evaluate查找链接时出错: {e}")
                pass

        # 如果仍然没有找到链接，使用默认URL
        if not url:
            url = f"https://truthsocial.com/@{account}"

        # 查找时间 - 修复逻辑错误
        created_at = None
        try:
            time_elem = element.locator('time').first
            count = await time_elem.count()
            if count > 0:
                datetime_attr = await time_elem.get_attribute('datetime')
                if datetime_attr:
                    created_at = datetime_attr
                else:
                    time_text = await time_elem.text_content()
                    if time_text:
                        created_at = time_text.strip()
        except Exception as e:
            LOGGER.debug(f"查找时间时出错: {e}")
            pass
        
        # 如果没找到time元素，尝试查找其他时间相关的元素
        if not created_at:
            try:
                time_text = await element.evaluate('''el => {
                    // 查找time元素
                    let timeEl = el.querySelector('time');
                    if (timeEl) {
                        return timeEl.getAttribute('datetime') || timeEl.textContent;
                    }
                    // 查找包含时间的元素
                    let allElements = el.querySelectorAll('[datetime], [class*="time"], [class*="date"]');
                    for (let elem of allElements) {
                        let dt = elem.getAttribute('datetime');
                        if (dt) return dt;
                        let text = elem.textContent;
                        if (text && /\\d{1,2}[\\/\\-]\\d{1,2}/.test(text)) {
                            return text;
                        }
                    }
                    return null;
                }''')
                if time_text:
                    created_at = time_text.strip()
            except Exception as e:
                LOGGER.debug(f"通过evaluate查找时间时出错: {e}")
                pass

        # 查找图片
        media_urls = []
        try:
            img_elements = await element.locator('img').all()
            for img in img_elements:
                src = await img.get_attribute('src')
                if src and 'avatar' not in src.lower() and 'icon' not in src.lower():
                    if src.startswith('http'):
                        media_urls.append(src)
                    elif src.startswith('/'):
                        media_urls.append(f"https://truthsocial.com{src}")
        except:
            pass

        # 查找视频
        video_urls = []
        try:
            video_elements = await element.locator('video').all()
            for video in video_elements:
                src = await video.get_attribute('src')
                if src:
                    if src.startswith('http'):
                        video_urls.append(src)
                    elif src.startswith('/'):
                        video_urls.append(f"https://truthsocial.com{src}")
        except:
            pass

        # 如果没有找到URL，使用默认URL
        if not url:
            url = f"https://truthsocial.com/@{account}"

        # 验证：如果文本太短，检查是否有有效的帖子链接或媒体内容
        if len(text) < 10:
            # 检查是否有有效的帖子链接（包含 /posts/ 或 /statuses/）
            has_valid_post_link = '/posts/' in url or '/statuses/' in url or url != f"https://truthsocial.com/@{account}"
            # 检查是否有媒体内容
            has_media = len(media_urls) > 0 or len(video_urls) > 0
            
            # 如果既没有有效链接也没有媒体内容，则跳过这个帖子
            if not has_valid_post_link and not has_media:
                LOGGER.debug(f"跳过无效帖子：文本太短且无有效链接或媒体内容")
                return None

        # 如果没有找到时间，使用当前时间
        if not created_at:
            created_at = datetime.utcnow().isoformat()

        return {
            'text': text,
            'description': '',
            'created_at': created_at,
            'url': url,
            'source': f"TruthSocial @{account}",
            'media_url': media_urls,
            'video_url': video_urls,
            'username': account
        }
            
    except Exception as e:
        LOGGER.debug(f"提取帖子数据时出错: {e}")
        return None

async def _extract_post_data_from_element(element_handle, account: str, page) -> Dict[str, Any]:
    """从元素句柄中提取帖子数据（备用方法）"""
    try:
        # 如果element_handle是Playwright的ElementHandle，尝试获取其文本和属性
        if hasattr(element_handle, 'text_content'):
            text_content = await element_handle.text_content()
            if not text_content or len(text_content.strip()) < 10:
                return None
            
            # 尝试获取链接
            url = None
            try:
                href = await element_handle.evaluate('''el => {
                    const link = el.closest("a") || el.querySelector("a[href*='/']");
                    return link ? link.href : null;
                }''')
                if href and f'/@{account}/' in href:
                    url = href
            except:
                pass
            
            if not url:
                url = f"https://truthsocial.com/@{account}"
            
            return {
                'text': text_content.strip(),
                'description': '',
                'created_at': datetime.utcnow().isoformat(),
                'url': url,
                'source': f"TruthSocial @{account}",
                'media_url': [],
                'video_url': [],
                'username': account
            }
        return None
    except Exception as e:
        LOGGER.debug(f"从元素句柄提取帖子数据时出错: {e}")
        return None

def fetch_truthsocial_news(accounts: List[str] = None, limit: int = 15) -> List[Dict[str, Any]]:
    """
    从TruthSocial获取指定账号的帖子。
    特别关注特朗普的账号，因为他的政策声明可能影响BTC价格。
    
    Args:
        accounts (List[str]): 要获取的账号列表，例如 ['realDonaldTrump']。默认为None时只获取特朗普的。
        limit (int): 每个账号要返回的最大帖子数。
    
    Returns:
        List[Dict[str, Any]]: 帖子列表，每个帖子是一个字典。
    """
    try:
        # 运行异步函数
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(fetch_truthsocial_news_with_playwright(accounts, limit))
        finally:
            loop.close()
        return result
    except Exception as e:
        LOGGER.error(f"获取TruthSocial帖子时发生错误: {e}")
        return [] 