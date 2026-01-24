"""
Clash API客户端，用于在代理节点失效时切换节点。
"""
import requests
import random
from typing import List, Dict, Optional, Any
from btc_predictor.utils import LOGGER


class ClashClient:
    """
    Clash代理客户端，用于管理代理节点切换。
    """
    
    def __init__(self, api_url: str = "http://127.0.0.1:9090", secret: Optional[str] = None):
        """
        初始化Clash客户端。
        
        Args:
            api_url: Clash API地址，默认 http://127.0.0.1:9090
            secret: Clash API密钥（如果设置了认证）
        """
        self.api_url = api_url.rstrip('/')
        self.secret = secret
        self.headers = {}
        if secret:
            self.headers['Authorization'] = f'Bearer {secret}'
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        发送HTTP请求到Clash API。
        
        Args:
            method: HTTP方法
            endpoint: API端点
            **kwargs: 其他请求参数
            
        Returns:
            响应JSON数据，失败返回None
        """
        url = f"{self.api_url}{endpoint}"
        try:
            response = requests.request(method, url, headers=self.headers, timeout=5, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}
        except Exception as e:
            LOGGER.debug(f"Clash API请求失败 ({method} {endpoint}): {e}")
            return None
    
    def get_proxies(self) -> Optional[Dict[str, Any]]:
        """
        获取所有代理组和节点信息。
        
        Returns:
            代理信息字典，失败返回None
        """
        return self._request('GET', '/proxies')
    
    def get_proxy_groups(self) -> List[str]:
        """
        获取所有代理组名称。
        
        Returns:
            代理组名称列表
        """
        proxies = self.get_proxies()
        if not proxies:
            return []
        
        # Clash API返回格式: {"proxies": {"GLOBAL": {...}, "Proxy": {...}, ...}}
        proxy_groups = []
        if 'proxies' in proxies:
            for name, info in proxies['proxies'].items():
                # 只返回代理组（包含all字段的）
                if isinstance(info, dict) and 'all' in info:
                    proxy_groups.append(name)
        return proxy_groups
    
    def get_proxy_nodes(self, group_name: str) -> List[str]:
        """
        获取指定代理组的所有节点名称。
        
        Args:
            group_name: 代理组名称
            
        Returns:
            节点名称列表
        """
        proxies = self.get_proxies()
        if not proxies or 'proxies' not in proxies:
            return []
        
        group_info = proxies['proxies'].get(group_name, {})
        if isinstance(group_info, dict) and 'all' in group_info:
            return group_info.get('all', [])
        return []
    
    def get_current_proxy(self, group_name: str = "GLOBAL") -> Optional[str]:
        """
        获取当前使用的代理节点。
        
        Args:
            group_name: 代理组名称，默认GLOBAL
            
        Returns:
            当前节点名称，失败返回None
        """
        proxies = self.get_proxies()
        if not proxies or 'proxies' not in proxies:
            return None
        
        group_info = proxies['proxies'].get(group_name, {})
        if isinstance(group_info, dict):
            return group_info.get('now')
        return None
    
    def switch_proxy(self, group_name: str, proxy_name: str) -> bool:
        """
        切换指定代理组的节点。
        
        Args:
            group_name: 代理组名称
            proxy_name: 要切换到的节点名称
            
        Returns:
            是否切换成功
        """
        result = self._request('PUT', f'/proxies/{group_name}', json={'name': proxy_name})
        if result is not None:
            LOGGER.info(f"Clash节点切换成功: {group_name} -> {proxy_name}")
            return True
        else:
            LOGGER.warning(f"Clash节点切换失败: {group_name} -> {proxy_name}")
            return False
    
    def switch_to_random_proxy(self, group_name: str = "GLOBAL") -> bool:
        """
        切换到随机节点。
        
        Args:
            group_name: 代理组名称，默认GLOBAL
            
        Returns:
            是否切换成功
        """
        nodes = self.get_proxy_nodes(group_name)
        if not nodes:
            LOGGER.warning(f"Clash代理组 {group_name} 没有可用节点")
            return False
        
        # 排除当前节点，避免切换到相同节点
        current = self.get_current_proxy(group_name)
        available_nodes = [n for n in nodes if n != current]
        
        if not available_nodes:
            # 如果没有其他节点，就使用所有节点（包括当前节点）
            available_nodes = nodes
        
        # 随机选择一个节点
        new_proxy = random.choice(available_nodes)
        return self.switch_proxy(group_name, new_proxy)
    
    def switch_to_next_proxy(self, group_name: str = "GLOBAL") -> bool:
        """
        切换到下一个节点（按顺序）。
        
        Args:
            group_name: 代理组名称，默认GLOBAL
            
        Returns:
            是否切换成功
        """
        nodes = self.get_proxy_nodes(group_name)
        if not nodes:
            LOGGER.warning(f"Clash代理组 {group_name} 没有可用节点")
            return False
        
        current = self.get_current_proxy(group_name)
        if not current or current not in nodes:
            # 如果当前节点不存在，切换到第一个节点
            return self.switch_proxy(group_name, nodes[0])
        
        # 找到当前节点的索引，切换到下一个
        current_index = nodes.index(current)
        next_index = (current_index + 1) % len(nodes)
        next_proxy = nodes[next_index]
        
        return self.switch_proxy(group_name, next_proxy)


# 全局Clash客户端实例（延迟初始化）
_clash_client: Optional[ClashClient] = None


def get_clash_client() -> Optional[ClashClient]:
    """
    获取全局Clash客户端实例。
    如果Clash API不可用或未启用，返回None。
    
    Returns:
        ClashClient实例或None
    """
    global _clash_client
    
    if _clash_client is None:
        try:
            import config
            clash_config = getattr(config, 'CLASH_CONFIG', {})
            
            # 检查是否启用
            if not clash_config.get('enabled', True):
                LOGGER.debug("Clash节点切换功能已禁用")
                return None
            
            api_url = clash_config.get('api_url', 'http://127.0.0.1:9090')
            secret = clash_config.get('secret')
            
            _clash_client = ClashClient(api_url=api_url, secret=secret)
            
            # 测试连接
            proxies = _clash_client.get_proxies()
            if proxies is None:
                LOGGER.warning("Clash API不可用，将不会自动切换节点")
                _clash_client = None
            else:
                LOGGER.info("Clash API连接成功，已启用自动节点切换功能")
        except Exception as e:
            LOGGER.debug(f"初始化Clash客户端失败: {e}")
            _clash_client = None
    
    return _clash_client


def switch_clash_proxy_on_retry(retry_state):
    """
    在retry失败时切换Clash节点的回调函数。
    用于tenacity的before_sleep参数。
    
    Args:
        retry_state: tenacity的retry状态对象
    """
    clash_client = get_clash_client()
    if clash_client is None:
        return
    
    # 只在重试时切换（不是第一次尝试）
    if retry_state.attempt_number > 1:
        LOGGER.info(f"检测到API请求失败（第{retry_state.attempt_number}次重试），尝试切换Clash节点...")
        
        # 尝试切换到下一个节点
        success = clash_client.switch_to_next_proxy("GLOBAL")
        if not success:
            # 如果顺序切换失败，尝试随机切换
            clash_client.switch_to_random_proxy("GLOBAL")
        
        # 等待一下让节点切换生效
        import time
        time.sleep(1)

