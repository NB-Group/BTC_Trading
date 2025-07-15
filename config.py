# config.py - 系统全局配置文件
# 此文件包含整个BTC交易系统的共享设置。

import os
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv()

# ----------------- 通用设置 -----------------
# 交易设置
DEMO_MODE = os.getenv('DEMO_MODE', 'true').lower() == 'true'

# ----------------- Defaults -----------------
DEFAULTS = {
    'proxy_url': os.getenv('PROXY_URL'),  # 代理地址，如果不需要则设为 None
    'model_name': 'btc-crossover-regression-v1', # 指定要用于实时信号的主模型
}

# ==============================================================================
# 社交媒体与情报分析 (Social Media & Intelligence)
# ==============================================================================
SOCIAL_MEDIA = {
    'twitter_query': "Bitcoin",  # 用于推文搜索的通用关键词
    'twitter_limit': 15,         # 每次通用搜索获取的推文数量
}

# ==============================================================================
# 重点监控的Twitter影响力人物
# ==============================================================================
# 在这里添加您想特别关注的Twitter用户名
INFLUENTIAL_TWITTER_USERS = [
    # --- 核心人物 (Core Figures) ---
    "elonmusk",          # 伊隆·马斯克 (技术与市场情绪)
    "realDonaldTrump",   # 唐纳德·特朗普 (宏观政策与市场情绪)
    "nayibbukele",       # Nayib Bukele (萨尔瓦多总统，国家级采用)
    
    # --- 比特币巨鲸与倡导者 (Bitcoin Whales & Advocates) ---
    "saylor",            # Michael Saylor (MicroStrategy CEO, 比特币巨鲸)
    "VitalikButerin",    # Vitalik Buterin (以太坊创始人, 加密领域思想领袖)
    "aantonop",          # Andreas M. Antonopoulos (比特币教育家和倡导者)

    # --- 分析师与模型创建者 (Analysts & Model Creators) ---
    "PlanB",             # PlanB (著名比特币S2F模型创建者)
    "woonomic",          # Willy Woo (顶级链上数据分析师)
    "RaoulGMI",          # Raoul Pal (宏观经济学家, Real Vision CEO)
    "KralowTom",         # Tom Kralow (著名技术分析师)

    # --- 机构与风险投资 (Institutions & Venture Capital) ---
    "CathieDWood",       # Cathie Wood (ARK Invest CEO, 知名科技投资者)
    "bhorowitz",         # Ben Horowitz (a16z 联合创始人, 顶级风投)
    "CaitlinLong_",      # Caitlin Long (数字银行 Avanti 创始人, 合规专家)
]

# ==============================================================================
# 宏观经济事件监控关键词
# ==============================================================================
# 在这里添加你想监控的宏观经济事件或主题的搜索关键词
# 系统会使用这些关键词在Twitter上进行搜索
MACRO_ECONOMIC_KEYWORDS = [
    '"Federal Reserve" OR "Fed" interest rates', # 美联储或联邦利率
    '"FOMC meeting" OR "FOMC statement"',        # 联邦公开市场委员会会议或声明
    '"CPI report" OR "inflation data"',          # CPI报告或通胀数据
    '"Jerome Powell" speech',                    # 鲍威尔讲话
    '"non-farm payrolls" OR NFP',                # 非农就业数据
]

# ==============================================================================
# VLM缓存配置
# ==============================================================================
VLM_CACHE = {
    'enabled': True,           # 是否启用缓存
    'cache_hours': 1,          # 缓存有效期（小时）
    'cache_dir': 'cache'       # 缓存目录
}

# ==============================================================================
# API密钥管理
# ==============================================================================
# 从环境变量加载API密钥
API_KEYS = {
    "okx": {
        "api_key": os.getenv('OKX_API_KEY'),
        "secret_key": os.getenv('OKX_SECRET_KEY'),
        "passphrase": os.getenv('OKX_PASSPHRASE'),
    },
    "deepseek": {
        'api_key': os.getenv('DEEPSEEK_API_KEY'),
        'base_url': os.getenv('DEEPSEEK_BASE_URL', "https://api.siliconflow.cn/v1"),
        'model': os.getenv('DEEPSEEK_MODEL', "deepseek-ai/DeepSeek-R1")
    },
    "twitter": {
        "x-rapidapi-key": os.getenv('TWITTER_RAPIDAPI_KEY'),
        "x-rapidapi-host": os.getenv('TWITTER_RAPIDAPI_HOST', "twitter154.p.rapidapi.com")
    }
}

# ==============================================================================
# 期货交易设置
# ==============================================================================
FUTURES = {
    'trade_symbol': 'BTC-USDT-SWAP', # 期货交易对 (永续合约)
    'leverage': 2,                   # 默认杠杆倍数 (降低风险)
    'margin_mode': 'isolated',       # 保证金模式: 'isolated' (逐仓) or 'cross' (全仓)
    'hedge_mode': False,             # 是否为双向持仓（hedge），False为单向持仓
} 