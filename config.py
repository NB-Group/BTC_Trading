# config.py - 系统全局配置文件
# 此文件包含整个BTC交易系统的共享设置。

import os
from typing import List
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv()


def _parse_email_list(raw: str) -> List[str]:
    """
    将环境变量中的收件人字符串解析为邮件地址列表。
    支持使用逗号、分号或换行分隔多个邮箱。
    """
    if not raw:
        return []

    separators = [',', ';', '\n']
    for sep in separators[1:]:
        raw = raw.replace(sep, separators[0])

    return [email.strip() for email in raw.split(separators[0]) if email.strip()]

# ----------------- 通用设置 -----------------
# 交易设置
DEMO_MODE = os.getenv('DEMO_MODE', 'false').lower() == 'true'

# Email Configuration
EMAIL_CONFIG = {
    'enabled': os.getenv('EMAIL_ENABLED', 'false').lower() == 'true',
    'smtp_server': os.getenv('EMAIL_SMTP_SERVER', 'smtp.qq.com'),
    'smtp_port': int(os.getenv('EMAIL_SMTP_PORT', '587')),
    'from_email': os.getenv('EMAIL_FROM', ''),
    'to_emails': _parse_email_list(os.getenv('EMAIL_TO', '')),
    'auth_code': os.getenv('EMAIL_AUTH_CODE', ''),
    'use_tls': os.getenv('EMAIL_USE_TLS', 'true').lower() == 'true'
}

# ----------------- Defaults -----------------
DEFAULTS = {
    'proxy_url': os.getenv('PROXY_URL'),  # 代理地址，如果不需要则设为 None
    # 默认实时信号主模型，改为使用 gpt-5 系列模型（可通过环境变量覆盖）
    'model_name': os.getenv('DEFAULT_MODEL_NAME', 'gpt-5.1'),
}

# ================= 交易决策规则（可调） =================
# - vlm_priority_weight: DeepSeek综合决策时，VLM结论的权重(0-1)。
# - cautious_rebound: 当识别为下跌后的弱反弹时，仅允许“小仓试探/等待突破确认”。
# - probe_position_ratio: 开启试探模式时，相比常规建议仓位的比例（0-1）。
# - strict_long_trigger: 要求VLM给出的做多信号必须带“价格高于X并放量/收盘确认”等条件才视为有效。
DECISION_RULES = {
    'vlm_priority_weight': float(os.getenv('VLM_PRIORITY_WEIGHT', '0.7')),
    'cautious_rebound': os.getenv('CAUTIOUS_REBOUND', 'true').lower() == 'true',
    'probe_position_ratio': float(os.getenv('PROBE_POSITION_RATIO', '0.3')),
    'strict_long_trigger': os.getenv('STRICT_LONG_TRIGGER', 'true').lower() == 'true',
    # 允许仅凭VLM做单（其余来源仅做反证）。
    'vlm_solo_trade': os.getenv('VLM_SOLO_TRADE', 'true').lower() == 'true',
    # 是否启用“统一Gemini”决策路径（VLM+LLM合并，直接用多模态）
    'use_unified_gemini': os.getenv('USE_UNIFIED_GEMINI', 'true').lower() == 'true',
    # 统一路径失败时是否自动回退到原先“VLM分析 + DeepSeek决策”路径
    'unified_fallback_enabled': os.getenv('UNIFIED_FALLBACK_ENABLED', 'true').lower() == 'true',
    # 是否启用 gpt-5.1 对 Gemini 决策的二次审核
    'enable_gpt_reviewer': os.getenv('ENABLE_GPT_REVIEWER', 'true').lower() == 'true',
}

# ----------------- 自动更新与热重载 -----------------
AUTO_UPDATE = {
    'enabled': os.getenv('AUTO_UPDATE_ENABLED', 'false').lower() == 'true',
    'interval_seconds': int(os.getenv('AUTO_UPDATE_INTERVAL', '300')),
    'branch': os.getenv('AUTO_UPDATE_BRANCH', 'main'),
    # update_strategy: 'hard_reset' | 'pull_ff_only' | 'pull_rebase' | 'pull_merge'
    'update_strategy': os.getenv('AUTO_UPDATE_STRATEGY', 'hard_reset'),
    # 保护本地修改：若存在未提交/未追踪文件则不更新
    'protect_local_changes': os.getenv('AUTO_UPDATE_PROTECT_LOCAL', 'true').lower() == 'true',
}

# ----------------- 开机自启动 -----------------
AUTO_START = {
    'enabled': os.getenv('AUTO_START_ENABLED', 'true').lower() == 'true',  # 默认开启
    'task_name': os.getenv('AUTO_START_TASK_NAME', 'BTC_Trading_AutoStart'),
    'args': os.getenv('AUTO_START_ARGS', ''),  # 例如 "--now"
    'conda_env': os.getenv('AUTO_START_CONDA_ENV', 'k-line'), # 指定要在此Conda环境中运行
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
    'enabled': False,           # 是否启用缓存
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
        'model': os.getenv('DEEPSEEK_MODEL', "gpt-5.1"),
    },
    "gemini": {
        'api_key': os.getenv('GEMINI_API_KEY'),
        'base_url': os.getenv('GEMINI_BASE_URL', 'https://jeniya.cn/v1'),
        'model': os.getenv('GEMINI_MODEL', 'gpt-5.2'),
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
    'trade_symbols': ['BTC-USDT-SWAP', 'ETH-USDT-SWAP'], # 升级为多币种列表
    'leverage': 3,                   # 默认杠杆倍数 (推荐 3x)
    'margin_mode': 'cross',          # 保证金模式: 'isolated' (逐仓) or 'cross' (全仓) - 默认全仓
    'hedge_mode': False,             # 是否为双向持仓（hedge），False为单向持仓
    # 为了避免把账户所有可用保证金一次性占满，限制最大可用仓位比例。
    # 例如 0.9 表示最多只使用理论可开张数的 90%，留出保证金用于手续费与维持保证金。
    'max_position_usage': float(os.getenv('MAX_POSITION_USAGE', '0.9')),
} 