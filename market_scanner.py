# market_scanner.py
"""
这个模块负责扫描整个加密货币市场，以发现潜在的交易机会。
它会获取交易所的所有交易对，并使用轻量级指标来筛选出最有可能出现价格大幅波动的币种。
"""

from typing import List, Dict, Any
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from btc_predictor.utils import LOGGER
from btc_predictor.data import get_exchange

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_markets_with_retry():
    """带重试机制地获取市场数据"""
    exchange = get_exchange()
    if not exchange:
        LOGGER.error("无法初始化交易所实例，无法获取市场。")
        return None
    try:
        # 加载市场数据，ccxt会自动处理缓存
        markets = exchange.load_markets()
        return markets
    except Exception as e:
        LOGGER.error(f"从交易所加载市场数据失败: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_tickers_with_retry(symbols: List[str]):
    """带重试机制地获取多个交易对的tickers"""
    exchange = get_exchange()
    if not exchange:
        LOGGER.error("无法初始化交易所实例，无法获取tickers。")
        return {}
    try:
        return exchange.fetch_tickers(symbols)
    except Exception as e:
        LOGGER.error(f"获取tickers失败: {e}")
        raise

def scan_for_opportunities(top_n: int = 5, min_quote_volume: float = 5_000_000) -> List[str]:
    """
    扫描OKX交易所的永续合约市场，寻找交易机会。

    Args:
        top_n (int): 要返回的最有潜力的币种数量。
        min_quote_volume (float): 24小时最低交易额（以USDT计），用于过滤掉低流动性币种。

    Returns:
        List[str]: 最有潜力的 top_n 个币种的列表 (例如 ['BTC-USDT-SWAP', 'ETH-USDT-SWAP'])。
    """
    LOGGER.info("开始扫描市场，寻找交易机会...")
    
    exchange = get_exchange()
    if not exchange:
        return []

    try:
        # 1. 获取所有永续合约市场
        markets = fetch_markets_with_retry()
        if not markets:
            return []
            
        swap_symbols = [
            symbol for symbol, market in markets.items() 
            if market.get('swap') and market.get('quote') == 'USDT'
        ]
        LOGGER.info(f"发现 {len(swap_symbols)} 个 USDT 永续合约交易对。")

        # 2. 批量获取 Tickers 并进行交易量筛选
        if not swap_symbols:
            return []
        
        tickers = fetch_tickers_with_retry(swap_symbols)
        
        high_volume_symbols = [
            symbol for symbol, ticker in tickers.items()
            if ticker and ticker.get('quoteVolume') and ticker['quoteVolume'] > min_quote_volume
        ]
        LOGGER.info(f"经过筛选 (24h交易额 > ${min_quote_volume:,.0f}), 剩余 {len(high_volume_symbols)} 个高流动性币种。")

        # 3. 分析每个币种的动能和成交量
        candidates = []
        for symbol in high_volume_symbols:
            try:
                # 获取最近24小时的1小时K线数据
                ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=24)
                if not ohlcv or len(ohlcv) < 24:
                    continue

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # 计算动能：最近12小时的价格变化百分比
                price_change_pct = ((df['close'].iloc[-1] - df['close'].iloc[-12]) / df['close'].iloc[-12]) * 100

                # 计算相对成交量：最后一小时的成交量与过去24小时平均成交量的比率
                avg_volume = df['volume'].mean()
                latest_volume = df['volume'].iloc[-1]
                relative_volume = latest_volume / avg_volume if avg_volume > 0 else 1.0

                # 综合得分，动能权重更高
                score = abs(price_change_pct) * 0.7 + relative_volume * 0.3
                
                candidates.append({'symbol': symbol, 'score': score, 'price_change': price_change_pct})
                
            except Exception as e:
                LOGGER.debug(f"分析币种 {symbol} 时出错: {e}")
                continue
        
        # 4. 根据得分排序并选出Top N
        if not candidates:
            LOGGER.warning("未能找到任何符合分析条件的候选币种。")
            return []

        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        
        top_candidates = sorted_candidates[:top_n]
        
        LOGGER.success(f"市场扫描完成，发现 Top {len(top_candidates)} 潜力币种:")
        for cand in top_candidates:
            LOGGER.info(f"  - {cand['symbol']}: 得分={cand['score']:.2f}, 12小时价格变动={cand['price_change']:.2f}%")
            
        return [c['symbol'] for c in top_candidates]

    except Exception as e:
        LOGGER.error(f"市场扫描过程中发生严重错误: {e}", exc_info=True)
        return []

if __name__ == '__main__':
    # 用于独立测试
    from btc_predictor.utils import setup_logger
    setup_logger()
    
    opportunities = scan_for_opportunities()
    print("\n--- 扫描结果 ---")
    if opportunities:
        print("发现以下机会:")
        for opp in opportunities:
            print(f"- {opp}")
    else:
        print("未发现交易机会。")
