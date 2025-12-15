"""快速检查当前资金费率"""
from btc_predictor import data as d
d.DATA_CONFIG['exchange'] = 'okx'
d._exchange = None

from btc_predictor.data import get_exchange

ex = get_exchange()
if ex:
    try:
        funding = ex.fetch_funding_rate('BTC-USDT-SWAP')
        if funding:
            rate = funding.get('fundingRate', 0)
            next_time = funding.get('nextFundingTime', '未知')
            print(f'当前资金费率: {rate*100:.4f}%')
            print(f'下次结算时间: {next_time}')
            print(f'资金费率方向: {"正(可做)" if rate > 0 else "负(不可做)"}')
            print(f'最小要求: 0.014% (单次)')
            print(f'是否满足: {"是" if abs(rate) >= 0.00014 else "否"}')
        else:
            print('无法获取资金费率')
    except Exception as e:
        print(f'获取资金费率失败: {e}')
else:
    print('交易所初始化失败')

