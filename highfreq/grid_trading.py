"""
网格交易策略
- 在价格区间内设置多个买卖订单
- 价格下跌时买入，上涨时卖出
- 赚取价格波动差价
"""
import time
import sys
import os
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_predictor import data as d
from btc_predictor.utils import LOGGER
from execution_engine.okx_trader import OKXTrader
import config


class GridTrader:
    """网格交易器"""
    
    def __init__(
        self,
        symbol: str = "BTC-USDT-SWAP",
        grid_count: int = 10,
        price_range_pct: float = 0.05,  # 价格区间：±5%
        order_amount: float = 0.01,  # 每格订单金额（BTC）
        max_capital_usd: float = 500.0,
        leverage: int = 3,
        demo_mode: bool = False,
    ):
        """
        初始化网格交易器
        
        Args:
            symbol: 交易对
            grid_count: 网格数量
            price_range_pct: 价格区间（百分比）
            order_amount: 每格订单金额（BTC）
            max_capital_usd: 最大资金
            leverage: 杠杆倍数
            demo_mode: 是否模拟模式
        """
        self.symbol = symbol
        self.grid_count = grid_count
        self.price_range_pct = price_range_pct
        self.order_amount = order_amount
        self.max_capital_usd = max_capital_usd
        self.leverage = leverage
        self.demo_mode = demo_mode
        
        # 初始化交易器
        self.trader = OKXTrader(demo_mode=demo_mode)
        
        # 确保交易对在配置中
        if symbol not in self.trader.trade_symbols:
            LOGGER.warning(f"[GRID] 交易对 {symbol} 不在配置中，添加到交易列表")
            self.trader.trade_symbols.append(symbol)
        
        # 网格状态
        self.grid_levels: List[Dict] = []  # 网格价格水平
        self.current_price: float = 0.0
        self.grid_orders: Dict[str, Dict] = {}  # 当前挂单
        self.filled_orders: List[Dict] = []  # 已成交订单
        
        # 交易记录
        self.trade_records: List[Dict] = []
        self.initial_balance = 0.0
        pnl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'highfreq')
        os.makedirs(pnl_dir, exist_ok=True)
        self.pnl_file = os.path.join(pnl_dir, 'grid_pnl_log.json')
        
        # 设置交易所
        d.DATA_CONFIG['exchange'] = 'okx'
        d._exchange = None
        
        # 记录初始余额
        try:
            self.initial_balance = self.trader.get_balance('USDT')
        except:
            pass
        
        LOGGER.info(
            f"[GRID] 初始化完成: symbol={symbol}, "
            f"grid_count={grid_count}, price_range={price_range_pct*100:.1f}%, "
            f"max_capital=${max_capital_usd}, leverage={leverage}x"
        )
    
    def _get_current_price(self) -> Optional[float]:
        """获取当前价格"""
        try:
            ex = self.trader.exchange
            if ex is None:
                return None
            ticker = ex.fetch_ticker(self.symbol)
            return float(ticker['last'])
        except Exception as e:
            LOGGER.error(f"[GRID] 获取价格失败: {e}")
            return None
    
    def _init_grid_levels(self):
        """初始化网格价格水平"""
        current_price = self._get_current_price()
        if current_price is None:
            LOGGER.error("[GRID] 无法获取当前价格，无法初始化网格")
            return False
        
        self.current_price = current_price
        
        # 计算价格区间
        upper_price = current_price * (1 + self.price_range_pct)
        lower_price = current_price * (1 - self.price_range_pct)
        
        # 计算网格间距
        price_step = (upper_price - lower_price) / self.grid_count
        
        # 生成网格水平
        self.grid_levels = []
        for i in range(self.grid_count + 1):
            price = lower_price + i * price_step
            self.grid_levels.append({
                'price': price,
                'level': i,
                'type': 'buy' if i < self.grid_count / 2 else 'sell',  # 下半部分买入，上半部分卖出
                'order_id': None,
                'filled': False,
            })
        
        LOGGER.info(
            f"[GRID] 网格初始化完成: "
            f"价格区间=[{lower_price:.2f}, {upper_price:.2f}], "
            f"网格数={len(self.grid_levels)}, 间距={price_step:.2f}"
        )
        return True
    
    def _place_grid_orders(self):
        """放置网格订单"""
        if not self.grid_levels:
            return
        
        current_price = self._get_current_price()
        if current_price is None:
            return
        
        self.current_price = current_price
        
        # 找到当前价格所在的网格区间
        current_level = None
        for i, level in enumerate(self.grid_levels):
            if level['price'] <= current_price < level['price'] + (self.grid_levels[1]['price'] - self.grid_levels[0]['price']):
                current_level = i
                break
        
        if current_level is None:
            return
        
        # 在下半部分放置买单，在上半部分放置卖单
        for i, level in enumerate(self.grid_levels):
            # 跳过当前价格附近的网格（避免立即成交）
            if abs(i - current_level) <= 1:
                continue
            
            # 如果已有订单，跳过
            if level.get('order_id') is not None:
                continue
            
            # 如果已成交，跳过
            if level.get('filled', False):
                continue
            
            # 放置买单（下半部分）
            if i < current_level and level['type'] == 'buy':
                try:
                    # 计算订单数量（转换为合约张数）
                    # 假设1张 = 100 USD面值
                    contract_value_usd = 100.0
                    contracts = int((self.order_amount * current_price) / contract_value_usd)
                    if contracts < 1:
                        contracts = 1
                    
                    # 放置限价买单
                    order = self.trader.exchange.create_limit_buy_order(
                        self.symbol,
                        contracts,
                        level['price'],
                        params={'tdMode': 'cross', 'leverage': self.leverage}
                    )
                    
                    level['order_id'] = order['id']
                    self.grid_orders[order['id']] = {
                        'level': i,
                        'price': level['price'],
                        'type': 'buy',
                        'amount': contracts,
                    }
                    
                    LOGGER.info(
                        f"[GRID] 放置买单: 价格={level['price']:.2f}, "
                        f"数量={contracts}张, 订单ID={order['id']}"
                    )
                except Exception as e:
                    LOGGER.error(f"[GRID] 放置买单失败: {e}")
            
            # 放置卖单（上半部分，需要先有持仓）
            elif i > current_level and level['type'] == 'sell':
                # 检查是否有持仓
                position = self.trader.get_position(self.symbol)
                if position and float(position.get('pos', 0)) > 0:
                    try:
                        # 计算订单数量
                        contract_value_usd = 100.0
                        contracts = int((self.order_amount * current_price) / contract_value_usd)
                        if contracts < 1:
                            contracts = 1
                        
                        # 放置限价卖单
                        order = self.trader.exchange.create_limit_sell_order(
                            self.symbol,
                            contracts,
                            level['price'],
                            params={'tdMode': 'cross', 'reduceOnly': True}
                        )
                        
                        level['order_id'] = order['id']
                        self.grid_orders[order['id']] = {
                            'level': i,
                            'price': level['price'],
                            'type': 'sell',
                            'amount': contracts,
                        }
                        
                        LOGGER.info(
                            f"[GRID] 放置卖单: 价格={level['price']:.2f}, "
                            f"数量={contracts}张, 订单ID={order['id']}"
                        )
                    except Exception as e:
                        LOGGER.error(f"[GRID] 放置卖单失败: {e}")
    
    def _check_filled_orders(self):
        """检查已成交订单"""
        if not self.grid_orders:
            return
        
        try:
            # 获取所有订单状态
            for order_id, order_info in list(self.grid_orders.items()):
                try:
                    order = self.trader.exchange.fetch_order(order_id, self.symbol)
                    
                    if order['status'] == 'closed':
                        # 订单已成交
                        level = self.grid_levels[order_info['level']]
                        level['filled'] = True
                        level['order_id'] = None
                        
                        # 记录成交
                        self.filled_orders.append({
                            'order_id': order_id,
                            'price': order_info['price'],
                            'type': order_info['type'],
                            'amount': order_info['amount'],
                            'filled_price': order.get('price', order_info['price']),
                            'timestamp': datetime.now().isoformat(),
                        })
                        
                        # 移除订单
                        del self.grid_orders[order_id]
                        
                        LOGGER.info(
                            f"[GRID] 订单成交: {order_info['type']}, "
                            f"价格={order_info['price']:.2f}, "
                            f"数量={order_info['amount']}张"
                        )
                        
                        # 如果买单成交，在更高价格放置卖单
                        if order_info['type'] == 'buy':
                            # 找到更高价格的网格水平
                            for higher_level in self.grid_levels:
                                if higher_level['price'] > order_info['price'] and not higher_level.get('filled', False):
                                    # 放置卖单
                                    try:
                                        sell_order = self.trader.exchange.create_limit_sell_order(
                                            self.symbol,
                                            order_info['amount'],
                                            higher_level['price'],
                                            params={'tdMode': 'cross', 'reduceOnly': True}
                                        )
                                        higher_level['order_id'] = sell_order['id']
                                        self.grid_orders[sell_order['id']] = {
                                            'level': self.grid_levels.index(higher_level),
                                            'price': higher_level['price'],
                                            'type': 'sell',
                                            'amount': order_info['amount'],
                                        }
                                        LOGGER.info(
                                            f"[GRID] 买单成交后放置卖单: "
                                            f"价格={higher_level['price']:.2f}"
                                        )
                                    except Exception as e:
                                        LOGGER.error(f"[GRID] 放置卖单失败: {e}")
                                    break
                        
                        # 如果卖单成交，在更低价格放置买单
                        elif order_info['type'] == 'sell':
                            # 找到更低价格的网格水平
                            for lower_level in reversed(self.grid_levels):
                                if lower_level['price'] < order_info['price'] and not lower_level.get('filled', False):
                                    # 放置买单
                                    try:
                                        buy_order = self.trader.exchange.create_limit_buy_order(
                                            self.symbol,
                                            order_info['amount'],
                                            lower_level['price'],
                                            params={'tdMode': 'cross', 'leverage': self.leverage}
                                        )
                                        lower_level['order_id'] = buy_order['id']
                                        self.grid_orders[buy_order['id']] = {
                                            'level': self.grid_levels.index(lower_level),
                                            'price': lower_level['price'],
                                            'type': 'buy',
                                            'amount': order_info['amount'],
                                        }
                                        LOGGER.info(
                                            f"[GRID] 卖单成交后放置买单: "
                                            f"价格={lower_level['price']:.2f}"
                                        )
                                    except Exception as e:
                                        LOGGER.error(f"[GRID] 放置买单失败: {e}")
                                    break
                    
                    elif order['status'] == 'canceled':
                        # 订单已取消
                        level = self.grid_levels[order_info['level']]
                        level['order_id'] = None
                        del self.grid_orders[order_id]
                        LOGGER.info(f"[GRID] 订单已取消: {order_id}")
                        
                except Exception as e:
                    LOGGER.warning(f"[GRID] 检查订单状态失败: {e}")
                    
        except Exception as e:
            LOGGER.error(f"[GRID] 检查成交订单失败: {e}")
    
    def run_once(self):
        """执行一次网格交易循环"""
        try:
            # 检查已成交订单
            self._check_filled_orders()
            
            # 放置网格订单
            self._place_grid_orders()
            
        except Exception as e:
            LOGGER.error(f"[GRID] 执行失败: {e}")
    
    def run_continuous(self, interval_sec: float = 10.0, duration_min: int = 60):
        """持续运行网格交易"""
        LOGGER.info(f"[GRID] 开始持续交易: 间隔={interval_sec}秒, 时长={duration_min}分钟")
        
        # 初始化网格
        if not self._init_grid_levels():
            LOGGER.error("[GRID] 网格初始化失败，无法开始交易")
            return
        
        start_time = time.time()
        end_time = start_time + duration_min * 60
        
        iteration = 0
        while time.time() < end_time:
            iteration += 1
            self.run_once()
            
            # 每100次循环输出一次状态
            if iteration % 100 == 0:
                current_balance = self.trader.get_balance('USDT')
                elapsed_min = (time.time() - start_time) / 60
                LOGGER.info(
                    f"[GRID] 运行中: {elapsed_min:.1f}分钟, "
                    f"当前价格={self.current_price:.2f}, "
                    f"挂单数={len(self.grid_orders)}, "
                    f"已成交={len(self.filled_orders)}, "
                    f"余额=${current_balance:.2f}"
                )
            
            time.sleep(interval_sec)
        
        # 保存交易记录
        self._save_pnl_log()
        
        LOGGER.info("[GRID] 交易结束")
    
    def _save_pnl_log(self):
        """保存盈亏日志"""
        try:
            current_balance = self.trader.get_balance('USDT')
            total_pnl = current_balance - self.initial_balance
            total_pnl_pct = (total_pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0
            
            log_data = {
                'initial_balance': self.initial_balance,
                'current_balance': current_balance,
                'total_pnl_usd': total_pnl,
                'total_pnl_pct': total_pnl_pct,
                'filled_orders': self.filled_orders,
                'grid_levels': [
                    {
                        'price': level['price'],
                        'type': level['type'],
                        'filled': level.get('filled', False),
                    }
                    for level in self.grid_levels
                ],
                'last_updated': datetime.now().isoformat(),
            }
            
            with open(self.pnl_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            LOGGER.info(
                f"[GRID] 盈亏统计: 初始=${self.initial_balance:.2f}, "
                f"当前=${current_balance:.2f}, "
                f"盈亏=${total_pnl:.2f} ({total_pnl_pct:.2f}%)"
            )
            
        except Exception as e:
            LOGGER.error(f"[GRID] 保存盈亏日志失败: {e}")

