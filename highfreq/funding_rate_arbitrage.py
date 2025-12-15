"""
资金费率套利策略
- 监控永续合约资金费率
- 当资金费率足够高时，通过现货+期货对冲赚取资金费率
- 每8小时收取一次资金费率
"""
import time
import sys
import os
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_predictor import data as d
from btc_predictor.utils import LOGGER
from execution_engine.okx_trader import OKXTrader
import config


class FundingRateArbitrage:
    """资金费率套利交易器"""
    
    def __init__(
        self,
        spot_symbol: str = "BTC/USDT",  # 现货交易对
        swap_symbol: str = "BTC-USDT-SWAP",  # 永续合约交易对
        min_funding_rate: float = 0.008,  # 最小资金费率（0.008% = 0.00008，降低阈值）
        max_capital_usd: float = 1000.0,
        leverage: int = 3,
        demo_mode: bool = False,
    ):
        """
        初始化资金费率套利交易器
        
        Args:
            spot_symbol: 现货交易对
            swap_symbol: 永续合约交易对
            min_funding_rate: 最小资金费率（百分比，如0.01表示0.01%）
            max_capital_usd: 最大资金
            leverage: 杠杆倍数
            demo_mode: 是否模拟模式
        """
        self.spot_symbol = spot_symbol
        self.swap_symbol = swap_symbol
        self.min_funding_rate = min_funding_rate / 100  # 转换为小数
        self.max_capital_usd = max_capital_usd
        self.leverage = leverage
        self.demo_mode = demo_mode
        
        # 初始化交易器
        self.trader = OKXTrader(demo_mode=demo_mode)
        
        # 确保交易对在配置中
        if swap_symbol not in self.trader.trade_symbols:
            LOGGER.warning(f"[FR-ARB] 交易对 {swap_symbol} 不在配置中，添加到交易列表")
            self.trader.trade_symbols.append(swap_symbol)
        
        # 持仓状态
        self.spot_position: Optional[Dict] = None  # 现货持仓
        self.swap_position: Optional[Dict] = None  # 期货持仓
        self.entry_funding_rate: float = 0.0  # 入场时的资金费率
        self.entry_time: Optional[datetime] = None
        
        # 交易记录
        self.trade_records: List[Dict] = []
        self.initial_balance = 0.0
        pnl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'highfreq')
        os.makedirs(pnl_dir, exist_ok=True)
        self.pnl_file = os.path.join(pnl_dir, 'funding_rate_arbitrage_log.json')
        
        # 设置交易所
        d.DATA_CONFIG['exchange'] = 'okx'
        d._exchange = None
        
        # 记录初始余额
        try:
            self.initial_balance = self.trader.get_balance('USDT')
        except:
            pass
        
        LOGGER.info(
            f"[FR-ARB] 初始化完成: spot={spot_symbol}, swap={swap_symbol}, "
            f"min_funding_rate={min_funding_rate}%, max_capital=${max_capital_usd}, "
            f"leverage={leverage}x"
        )
    
    def _get_funding_rate(self) -> Optional[float]:
        """获取当前资金费率"""
        try:
            ex = self.trader.exchange
            if ex is None:
                return None
            
            # 获取资金费率信息
            # OKX API: GET /api/v5/public/funding-rate
            try:
                # 使用ccxt的方法获取资金费率
                funding_info = ex.fetch_funding_rate(self.swap_symbol)
                if funding_info:
                    funding_rate = float(funding_info.get('fundingRate', 0))
                    next_funding_time = funding_info.get('nextFundingTime', '')
                    
                    next_funding_str = str(next_funding_time) if next_funding_time else "未知"
                    LOGGER.debug(
                        f"[FR-ARB] 资金费率: {funding_rate*100:.4f}%, "
                        f"下次结算时间: {next_funding_str}"
                    )
                    
                    return funding_rate
            except:
                # 如果fetch_funding_rate不支持，尝试直接调用API
                try:
                    funding_info = ex.public_get_public_funding_rate({
                        'instId': self.swap_symbol
                    })
                    
                    if funding_info and 'data' in funding_info and len(funding_info['data']) > 0:
                        funding_rate = float(funding_info['data'][0].get('fundingRate', 0))
                        next_funding_time = funding_info['data'][0].get('nextFundingTime', '')
                        
                        next_funding_str = str(next_funding_time) if next_funding_time else "未知"
                        LOGGER.debug(
                            f"[FR-ARB] 资金费率: {funding_rate*100:.4f}%, "
                            f"下次结算时间: {next_funding_str}"
                        )
                        
                        return funding_rate
                except Exception as e2:
                    LOGGER.warning(f"[FR-ARB] 通过API获取资金费率失败: {e2}")
            
            LOGGER.warning("[FR-ARB] 无法获取资金费率信息")
            return None
                
        except Exception as e:
            LOGGER.error(f"[FR-ARB] 获取资金费率失败: {e}")
            return None
    
    def _get_spot_price(self) -> Optional[float]:
        """获取现货价格"""
        try:
            ex = self.trader.exchange
            if ex is None:
                LOGGER.warning("[FR-ARB] 交易所实例为None，无法获取现货价格")
                return None
            ticker = ex.fetch_ticker(self.spot_symbol)
            if ticker and 'last' in ticker:
                price = float(ticker['last'])
                LOGGER.debug(f"[FR-ARB] 现货价格: {price:.2f}")
                return price
            else:
                LOGGER.warning(f"[FR-ARB] 现货ticker数据无效: {ticker}")
                return None
        except Exception as e:
            LOGGER.error(f"[FR-ARB] 获取现货价格失败: {e}")
            return None
    
    def _get_swap_price(self) -> Optional[float]:
        """获取期货价格"""
        try:
            ex = self.trader.exchange
            if ex is None:
                LOGGER.warning("[FR-ARB] 交易所实例为None，无法获取期货价格")
                return None
            ticker = ex.fetch_ticker(self.swap_symbol)
            if ticker and 'last' in ticker:
                price = float(ticker['last'])
                LOGGER.debug(f"[FR-ARB] 期货价格: {price:.2f}")
                return price
            else:
                LOGGER.warning(f"[FR-ARB] 期货ticker数据无效: {ticker}")
                return None
        except Exception as e:
            LOGGER.error(f"[FR-ARB] 获取期货价格失败: {e}")
            return None
    
    def _calculate_cost(self, amount_btc: float) -> Dict[str, float]:
        """计算交易成本"""
        # 现货手续费（假设0.1%）
        spot_fee_pct = 0.001
        # 期货手续费（Taker 0.028%）
        swap_fee_pct = 0.00028
        # 资金费率（每8小时一次，一天3次）
        funding_rate_daily = self.entry_funding_rate * 3  # 假设资金费率不变
        
        spot_fee = amount_btc * spot_fee_pct
        swap_fee = amount_btc * swap_fee_pct
        total_fee = spot_fee + swap_fee
        
        # 预期收益（资金费率收入）
        expected_profit = amount_btc * funding_rate_daily
        
        # 净收益
        net_profit = expected_profit - total_fee
        
        return {
            'spot_fee': spot_fee,
            'swap_fee': swap_fee,
            'total_fee': total_fee,
            'expected_profit': expected_profit,
            'net_profit': net_profit,
            'net_profit_pct': (net_profit / amount_btc * 100) if amount_btc > 0 else 0,
        }
    
    def _check_arbitrage_opportunity(self) -> Tuple[bool, Optional[float], Optional[float]]:
        """检查套利机会"""
        funding_rate = self._get_funding_rate()
        if funding_rate is None:
            LOGGER.warning("[FR-ARB] 无法获取资金费率")
            return False, None, None
        
        # 检查资金费率是否足够高（覆盖成本）
        # 成本包括：现货手续费0.1% + 期货手续费0.028% = 0.128%
        # 资金费率需要 > 0.128% / 3 = 0.043% 才能盈利（每天）
        # 单次资金费率需要 > 0.014% 才能盈利
        # 但考虑到实际市场情况，使用用户设置的min_funding_rate（默认0.008%）
        
        # 直接使用用户设置的最小资金费率，不再强制0.014%
        min_required = self.min_funding_rate
        
        LOGGER.debug(
            f"[FR-ARB] 检查套利机会: 当前资金费率={funding_rate*100:.4f}%, "
            f"最小要求={min_required*100:.4f}%, "
            f"资金费率方向={'正' if funding_rate > 0 else '负'}"
        )
        
        if abs(funding_rate) < min_required:
            LOGGER.debug(
                f"[FR-ARB] 资金费率不足: {abs(funding_rate)*100:.4f}% < {min_required*100:.4f}%"
            )
            return False, funding_rate, None
        
        # 获取价格
        spot_price = self._get_spot_price()
        swap_price = self._get_swap_price()
        
        if spot_price is None or swap_price is None:
            return False, funding_rate, None
        
        # 计算价差
        price_diff = swap_price - spot_price
        price_diff_pct = (price_diff / spot_price * 100) if spot_price > 0 else 0
        
        LOGGER.debug(
            f"[FR-ARB] 价格信息: 现货={spot_price:.2f}, 期货={swap_price:.2f}, 价差={price_diff_pct:.4f}%"
        )
        
        # 如果资金费率为正，做多现货+做空期货
        # 如果资金费率为负，做空现货+做多期货（但现货通常不能做空，所以只做资金费率为正的情况）
        if funding_rate > 0:
            # 资金费率为正，做多现货+做空期货
            return True, funding_rate, price_diff_pct
        else:
            # 资金费率为负，理论上可以做空现货+做多期货，但现货通常不能做空
            # 所以只做资金费率为正的情况
            return False, funding_rate, price_diff_pct
    
    def _open_arbitrage_position(self, amount_btc: float):
        """开仓套利"""
        try:
            spot_price = self._get_spot_price()
            swap_price = self._get_swap_price()
            funding_rate = self._get_funding_rate()
            
            if spot_price is None or swap_price is None or funding_rate is None:
                LOGGER.error("[FR-ARB] 无法获取价格或资金费率，无法开仓")
                return False
            
            # 计算成本
            cost_info = self._calculate_cost(amount_btc)
            
            LOGGER.info(
                f"[FR-ARB] 开仓套利: 资金费率={funding_rate*100:.4f}%, "
                f"现货价={spot_price:.2f}, 期货价={swap_price:.2f}, "
                f"数量={amount_btc:.4f} BTC, 预期净收益={cost_info['net_profit_pct']:.4f}%"
            )
            
            # 1. 现货买入
            try:
                # 使用create_order方法，指定现货市场
                spot_order = self.trader.exchange.create_order(
                    self.spot_symbol,
                    'market',
                    'buy',
                    amount_btc,
                    None,
                    params={'tdMode': 'cash'}  # 现货模式
                )
                LOGGER.info(f"[FR-ARB] 现货买入成功: 订单ID={spot_order.get('id', 'N/A')}")
                self.spot_position = {
                    'order_id': spot_order.get('id'),
                    'amount': amount_btc,
                    'price': spot_price,
                    'side': 'long',
                }
            except Exception as e:
                LOGGER.error(f"[FR-ARB] 现货买入失败: {e}")
                return False
            
            # 2. 期货做空（对冲）
            try:
                # 计算合约数量（转换为张数）
                contract_value_usd = 100.0  # BTC永续合约面值
                contracts = int((amount_btc * swap_price) / contract_value_usd)
                if contracts < 1:
                    contracts = 1
                
                swap_order = self.trader.exchange.create_order(
                    self.swap_symbol,
                    'market',
                    'sell',
                    contracts,
                    None,
                    params={
                        'tdMode': 'cross',
                        'leverage': self.leverage,
                    }
                )
                # 安全获取订单ID
                if isinstance(swap_order, dict):
                    swap_order_id = swap_order.get('id', 'N/A')
                elif hasattr(swap_order, 'get'):
                    swap_order_id = swap_order.get('id', 'N/A')
                elif hasattr(swap_order, 'id'):
                    swap_order_id = swap_order.id
                else:
                    swap_order_id = 'N/A'
                
                LOGGER.info(f"[FR-ARB] 期货做空成功: 订单ID={swap_order_id}, 数量={contracts}张")
                self.swap_position = {
                    'order_id': swap_order_id if swap_order_id != 'N/A' else None,
                    'amount': contracts,
                    'price': swap_price,
                    'side': 'short',
                }
            except Exception as e:
                LOGGER.error(f"[FR-ARB] 期货做空失败: {e}")
                # 如果期货开仓失败，需要平掉现货
                try:
                    self.trader.exchange.create_order(
                        self.spot_symbol,
                        'market',
                        'sell',
                        amount_btc,
                        None,
                        params={'tdMode': 'cash'}
                    )
                    LOGGER.info("[FR-ARB] 期货开仓失败，已平掉现货")
                except:
                    pass
                return False
            
            # 记录入场信息
            self.entry_funding_rate = funding_rate
            self.entry_time = datetime.now()
            
            LOGGER.info(
                f"[FR-ARB] 套利开仓成功: 现货={amount_btc:.4f} BTC @ {spot_price:.2f}, "
                f"期货={contracts}张 @ {swap_price:.2f}, 资金费率={funding_rate*100:.4f}%"
            )
            
            return True
            
        except Exception as e:
            LOGGER.error(f"[FR-ARB] 开仓套利失败: {e}")
            return False
    
    def _close_arbitrage_position(self, reason: str = "manual"):
        """平仓套利"""
        try:
            if not self.spot_position or not self.swap_position:
                LOGGER.warning("[FR-ARB] 没有持仓，无法平仓")
                return False
            
            spot_price = self._get_spot_price()
            swap_price = self._get_swap_price()
            
            if spot_price is None or swap_price is None:
                LOGGER.error("[FR-ARB] 无法获取价格，无法平仓")
                return False
            
            # 计算盈亏
            spot_pnl = (spot_price - self.spot_position['price']) * self.spot_position['amount']
            swap_pnl = (self.swap_position['price'] - swap_price) * (self.swap_position['amount'] * 100 / swap_price)
            total_pnl = spot_pnl + swap_pnl
            
            # 1. 平掉现货（卖出）
            try:
                spot_close_order = self.trader.exchange.create_order(
                    self.spot_symbol,
                    'market',
                    'sell',
                    self.spot_position['amount'],
                    None,
                    params={'tdMode': 'cash'}
                )
                LOGGER.info(f"[FR-ARB] 现货平仓成功: 订单ID={spot_close_order.get('id', 'N/A')}")
            except Exception as e:
                LOGGER.error(f"[FR-ARB] 现货平仓失败: {e}")
            
            # 2. 平掉期货（买入平空）
            try:
                swap_close_order = self.trader.exchange.create_order(
                    self.swap_symbol,
                    'market',
                    'buy',
                    self.swap_position['amount'],
                    None,
                    params={
                        'tdMode': 'cross',
                        'reduceOnly': True,
                    }
                )
                LOGGER.info(f"[FR-ARB] 期货平仓成功: 订单ID={swap_close_order.get('id', 'N/A')}")
            except Exception as e:
                LOGGER.error(f"[FR-ARB] 期货平仓失败: {e}")
            
            # 计算持仓时间
            hold_time = (datetime.now() - self.entry_time).total_seconds() if self.entry_time else 0
            
            # 记录交易
            trade_record = {
                'trade_id': len(self.trade_records) + 1,
                'timestamp': datetime.now().isoformat(),
                'entry_time': self.entry_time.isoformat() if self.entry_time else None,
                'exit_time': datetime.now().isoformat(),
                'hold_seconds': hold_time,
                'entry_funding_rate': self.entry_funding_rate,
                'spot_entry_price': self.spot_position['price'],
                'swap_entry_price': self.swap_position['price'],
                'spot_exit_price': spot_price,
                'swap_exit_price': swap_price,
                'spot_pnl': spot_pnl,
                'swap_pnl': swap_pnl,
                'total_pnl': total_pnl,
                'reason': reason,
            }
            self.trade_records.append(trade_record)
            
            LOGGER.info(
                f"[FR-ARB] 套利平仓成功: 原因={reason}, "
                f"持仓时间={hold_time/3600:.2f}小时, "
                f"总盈亏=${total_pnl:.2f}"
            )
            
            # 重置持仓
            self.spot_position = None
            self.swap_position = None
            self.entry_funding_rate = 0.0
            self.entry_time = None
            
            return True
            
        except Exception as e:
            LOGGER.error(f"[FR-ARB] 平仓套利失败: {e}")
            return False
    
    def _check_exit_conditions(self) -> Tuple[bool, str]:
        """检查平仓条件"""
        if not self.spot_position or not self.swap_position:
            return False, ""
        
        # 1. 资金费率变化（如果资金费率降到很低，平仓）
        current_funding_rate = self._get_funding_rate()
        if current_funding_rate is not None:
            # 如果资金费率降到原来的50%以下，考虑平仓
            if current_funding_rate < self.entry_funding_rate * 0.5:
                return True, "资金费率下降"
        
        # 2. 持仓时间过长（超过24小时，资金费率可能变化）
        if self.entry_time:
            hold_time = (datetime.now() - self.entry_time).total_seconds()
            if hold_time > 24 * 3600:  # 24小时
                return True, "持仓时间过长"
        
        # 3. 价差过大（可能风险增加）
        spot_price = self._get_spot_price()
        swap_price = self._get_swap_price()
        if spot_price and swap_price:
            price_diff_pct = abs((swap_price - spot_price) / spot_price * 100)
            if price_diff_pct > 2.0:  # 价差超过2%
                return True, "价差过大"
        
        return False, ""
    
    def _calculate_position_size(self) -> float:
        """计算仓位大小"""
        try:
            balance = self.trader.get_balance('USDT')
            available_usd = min(float(balance), self.max_capital_usd)
            
            # 获取现货价格
            spot_price = self._get_spot_price()
            if spot_price is None:
                return 0.0
            
            # 计算可以买入的BTC数量（使用50%的资金，留50%作为保证金）
            amount_usd = available_usd * 0.5
            amount_btc = amount_usd / spot_price
            
            # 限制最小和最大数量
            min_btc = 0.001  # 最小0.001 BTC
            max_btc = available_usd / spot_price * 0.8  # 最大80%资金
            
            amount_btc = max(min_btc, min(amount_btc, max_btc))
            
            LOGGER.info(
                f"[FR-ARB] 仓位计算: 可用=${available_usd:.2f}, "
                f"现货价={spot_price:.2f}, 数量={amount_btc:.4f} BTC"
            )
            
            return amount_btc
            
        except Exception as e:
            LOGGER.error(f"[FR-ARB] 计算仓位大小失败: {e}")
            return 0.0
    
    def run_once(self):
        """执行一次套利检查"""
        try:
            # 如果已有持仓，检查平仓条件
            if self.spot_position and self.swap_position:
                should_exit, reason = self._check_exit_conditions()
                if should_exit:
                    self._close_arbitrage_position(reason)
                return
            
            # 如果没有持仓，检查套利机会
            has_opportunity, funding_rate, price_diff_pct = self._check_arbitrage_opportunity()
            
            if has_opportunity:
                # 计算仓位大小
                amount_btc = self._calculate_position_size()
                
                if amount_btc > 0:
                    # 开仓套利
                    self._open_arbitrage_position(amount_btc)
                else:
                    LOGGER.warning("[FR-ARB] 仓位大小为0，无法开仓")
            else:
                # 输出详细信息，帮助调试
                if funding_rate is not None:
                    price_diff_str = f"{price_diff_pct:.4f}%" if price_diff_pct is not None else "未知"
                    min_required = self.min_funding_rate * 100
                    LOGGER.info(
                        f"[FR-ARB] 无套利机会: 资金费率={funding_rate*100:.4f}%, "
                        f"最小要求={min_required:.4f}%, "
                        f"价差={price_diff_str}, "
                        f"资金费率方向={'正(可做)' if funding_rate > 0 else '负(不可做)'}"
                    )
                else:
                    LOGGER.warning("[FR-ARB] 无法获取资金费率，跳过本次检查")
                    
        except Exception as e:
            LOGGER.error(f"[FR-ARB] 执行失败: {e}")
    
    def run_continuous(self, interval_sec: float = 60.0, duration_min: int = 1440):
        """持续运行资金费率套利"""
        LOGGER.info(f"[FR-ARB] 开始持续交易: 间隔={interval_sec}秒, 时长={duration_min}分钟")
        
        start_time = time.time()
        end_time = start_time + duration_min * 60
        
        iteration = 0
        while time.time() < end_time:
            iteration += 1
            self.run_once()
            
            # 每10次循环输出一次状态（更频繁，便于观察）
            if iteration % 10 == 0:
                current_balance = self.trader.get_balance('USDT')
                elapsed_min = (time.time() - start_time) / 60
                funding_rate = self._get_funding_rate()
                
                status = "持仓中" if (self.spot_position and self.swap_position) else "等待机会"
                LOGGER.info(
                    f"[FR-ARB] 运行中: {elapsed_min:.1f}分钟, "
                    f"状态={status}, "
                    f"资金费率={funding_rate*100:.4f}% (当前) / {self.entry_funding_rate*100:.4f}% (入场) "
                    if funding_rate and self.entry_funding_rate else f"资金费率={funding_rate*100:.4f}% (当前) "
                    if funding_rate else "资金费率=未知, ",
                    f"余额=${current_balance:.2f}, "
                    f"交易次数={len(self.trade_records)}"
                )
            
            time.sleep(interval_sec)
        
        # 如果有持仓，平仓
        if self.spot_position and self.swap_position:
            self._close_arbitrage_position("策略结束")
        
        # 保存交易记录
        self._save_pnl_log()
        
        LOGGER.info("[FR-ARB] 交易结束")
    
    def _save_pnl_log(self):
        """保存盈亏日志"""
        try:
            current_balance = self.trader.get_balance('USDT')
            total_pnl = current_balance - self.initial_balance
            total_pnl_pct = (total_pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0
            
            # 计算总盈亏（从交易记录）
            total_trade_pnl = sum(trade.get('total_pnl', 0) for trade in self.trade_records)
            
            log_data = {
                'initial_balance': self.initial_balance,
                'current_balance': current_balance,
                'total_pnl_usd': total_pnl,
                'total_pnl_pct': total_pnl_pct,
                'total_trade_pnl': total_trade_pnl,
                'total_trades': len(self.trade_records),
                'trades': self.trade_records,
                'last_updated': datetime.now().isoformat(),
            }
            
            with open(self.pnl_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            LOGGER.info(
                f"[FR-ARB] 盈亏统计: 初始=${self.initial_balance:.2f}, "
                f"当前=${current_balance:.2f}, "
                f"盈亏=${total_pnl:.2f} ({total_pnl_pct:.2f}%), "
                f"交易次数={len(self.trade_records)}"
            )
            
        except Exception as e:
            LOGGER.error(f"[FR-ARB] 保存盈亏日志失败: {e}")

