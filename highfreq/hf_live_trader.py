"""
高频策略实盘交易模块
- 实时获取订单簿数据
- 生成micro+OFI信号
- 执行交易（支持杠杆）
- 风险控制（止盈止损、时间止损）
"""
import time
import sys
import os
import json
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from decimal import Decimal

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_predictor import data as d
from btc_predictor.utils import LOGGER
from highfreq.hf_orderbook import fetch_orderbook, compute_obi, _enrich_micro_ofi_features
from execution_engine.okx_trader import OKXTrader
import pandas as pd
import config


class HFLiveTrader:
    """高频策略实盘交易器"""
    
    def __init__(
        self,
        symbol: str = "BTC-USDT-SWAP",
        max_capital_usd: float = 500.0,
        bias_long: float = 5e-7,
        bias_short: float = -5e-7,
        ofi_long: float = 0.0,
        ofi_short: float = 0.0,
        tp_pct: float = 0.0010,
        sl_pct: float = 0.0012,
        min_depth_total: float = 2.0,
        max_spread_bps: float = 2.0,
        cooldown_sec: int = 10,
        time_stop_sec: int = 30,
        ofi_span: int = 8,
        use_taker: bool = False,
        demo_mode: bool = False,
    ):
        """
        初始化高频实盘交易器
        
        Args:
            symbol: 交易对（合约格式，如 BTC/USDT:USDT）
            max_capital_usd: 最大使用资金（美元）
            bias_long/bias_short: micro_bias阈值
            ofi_long/ofi_short: OFI阈值
            tp_pct/sl_pct: 止盈止损比例
            min_depth_total: 最小深度过滤
            max_spread_bps: 最大点差过滤（基点）
            cooldown_sec: 冷却时间（秒）
            time_stop_sec: 时间止损（秒）
            ofi_span: OFI平滑窗口
            use_taker: 是否使用Taker订单
            demo_mode: 是否模拟模式
        """
        self.symbol = symbol
        self.max_capital_usd = max_capital_usd
        self.bias_long = bias_long
        self.bias_short = bias_short
        self.ofi_long = ofi_long
        self.ofi_short = ofi_short
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.min_depth_total = min_depth_total
        self.max_spread_bps = max_spread_bps
        self.cooldown_sec = cooldown_sec
        self.time_stop_sec = time_stop_sec
        self.ofi_span = ofi_span
        self.use_taker = use_taker
        self.demo_mode = demo_mode
        
        # 初始化交易器
        self.trader = OKXTrader(demo_mode=demo_mode)
        
        # 确保交易对在配置中
        if symbol not in self.trader.trade_symbols:
            LOGGER.warning(f"[HF-LIVE] 交易对 {symbol} 不在配置中，添加到交易列表")
            self.trader.trade_symbols.append(symbol)
        
        # 状态管理
        self.position = None  # 当前仓位：None, 'LONG', 'SHORT'
        self.entry_price = 0.0
        self.entry_time = None
        self.last_trade_time = None
        self.cooldown_until = None
        
        # 历史数据（用于计算OFI）
        self.orderbook_history = []
        self.max_history = 100  # 保留最近100条记录
        
        # 交易记录和统计
        self.trade_records: List[Dict[str, Any]] = []
        self.initial_balance = 0.0
        # 盈亏日志文件路径
        pnl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'highfreq')
        os.makedirs(pnl_dir, exist_ok=True)
        self.pnl_file = os.path.join(pnl_dir, 'hf_pnl_log.json')
        
        # 限流统计
        self.rate_limit_count = 0  # 限流总次数
        self.rate_limit_last_time = None  # 最后一次限流时间
        
        # 信号确认机制（提高信号质量）
        self.signal_confirmation_count = 0  # 连续信号计数
        self.signal_confirmation_threshold = 2  # 需要连续2个信号才开仓
        self.last_signal = None  # 上一个信号
        
        # 设置交易所
        d.DATA_CONFIG['exchange'] = 'okx'
        d._exchange = None
        
        # 获取合约信息
        self._init_contract_info()
        
        # 记录初始余额
        try:
            self.initial_balance = self.trader.get_balance('USDT')
        except:
            pass
        
        LOGGER.info(f"[HF-LIVE] 初始化完成: symbol={symbol}, max_capital=${max_capital_usd}, demo={demo_mode}")
    
    def _get_spot_symbol(self) -> str:
        """将合约格式转换为现货格式（用于订单簿查询）"""
        # BTC-USDT-SWAP -> BTC/USDT
        return self.symbol.replace('-USDT-SWAP', '/USDT').replace('-', '/')
    
    def _init_contract_info(self):
        """初始化合约信息（面值、最小开仓等）"""
        try:
            # 使用 trader 的 exchange 实例（已配置为合约类型）
            ex = self.trader.exchange
            if ex is None:
                LOGGER.error("[HF-LIVE] 无法获取交易所实例")
                return
            
            # 先加载市场数据
            LOGGER.info("[HF-LIVE] 正在加载市场数据...")
            ex.load_markets()
            
            # 获取市场信息
            market = ex.market(self.symbol)
            info = market.get('info', {}) if isinstance(market, dict) else {}
            
            # 合约面值（OKX BTC永续合约固定为100 USD）
            # 强制使用标准值，避免从API获取错误的值
            symbol_upper = self.symbol.upper()
            if 'BTC' in symbol_upper and 'SWAP' in symbol_upper:
                self.contract_value_usd = 100.0  # BTC永续合约标准面值
            elif 'ETH' in symbol_upper and 'SWAP' in symbol_upper:
                self.contract_value_usd = 10.0   # ETH永续合约标准面值
            else:
                # 其他合约尝试从API获取
                self.contract_value_usd = 100.0  # 默认值
                ct_val = info.get('ctVal')
                ct_val_ccy = info.get('ctValCcy', 'USD')
                if ct_val:
                    try:
                        ct_val = float(ct_val)
                        if ct_val_ccy == 'USD':
                            self.contract_value_usd = ct_val
                        elif ct_val_ccy and ct_val_ccy.upper() in ('BTC', 'ETH'):
                            # 面值以币计价，换算为USD
                            ticker = ex.fetch_ticker(self.symbol)
                            current_price = float(ticker['last'])
                            self.contract_value_usd = ct_val * current_price
                    except:
                        pass
            
            # 最小开仓张数
            self.min_contracts = int(market.get('limits', {}).get('amount', {}).get('min', 1) or 1)
            if self.min_contracts < 1:
                self.min_contracts = 1
            
            # 确保合约面值有效
            if self.contract_value_usd <= 0:
                LOGGER.warning(f"[HF-LIVE] 合约面值无效 ({self.contract_value_usd})，使用默认值 100 USD")
                self.contract_value_usd = 100.0
            
            # 计算最大杠杆（基于最小开仓价格）
            # 最小开仓价格 = 最小张数 * 合约面值
            min_open_value = self.min_contracts * self.contract_value_usd
            
            # 防止除零
            if min_open_value <= 0:
                LOGGER.warning(f"[HF-LIVE] 最小开仓价格无效 ({min_open_value})，使用默认杠杆 5x")
                self.optimal_leverage = 5
            else:
                # 获取实际可用保证金
                try:
                    available_margin = self.trader.get_balance('USDT')
                except:
                    available_margin = self.max_capital_usd
                
                # 使用最小开仓价格，计算刚好能开1张的杠杆
                # 目标：使用最小资金，开最小仓位（1张）
                # 杠杆 = 最小开仓价格 / 可用资金（但至少5x，最多不超过合理范围）
                usable_margin = min(available_margin, self.max_capital_usd)
                
                # 计算刚好能开1张的杠杆（最小杠杆）
                min_leverage_for_1_contract = min_open_value / usable_margin if usable_margin > 0 else 5
                
                # OKX要求最小杠杆为5x
                if min_leverage_for_1_contract < 5:
                    # 如果计算出的杠杆小于5x，使用5x（但需要确保资金足够）
                    if usable_margin >= min_open_value * 5:
                        self.optimal_leverage = 5
                    else:
                        # 资金不足，使用最小可用杠杆
                        self.optimal_leverage = max(5, int(min_leverage_for_1_contract) + 1)
                        LOGGER.warning(
                            f"[HF-LIVE] 资金不足，使用最小杠杆 {self.optimal_leverage}x"
                        )
                else:
                    # 使用刚好能开1张的杠杆（向上取整，但不超过20x以降低风险）
                    self.optimal_leverage = min(int(min_leverage_for_1_contract) + 1, 20)
            
            # 获取可用保证金信息（在计算杠杆之前）
            try:
                available_margin = self.trader.get_balance('USDT')
                # 获取总资产
                balance = ex.fetch_balance()
                total_equity = 0.0
                for currency, asset in balance.items():
                    if isinstance(asset, dict) and currency.upper() == 'USDT':
                        total_equity = float(asset.get('total', 0) or asset.get('eq', 0) or 0)
                        break
                
                # 重新计算杠杆（基于实际可用保证金）
                if min_open_value > 0 and available_margin > 0:
                    # 使用实际可用保证金和最大使用资金的较小值
                    usable_margin = min(available_margin, self.max_capital_usd)
                    max_leverage_by_margin = int(usable_margin / min_open_value)
                    # OKX要求最小杠杆为5x（2-4x经常导致保证金不足错误）
                    if max_leverage_by_margin < 5:
                        if usable_margin < min_open_value * 5:
                            LOGGER.warning(
                                f"[HF-LIVE] 资金不足：需要至少 ${min_open_value * 5:.2f} "
                                f"才能以5x杠杆开仓，当前可用=${usable_margin:.2f}"
                            )
                            self.optimal_leverage = 5  # 尝试5x，但可能失败
                        else:
                            self.optimal_leverage = 5  # 使用最小5x
                    else:
                        # 限制在125x以内
                        self.optimal_leverage = min(max_leverage_by_margin, 125)
                
                LOGGER.info("="*60)
                LOGGER.info(f"[HF-LIVE] 账户资金信息:")
                LOGGER.info(f"  可用保证金 (availEq): ${available_margin:.2f}")
                LOGGER.info(f"  账户总资产 (total): ${total_equity:.2f}")
                LOGGER.info(f"  最大使用资金: ${self.max_capital_usd:.2f}")
                LOGGER.info("="*60)
                
                # 计算可开张数
                if available_margin > 0:
                    max_contracts = int((min(available_margin, self.max_capital_usd) * self.optimal_leverage) / self.contract_value_usd)
                    LOGGER.info(f"[HF-LIVE] 基于可用保证金，最多可开: {max_contracts}张")
                else:
                    LOGGER.warning(f"[HF-LIVE] 可用保证金为0，无法开仓")
            except Exception as e:
                LOGGER.warning(f"[HF-LIVE] 获取保证金信息失败: {e}")
                # 如果获取失败，使用默认5x
                if self.optimal_leverage < 5:
                    self.optimal_leverage = 5
            
            LOGGER.info(
                f"[HF-LIVE] 合约信息: 面值=${self.contract_value_usd}, "
                f"最小张数={self.min_contracts}, 最优杠杆={self.optimal_leverage}x"
            )
            
            # 设置杠杆（重试机制已在set_leverage中实现）
            try:
                self.trader.set_leverage(self.symbol, self.optimal_leverage)
            except Exception as e:
                LOGGER.error(f"[HF-LIVE] 设置杠杆失败: {e}")
                # 如果设置失败，尝试使用更高的杠杆（10x）
                if self.optimal_leverage < 10:
                    LOGGER.warning(f"[HF-LIVE] 尝试使用10x杠杆...")
                    try:
                        self.trader.set_leverage(self.symbol, 10)
                        self.optimal_leverage = 10
                        LOGGER.info(f"[HF-LIVE] 成功设置杠杆为10x")
                    except:
                        LOGGER.error(f"[HF-LIVE] 设置10x杠杆也失败，可能资金不足")
            
        except Exception as e:
            LOGGER.error(f"[HF-LIVE] 初始化合约信息失败: {e}")
            self.contract_value_usd = 100.0
            self.min_contracts = 1
            self.optimal_leverage = 1
    
    def _calculate_position_size(self) -> int:
        """计算开仓张数（使用最小开仓价格，降低资金使用）"""
        try:
            # 获取可用余额
            balance = self.trader.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            # 限制使用最大资金
            available_usd = min(float(usdt_balance), self.max_capital_usd)
            
            # 最小开仓价格
            min_open_value = self.min_contracts * self.contract_value_usd
            
            # 计算所需保证金：最小开仓价格 / 杠杆
            required_margin = min_open_value / self.optimal_leverage
            
            if available_usd < required_margin:
                LOGGER.warning(
                    f"[HF-LIVE] 可用资金不足: ${available_usd:.2f}, "
                    f"需要=${required_margin:.2f} (最小开仓=${min_open_value:.2f}, 杠杆={self.optimal_leverage}x)"
                )
                return 0
            
            # 只开最小仓位（1张），使用最小资金
            contracts = self.min_contracts
            
            LOGGER.info(
                f"[HF-LIVE] 仓位计算: 开仓={contracts}张, "
                f"所需保证金=${required_margin:.2f}, "
                f"杠杆={self.optimal_leverage}x, "
                f"可用=${available_usd:.2f}"
            )
            
            return contracts
            
        except Exception as e:
            LOGGER.error(f"[HF-LIVE] 计算仓位大小失败: {e}")
            return self.min_contracts
    
    def _generate_signal_from_obi(self, obi: Dict) -> Optional[str]:
        """从OBI结果生成交易信号：BUY, SELL, HOLD"""
        try:
            # 先添加到历史记录（不管是否通过过滤，都需要数据来计算特征）
            self.orderbook_history.append({
                'timestamp': time.time() * 1000,
                'ratio': obi['ratio'],
                'buy_depth': obi['buy_depth'],
                'sell_depth': obi['sell_depth'],
                'best_bid': obi['best_bid'],
                'best_ask': obi['best_ask'],
                'mid': obi['mid'],
                'spread_bps': obi.get('spread_bps'),
            })
            
            # 只保留最近N条
            if len(self.orderbook_history) > self.max_history:
                self.orderbook_history.pop(0)
            
            # 每30秒记录一次数据积累状态（无论是否达到阈值）
            if not hasattr(self, '_last_data_status_log') or (time.time() - self._last_data_status_log) > 30:
                status = "就绪" if len(self.orderbook_history) >= self.ofi_span else "积累中"
                LOGGER.info(
                    f"[HF-LIVE] 数据状态: {status}, "
                    f"当前={len(self.orderbook_history)}条, 需要={self.ofi_span}条"
                )
                self._last_data_status_log = time.time()
            
            # 深度和点差过滤（在数据积累后检查）
            total_depth = obi['buy_depth'] + obi['sell_depth']
            spread_bps = obi.get('spread_bps')
            
            if total_depth < self.min_depth_total:
                # 每60秒记录一次过滤原因（避免日志过多）
                if not hasattr(self, '_last_filter_log') or (time.time() - self._last_filter_log) > 60:
                    LOGGER.debug(f"[HF-LIVE] 深度不足: {total_depth:.2f} < {self.min_depth_total}")
                    self._last_filter_log = time.time()
                # 继续处理，但会在信号生成时返回None
            elif spread_bps and spread_bps > self.max_spread_bps:
                if not hasattr(self, '_last_filter_log') or (time.time() - self._last_filter_log) > 60:
                    LOGGER.debug(f"[HF-LIVE] 点差过大: {spread_bps:.2f}bp > {self.max_spread_bps}bp")
                    self._last_filter_log = time.time()
                # 继续处理，但会在信号生成时返回None
            
            # 需要至少ofi_span条数据才能计算OFI
            if len(self.orderbook_history) < self.ofi_span:
                # 每5秒记录一次数据积累进度（更频繁，便于观察）
                if not hasattr(self, '_last_data_log') or (time.time() - self._last_data_log) > 5:
                    LOGGER.info(f"[HF-LIVE] 数据积累中: {len(self.orderbook_history)}/{self.ofi_span}条")
                    self._last_data_log = time.time()
                return None
            
            # 转换为DataFrame并计算特征
            df = pd.DataFrame(self.orderbook_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df = _enrich_micro_ofi_features(df, ofi_span=self.ofi_span)
            
            if df.empty or 'micro_bias' not in df.columns:
                return None
            
            # 再次检查深度和点差过滤（在计算特征后）
            if total_depth < self.min_depth_total:
                return None
            
            if spread_bps and spread_bps > self.max_spread_bps:
                return None
            
            # 获取最新一行
            latest = df.iloc[-1]
            micro_bias = float(latest['micro_bias'])
            ofi_ema = float(latest.get('ofi_ema', 0.0))
            
            # 每30秒记录一次信号状态（用于调试，更频繁）
            if not hasattr(self, '_last_signal_log') or (time.time() - self._last_signal_log) > 30:
                # 计算距离阈值的距离
                dist_to_long = micro_bias - self.bias_long if micro_bias < self.bias_long else 0
                dist_to_short = self.bias_short - micro_bias if micro_bias > self.bias_short else 0
                closest_threshold = min(abs(dist_to_long), abs(dist_to_short)) if (dist_to_long != 0 or dist_to_short != 0) else 0
                
                LOGGER.info(
                    f"[HF-LIVE] 信号状态: micro_bias={micro_bias:.2e} "
                    f"(阈值: {self.bias_short:.2e} ~ {self.bias_long:.2e}, "
                    f"距离阈值: {closest_threshold:.2e}), "
                    f"ofi_ema={ofi_ema:.2e}, 深度={total_depth:.2f}, 点差={spread_bps:.2f}bp"
                )
                self._last_signal_log = time.time()
            
            # 生成信号（带确认机制）
            current_signal = None
            if micro_bias > self.bias_long:
                if self.ofi_long == 0.0 or ofi_ema > self.ofi_long:
                    current_signal = 'BUY'
            elif micro_bias < self.bias_short:
                if self.ofi_short == 0.0 or ofi_ema < self.ofi_short:
                    current_signal = 'SELL'
            
            # 信号确认机制：需要连续N个相同信号才开仓
            if current_signal and current_signal == self.last_signal:
                self.signal_confirmation_count += 1
            else:
                # 信号变化，重置计数
                self.signal_confirmation_count = 1 if current_signal else 0
                self.last_signal = current_signal
            
            # 只有连续达到阈值才返回信号
            if current_signal and self.signal_confirmation_count >= self.signal_confirmation_threshold:
                LOGGER.info(
                    f"[HF-LIVE] 生成{current_signal}信号（已确认{self.signal_confirmation_count}次）: "
                    f"micro_bias={micro_bias:.2e}, 阈值={self.bias_long if current_signal == 'BUY' else self.bias_short:.2e}"
                )
                # 重置计数（避免重复开仓）
                self.signal_confirmation_count = 0
                return current_signal
            elif current_signal:
                # 信号未确认，记录但不返回
                if not hasattr(self, '_last_confirmation_log') or (time.time() - self._last_confirmation_log) > 30:
                    LOGGER.debug(
                        f"[HF-LIVE] 信号确认中: {current_signal} ({self.signal_confirmation_count}/{self.signal_confirmation_threshold})"
                    )
                    self._last_confirmation_log = time.time()
            
            return 'HOLD'
            
        except Exception as e:
            LOGGER.error(f"[HF-LIVE] 生成信号失败: {e}")
            return None
    
    def _check_exit_conditions(self, current_price: float) -> bool:
        """检查是否满足平仓条件（止盈止损、时间止损）"""
        if not self.position or self.entry_price == 0:
            return False
        
        # 计算价格变动
        if self.position == 'LONG':
            move_pct = (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            move_pct = (self.entry_price - current_price) / self.entry_price
        
        # 止盈止损
        if move_pct >= self.tp_pct:
            LOGGER.info(f"[HF-LIVE] 触发止盈: {self.position}, 收益={move_pct*100:.4f}%")
            return True
        
        if move_pct <= -self.sl_pct:
            LOGGER.warning(f"[HF-LIVE] 触发止损: {self.position}, 亏损={move_pct*100:.4f}%")
            return True
        
        # 时间止损
        if self.entry_time:
            hold_time = (datetime.now() - self.entry_time).total_seconds()
            if hold_time >= self.time_stop_sec:
                LOGGER.info(f"[HF-LIVE] 触发时间止损: 持仓{hold_time:.1f}秒")
                return True
        
        return False
    
    def _open_position(self, signal: str, contracts: int):
        """开仓"""
        try:
            decision = 'LONG' if signal == 'BUY' else 'SHORT'
            
            decision_data = {
                'symbol': self.symbol,
                'decision': decision,
                'suggested_trade_size': contracts,  # 直接传张数
                'trade_params': {
                    'leverage': self.optimal_leverage,
                }
            }
            
            LOGGER.info(f"[HF-LIVE] 开仓: {decision}, {contracts}张, 杠杆={self.optimal_leverage}x")
            
            self.trader.execute_decision(decision_data)
            
            # 等待订单成交后，从API获取实际成交价
            import time
            time.sleep(0.5)  # 等待订单成交
            
            # 从API获取实际成交价
            position_info = self.trader.get_position(self.symbol)
            if position_info:
                avg_px = position_info.get('avgPx')
                if avg_px:
                    self.entry_price = float(avg_px)
                    self.entry_time = datetime.now()
                    self.position = decision
                    self.last_trade_time = datetime.now()
                    LOGGER.info(f"[HF-LIVE] 开仓成功: {decision}, 实际入场价={self.entry_price:.2f}")
                else:
                    # 备选：使用订单簿价格
                    ob = fetch_orderbook(self._get_spot_symbol(), depth=5)
                    if ob:
                        obi = compute_obi(ob, levels=3)
                        if obi:
                            self.entry_price = obi['mid']
                            self.entry_time = datetime.now()
                            self.position = decision
                            self.last_trade_time = datetime.now()
                            LOGGER.warning(f"[HF-LIVE] 开仓成功（使用订单簿价格）: {decision}, 入场价={self.entry_price:.2f}")
            else:
                # 备选：使用订单簿价格
                ob = fetch_orderbook(self._get_spot_symbol(), depth=5)
                if ob:
                    obi = compute_obi(ob, levels=3)
                    if obi:
                        self.entry_price = obi['mid']
                        self.entry_time = datetime.now()
                        self.position = decision
                        self.last_trade_time = datetime.now()
                        LOGGER.warning(f"[HF-LIVE] 开仓成功（使用订单簿价格）: {decision}, 入场价={self.entry_price:.2f}")
            
        except Exception as e:
            LOGGER.error(f"[HF-LIVE] 开仓失败: {e}")
    
    def _close_position(self):
        """平仓"""
        try:
            if not self.position:
                return
            
            # 获取当前仓位信息
            position_info = self.trader.get_position(self.symbol)
            if not position_info:
                LOGGER.warning("[HF-LIVE] 未找到仓位信息，可能已平仓")
                self.position = None
                return
            
            # 保存当前状态（在重置前）
            old_position = self.position
            old_entry_price = self.entry_price
            old_entry_time = self.entry_time
            
            # 执行平仓（先平仓，再获取实际成交价）
            decision = 'CLOSE_LONG' if old_position == 'LONG' else 'CLOSE_SHORT'
            decision_data = {
                'symbol': self.symbol,
                'decision': decision,
            }
            
            # 记录平仓前的余额（用于计算总盈亏）
            balance_before = self.trader.get_balance('USDT') if not self.demo_mode else 0.0
            
            self.trader.execute_decision(decision_data)
            
            # 等待订单成交（平仓是市价单，通常很快）
            import time
            time.sleep(1.0)
            
            # 平仓后，从更新的仓位信息或订单信息获取实际成交价和盈亏
            exit_price = 0.0
            realized_pnl_usd = 0.0
            
            # 方法1：从余额变化计算总盈亏
            balance_after = self.trader.get_balance('USDT') if not self.demo_mode else 0.0
            if balance_before > 0 and balance_after > 0:
                realized_pnl_usd = balance_after - balance_before
            
            # 方法2：尝试从订单信息获取成交价
            try:
                # 获取最近的订单（平仓订单）
                orders = self.trader.exchange.fetch_orders(self.symbol, limit=1)
                if orders and len(orders) > 0:
                    latest_order = orders[0]
                    # 获取订单成交价
                    if 'price' in latest_order and latest_order['price']:
                        exit_price = float(latest_order['price'])
                    elif 'average' in latest_order and latest_order['average']:
                        exit_price = float(latest_order['average'])
                    elif 'info' in latest_order:
                        # OKX返回的订单信息
                        order_info = latest_order['info']
                        if 'avgPx' in order_info:
                            exit_price = float(order_info['avgPx'])
            except Exception as e:
                LOGGER.warning(f"[HF-LIVE] 无法从订单获取成交价: {e}")
            
            # 方法3：如果无法从订单获取，使用当前市价作为备选
            if exit_price == 0.0:
                ob = fetch_orderbook(self._get_spot_symbol(), depth=5)
                if ob:
                    obi = compute_obi(ob, levels=3)
                    if obi:
                        exit_price = obi['mid']
                        LOGGER.warning(f"[HF-LIVE] 使用订单簿价格作为出场价: {exit_price:.2f}")
            
            # 计算盈亏百分比（基于实际成交价）
            pnl_pct = 0.0
            hold_seconds = 0.0
            if exit_price > 0 and old_entry_price > 0:
                if old_position == 'LONG':
                    pnl_pct = (exit_price - old_entry_price) / old_entry_price
                else:
                    pnl_pct = (old_entry_price - exit_price) / old_entry_price
                
                if old_entry_time:
                    hold_seconds = (datetime.now() - old_entry_time).total_seconds()
            
            # 记录实际盈亏
            LOGGER.info(
                f"[HF-LIVE] 实际盈亏: ${realized_pnl_usd:.4f} "
                f"(入场={old_entry_price:.2f}, 出场={exit_price:.2f}, "
                f"理论收益={pnl_pct*100:.4f}%, 余额变化=${balance_after - balance_before:.4f})"
            )
            
            # 记录交易
            trade_record = {
                'trade_id': len(self.trade_records) + 1,
                'timestamp': datetime.now().isoformat(),
                'position': old_position,
                'entry_price': old_entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct * 100,  # 转换为百分比
                'realized_pnl_usd': realized_pnl_usd,  # 实际盈亏（美元）
                'hold_seconds': hold_seconds,
                'leverage': self.optimal_leverage,
            }
            self.trade_records.append(trade_record)
            
            # 保存到文件
            self._save_pnl_log()
            
            # 输出盈亏统计
            self._print_pnl_stats()
            
            # 重置状态
            self.position = None
            self.entry_price = 0.0
            self.entry_time = None
            self.cooldown_until = datetime.now() + timedelta(seconds=self.cooldown_sec)
            self.last_trade_time = datetime.now()
            # 重置信号确认计数（平仓后重新开始确认）
            self.signal_confirmation_count = 0
            self.last_signal = None
            
            LOGGER.info(
                f"[HF-LIVE] 平仓: {old_position}, "
                f"入场={old_entry_price:.2f}, 出场={exit_price:.2f}, "
                f"收益={pnl_pct*100:.4f}%, 持仓{hold_seconds:.1f}秒"
            )
            LOGGER.info(f"[HF-LIVE] 平仓成功，进入冷却期{self.cooldown_sec}秒")
            
        except Exception as e:
            LOGGER.error(f"[HF-LIVE] 平仓失败: {e}")
    
    def run_once(self):
        """执行一次交易循环"""
        try:
            # 检查冷却期
            if self.cooldown_until and datetime.now() < self.cooldown_until:
                return
            
            # 获取订单簿
            ob = fetch_orderbook(self._get_spot_symbol(), depth=5)
            if not ob:
                # 检查是否是因为限流失败（通过检查日志或异常信息）
                # 由于 fetch_orderbook 内部已经记录了限流日志，这里只记录统计
                self.rate_limit_count += 1
                self.rate_limit_last_time = datetime.now()
                
                # 每30秒记录一次获取失败（包括限流统计）
                if not hasattr(self, '_last_ob_fail_log') or (time.time() - self._last_ob_fail_log) > 30:
                    LOGGER.warning(
                        f"[HF-LIVE] 订单簿获取失败，可能API限流 - "
                        f"累计限流次数: {self.rate_limit_count}, "
                        f"最后限流时间: {self.rate_limit_last_time.strftime('%H:%M:%S') if self.rate_limit_last_time else 'N/A'}"
                    )
                    self._last_ob_fail_log = time.time()
                return
            
            obi = compute_obi(ob, levels=3)
            if not obi:
                # 每30秒记录一次OBI计算失败
                if not hasattr(self, '_last_obi_fail_log') or (time.time() - self._last_obi_fail_log) > 30:
                    LOGGER.warning(f"[HF-LIVE] OBI计算失败")
                    self._last_obi_fail_log = time.time()
                return
            
            current_price = obi['mid']
            
            # 无论是否有仓位，都要继续积累数据（平仓后需要数据来生成新信号）
            # 生成信号（直接传入OBI结果，避免重复计算）
            signal = self._generate_signal_from_obi(obi)
            
            # 如果有仓位，检查平仓条件
            if self.position:
                if self._check_exit_conditions(current_price):
                    self._close_position()
                return
            if not signal or signal == 'HOLD':
                return
            
            # 计算仓位大小
            contracts = self._calculate_position_size()
            if contracts < self.min_contracts:
                LOGGER.warning(f"[HF-LIVE] 仓位不足，无法开仓")
                return
            
            # 开仓
            self._open_position(signal, contracts)
            
        except Exception as e:
            LOGGER.error(f"[HF-LIVE] 交易循环失败: {e}", exc_info=True)
    
    def _save_pnl_log(self):
        """保存盈亏记录到文件"""
        try:
            
            # 计算统计信息
            stats = self._calculate_stats()
            
            # 保存数据
            data = {
                'initial_balance': self.initial_balance,
                'current_balance': self.trader.get_balance('USDT') if not self.demo_mode else 0.0,
                'total_trades': len(self.trade_records),
                'statistics': stats,
                'trades': self.trade_records,
                'last_updated': datetime.now().isoformat(),
            }
            
            with open(self.pnl_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            LOGGER.error(f"[HF-LIVE] 保存盈亏记录失败: {e}")
    
    def _calculate_stats(self) -> Dict[str, Any]:
        """计算统计信息"""
        if not self.trade_records:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl_pct': 0.0,
                'avg_pnl_pct': 0.0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_hold_seconds': 0.0,
            }
        
        winning_trades = sum(1 for t in self.trade_records if t.get('realized_pnl_usd', t.get('pnl_pct', 0)) > 0)
        losing_trades = sum(1 for t in self.trade_records if t.get('realized_pnl_usd', t.get('pnl_pct', 0)) < 0)
        # 优先使用实际盈亏（美元），如果没有则使用百分比
        total_pnl_usd = sum(t.get('realized_pnl_usd', 0) for t in self.trade_records)
        total_pnl = sum(t['pnl_pct'] for t in self.trade_records)
        avg_hold = sum(t['hold_seconds'] for t in self.trade_records) / len(self.trade_records)
        
        return {
            'total_trades': len(self.trade_records),
            'win_rate': winning_trades / len(self.trade_records) * 100 if self.trade_records else 0.0,
            'total_pnl_pct': total_pnl,
            'total_pnl_usd': total_pnl_usd,  # 实际盈亏（美元）
            'avg_pnl_pct': total_pnl / len(self.trade_records) if self.trade_records else 0.0,
            'avg_pnl_usd': total_pnl_usd / len(self.trade_records) if self.trade_records else 0.0,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_hold_seconds': avg_hold,
        }
    
    def _print_pnl_stats(self):
        """输出盈亏统计"""
        stats = self._calculate_stats()
        
        if stats['total_trades'] == 0:
            return
        
        LOGGER.info("="*60)
        LOGGER.info("[HF-LIVE] 盈亏统计:")
        LOGGER.info(f"  总交易数: {stats['total_trades']}")
        LOGGER.info(f"  胜率: {stats['win_rate']:.2f}% ({stats['winning_trades']}胜/{stats['losing_trades']}负)")
        LOGGER.info(f"  总收益率: {stats['total_pnl_pct']:.4f}%")
        if 'total_pnl_usd' in stats:
            LOGGER.info(f"  实际盈亏: ${stats['total_pnl_usd']:.4f} (平均每笔: ${stats.get('avg_pnl_usd', 0):.4f})")
        LOGGER.info(f"  平均每笔: {stats['avg_pnl_pct']:.4f}%")
        LOGGER.info(f"  平均持仓: {stats['avg_hold_seconds']:.1f}秒")
        
        # 计算当前余额变化
        try:
            current_balance = self.trader.get_balance('USDT') if not self.demo_mode else self.initial_balance
            if self.initial_balance > 0:
                balance_change = current_balance - self.initial_balance
                balance_change_pct = (balance_change / self.initial_balance) * 100
                LOGGER.info(f"  初始余额: ${self.initial_balance:.2f}")
                LOGGER.info(f"  当前余额: ${current_balance:.2f}")
                LOGGER.info(f"  余额变化: ${balance_change:.2f} ({balance_change_pct:.4f}%)")
        except:
            pass
        
        LOGGER.info("="*60)
    
    def run_continuous(self, interval_sec: float = 1.0, duration_min: int = 60):
        """持续运行交易循环"""
        LOGGER.info(
            f"[HF-LIVE] 开始持续交易: 间隔={interval_sec}秒, 时长={duration_min}分钟"
        )
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_min)
        iteration = 0
        
        try:
            while datetime.now() < end_time:
                iteration += 1
                self.run_once()
                
                # 每100次循环输出一次状态（包括限流统计）
                if iteration % 100 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    rate_limit_info = ""
                    if self.rate_limit_count > 0:
                        last_time_str = self.rate_limit_last_time.strftime('%H:%M:%S') if self.rate_limit_last_time else 'N/A'
                        rate_limit_info = f", 限流次数={self.rate_limit_count} (最后: {last_time_str})"
                    LOGGER.info(
                        f"[HF-LIVE] 运行中: {elapsed:.1f}分钟, "
                        f"当前仓位={self.position}, 入场价={self.entry_price:.2f}{rate_limit_info}"
                    )
                
                time.sleep(interval_sec)
                
        except KeyboardInterrupt:
            LOGGER.info("[HF-LIVE] 收到停止信号，正在平仓...")
            if self.position:
                self._close_position()
        except Exception as e:
            LOGGER.error(f"[HF-LIVE] 持续运行失败: {e}", exc_info=True)
            if self.position:
                self._close_position()
        finally:
            # 输出最终统计
            LOGGER.info("\n" + "="*60)
            LOGGER.info("[HF-LIVE] 交易结束，最终统计:")
            self._print_pnl_stats()
            # 输出限流统计
            if self.rate_limit_count > 0:
                last_time_str = self.rate_limit_last_time.strftime('%Y-%m-%d %H:%M:%S') if self.rate_limit_last_time else 'N/A'
                LOGGER.info(f"[HF-LIVE] API限流统计: 总次数={self.rate_limit_count}, 最后限流时间={last_time_str}")
            else:
                LOGGER.info("[HF-LIVE] API限流统计: 无限流")
            LOGGER.info(f"[HF-LIVE] 盈亏记录已保存到: {self.pnl_file}")
            LOGGER.info("="*60 + "\n")


if __name__ == "__main__":
    # 从环境变量读取配置
    demo_mode = config.DEMO_MODE
    
    trader = HFLiveTrader(
        symbol="BTC/USDT:USDT",
        max_capital_usd=500.0,
        bias_long=5e-7,
        bias_short=-5e-7,
        ofi_long=0.0,
        ofi_short=0.0,
        tp_pct=0.0010,
        sl_pct=0.0012,
        min_depth_total=2.0,
        max_spread_bps=2.0,
        cooldown_sec=10,
        time_stop_sec=30,
        ofi_span=8,
        use_taker=False,
        demo_mode=demo_mode,
    )
    
    # 运行1小时测试
    trader.run_continuous(interval_sec=1.0, duration_min=60)

