import ccxt
from typing import Optional, Dict, Any, Union
from decimal import Decimal

from tenacity import retry, stop_after_attempt, wait_exponential

import config
from btc_predictor.utils import LOGGER
from utils.email_notifier import EmailNotifier

class OKXTrader:
    """
    OKXTrader 类用于与 OKX 交易所进行期货交互。
    """
    def __init__(self, demo_mode: bool = True):
        """初始化OKXTrader。"""
        self.demo_mode = demo_mode
        
        # 从配置加载期货设置
        futures_config = config.FUTURES
        self.trade_symbol = futures_config['trade_symbol']
        self.leverage = futures_config['leverage']
        self.margin_mode = futures_config['margin_mode']
        self.hedge_mode = futures_config.get('hedge_mode', False)

        creds = config.API_KEYS.get('okx', {})
        self.exchange_config: Dict[str, Any] = {
            'apiKey': creds.get('api_key', ''),
            'secret': creds.get('secret_key', ''),
            'password': creds.get('passphrase', ''),
            'options': {
                'defaultType': 'swap',
            },
        }
        
        self.exchange: ccxt.okx = ccxt.okx(self.exchange_config) # type: ignore
        
        proxy_url = config.DEFAULTS.get('proxy_url')
        if proxy_url:
            self.exchange.proxies = {'http': proxy_url, 'https': proxy_url} # type: ignore

        if self.demo_mode:
            self.exchange.set_sandbox_mode(True)
            LOGGER.info(f"OKX Trader 已初始化 (模拟盘模式) - 交易对: {self.trade_symbol}")
        else:
            LOGGER.info(f"OKX Trader 已初始化 (实盘模式) - 交易对: {self.trade_symbol}")

        # 初始化邮件通知器
        self.email_notifier = EmailNotifier()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_position(self) -> Optional[Dict[str, Any]]:
        """
        获取指定交易对的仓位信息。
        如果无仓位，返回 None。
        """
        if self.demo_mode:
            LOGGER.info("模拟模式：不获取真实仓位，返回 None。")
            return None
        try:
            # ccxt V4 接受符号列表
            positions = self.exchange.fetch_positions([self.trade_symbol])
            
            # 过滤掉数量为0的仓位并返回第一个
            for p in positions:
                # OKX即使仓位已关闭也会返回仓位信息，所以我们需要检查合约/数量
                if float(p.get('contracts', 0) or p.get('info', {}).get('pos', 0)) != 0:
                    LOGGER.info(f"获取到仓位信息: {p['info']}")
                    return p['info'] # 返回原始info字典，包含更多信息
            return None
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            LOGGER.error("获取仓位失败: {}", e)
            raise
        return None

    def get_positions(self, instId: Optional[str] = None):
        """
        Dashboard compatibility: returns a list with the current position (or empty list).
        """
        pos = self.get_position()
        return [pos] if pos else []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def set_leverage(self):
        """为交易对设置杠杆。"""
        if self.demo_mode:
            LOGGER.info(f"模拟模式：假装为 {self.trade_symbol} 设置杠杆为 {self.leverage}x。")
            return

        try:
            self.exchange.set_leverage(self.leverage, self.trade_symbol)
            LOGGER.success(f"已为 {self.trade_symbol} 设置杠杆为 {self.leverage}x")
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            LOGGER.error("设置杠杆失败: {}", e)
            raise

    def execute_decision(self, decision_data: Dict[str, Any]):
        """
        根据LLM决策字典智能执行期货交易。
        """
        decision = decision_data.get('decision', 'HOLD').upper()
        params = decision_data.get('trade_params', {})
        suggested_trade_size = decision_data.get('suggested_trade_size', 0.95)  # 默认使用95%资金
        
        if self.demo_mode:
            LOGGER.warning("当前为模拟盘模式，所有交易操作将被记录但不会真实执行。")
            LOGGER.info(f"模拟执行决策: {decision}，参数: {params}")
            # 在模拟模式下，我们直接返回，不执行任何操作
            return

        try:
            position_info = self.get_position()
            
            # --- 决策执行逻辑 ---
            if decision == 'HOLD':
                LOGGER.success("决策为 HOLD，无需执行任何交易。")
                return

            # 如果需要开仓 (LONG or SHORT)
            if decision in ['LONG', 'SHORT']:
                if position_info:
                    LOGGER.warning(f"决策为 {decision}，但已存在仓位，请先平仓。操作已取消。")
                    return
                
                # 获取当前市场价格
                ticker = self.exchange.fetch_ticker(self.trade_symbol)
                current_price = ticker['last']
                
                # 设置杠杆（使用决策中的杠杆参数）
                leverage = params.get('leverage', self.leverage)
                self.exchange.set_leverage(leverage, self.trade_symbol)
                
                # 计算下单数量（使用建议的仓位大小）
                balance = self.get_balance('USDT')
                trade_value = balance * suggested_trade_size
                
                # 计算下单张数（OKX永续合约1张=100美元名义价值）
                contract_value = 100  # BTC-USDT-SWAP
                amount = max(1, round(trade_value / contract_value))  # 至少1张
                actual_value = amount * contract_value
                LOGGER.info(f"实际下单张数: {amount} 张, 名义价值: ${actual_value}")
                
                # 检查是否满足OKX的最小数量要求
                min_amount = 0.01  # OKX要求的最小数量
                if amount < min_amount:
                    # 如果计算的数量小于最小要求，检查是否有足够资金
                    if current_price is not None:
                        required_value = (min_amount * float(current_price)) / leverage
                        if required_value <= balance:
                            # 有足够资金，使用最小数量
                            amount = min_amount
                            trade_value = required_value
                            LOGGER.warning(f"计算数量 {amount:.6f} BTC 小于最小要求，调整为最小数量 {min_amount} BTC")
                        else:
                            # 资金不足，无法开仓
                            LOGGER.error("资金不足，无法满足最小数量要求。需要 ${:.2f}，当前余额 ${:.2f}", required_value, balance)
                            return
                    else:
                        LOGGER.error("无法获取当前价格，无法计算所需资金")
                        return
                else:
                    # 确保数量在合理范围内
                    amount = min(amount, 1.0)  # 最大1.0 BTC
                
                side = 'buy' if decision == 'LONG' else 'sell'
                pos_side = 'long' if decision == 'LONG' else 'short'
                
                LOGGER.info(f"准备开新仓: {decision} {amount}张 {self.trade_symbol} (杠杆: {leverage}x, 价值: ${actual_value})...")
                
                # 创建市价单
                order_params = {'tdMode': self.margin_mode}
                if self.hedge_mode:
                    order_params['posSide'] = pos_side
                order = self.exchange.create_order(
                    symbol=self.trade_symbol,
                    type='market',
                    side=side,
                    amount=amount,
                    params=order_params
                )
                if isinstance(order, dict) and 'code' in order:
                    LOGGER.error("下单失败，返回错误: {}", order)
                    raise RuntimeError(f"下单失败: {order}")
                LOGGER.success(f"开仓 ({decision}) 订单已成功提交，订单ID: {order.get('id', 'N/A')}")
                
                # 设置止损和止盈订单（如果提供了百分比参数）
                if 'stop_loss_pct' in params or 'take_profit_pct' in params:
                    if current_price is not None:
                        # 使用下单时的amount参数，而不是order中的amount（市价单为None）
                        self._set_stop_orders(order, float(current_price), params, pos_side, amount)

            # 如果需要平仓 (CLOSE_LONG or CLOSE_SHORT)
            elif decision in ['CLOSE_LONG', 'CLOSE_SHORT']:
                if not position_info:
                    LOGGER.warning(f"决策为 {decision}，但当前无仓位。无需操作。")
                    return

                current_pos_side = position_info.get('posSide')
                decision_pos_side = 'long' if decision == 'CLOSE_LONG' else 'short'
                
                # 兼容OKX的net持仓模式
                if current_pos_side == 'net':
                    pos_amount_str = position_info.get('pos')
                    if pos_amount_str and float(pos_amount_str) > 0:
                        effective_side = 'long'
                    elif pos_amount_str and float(pos_amount_str) < 0:
                        effective_side = 'short'
                    else:
                        effective_side = 'none' # 无有效持仓
                    LOGGER.info(f"检测到 'net' 持仓模式, 根据持仓量判断实际方向为: {effective_side}")
                    current_pos_side = effective_side

                if current_pos_side != decision_pos_side:
                    LOGGER.error("决策平仓方向 ({}) 与实际持仓方向 ({}) 不符！操作取消。", decision_pos_side, current_pos_side)
                    return
                
                pos_amount_str = position_info.get('pos')
                contracts = position_info.get('contracts')
                # 判断持仓模式
                if current_pos_side == 'net':
                    # 单向持仓，使用notionalUsd字段计算张数
                    notional_usd = position_info.get('notionalUsd')
                    if notional_usd is None:
                        LOGGER.error("无法从仓位信息中获取名义价值 ('notionalUsd')。平仓操作取消。")
                        return
                    
                    notional_value = abs(float(notional_usd))  # 取绝对值
                    contract_value = 100  # OKX永续合约1张=100美元名义价值
                    amount = max(1, round(notional_value / contract_value))
                    
                    btc_amount = abs(float(pos_amount_str)) if pos_amount_str else 0
                    LOGGER.info(f"准备平仓: {decision} {amount}张 {self.trade_symbol} (名义价值${notional_value:.2f}, 约{btc_amount}BTC)...")
                else:
                    # 双向持仓，amount用张数
                    if contracts is not None and pos_amount_str is not None and float(contracts) > 0:
                        amount = int(float(contracts))
                        btc_amount = float(pos_amount_str)
                    elif pos_amount_str is not None:
                        btc_amount = float(pos_amount_str)
                        ticker = self.exchange.fetch_ticker(self.trade_symbol)
                        mark_price = ticker.get('last')
                        if mark_price is None:
                            LOGGER.error("无法获取当前价格，无法计算平仓张数。平仓操作取消。")
                            return
                        mark_price = float(mark_price)
                        contract_value = 100
                        amount = max(1, round(btc_amount * mark_price / contract_value))
                    else:
                        LOGGER.error("无法从仓位信息中获取持仓数量 ('pos'/'contracts')。平仓操作取消。")
                        return
                    LOGGER.info(f"准备平仓: {decision} {amount}张 {self.trade_symbol} (约{btc_amount}BTC)...")
                side = 'sell' if current_pos_side == 'long' else 'buy'
                order_params = {'tdMode': self.margin_mode, 'reduceOnly': True}
                if self.hedge_mode:
                    order_params['posSide'] = current_pos_side
                order = self.exchange.create_order(
                    symbol=self.trade_symbol,
                    type='market',
                    side=side,
                    amount=amount,
                    params=order_params
                )
                if isinstance(order, dict) and 'code' in order:
                    LOGGER.error("平仓下单失败，返回错误: {}", order)
                    raise RuntimeError(f"平仓下单失败: {order}")
                LOGGER.success(f"平仓 ({decision}) 订单已成功提交，订单ID: {order.get('id', 'N/A')}")

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            LOGGER.error("执行交易决策时发生交易所错误: {}", e, exc_info=True)
            self.email_notifier.send_error_notification("OKX交易所错误", str(e))
        except Exception as e:
            LOGGER.error("执行交易决策时发生未知错误: {}", e, exc_info=True)
            self.email_notifier.send_error_notification("OKX交易器未知错误", str(e))

    def _set_stop_orders(self, main_order: Union[Dict[str, Any], Any], entry_price: Union[float, Decimal], params: Dict[str, Any], pos_side: str, order_amount: float = None):
        """
        设置止损和止盈订单。
        
        Args:
            main_order: 主订单对象
            entry_price: 入场价格
            params: 交易参数
            pos_side: 持仓方向
            order_amount: 订单数量（用于市价单，因为市价单的amount字段为None）
        """
        try:
            # 确保entry_price是float类型
            LOGGER.info(f"[止盈止损] entry_price={entry_price}, params={params}, pos_side={pos_side}, order_amount={order_amount}")
            if entry_price is None:
                LOGGER.error("entry_price为None，无法设置止盈止损单。直接返回。")
                return
            
            entry_price = float(entry_price)
            stop_loss_pct = params.get('stop_loss_pct')
            take_profit_pct = params.get('take_profit_pct')
            
            # 优先使用传入的order_amount，如果没有则尝试从main_order获取
            if order_amount is not None:
                amount = float(order_amount)
            else:
                amount = main_order.get('amount') if isinstance(main_order, dict) else getattr(main_order, 'amount', None)
                if amount is None:
                    LOGGER.error("无法获取订单数量，无法设置止盈止损单。main_order={}, order_amount={}".format(main_order, order_amount))
                    return
                amount = float(amount)
            
            amount = round(amount, 4)
            stop_loss_price = None
            take_profit_price = None
            if stop_loss_pct:
                if pos_side == 'long':
                    stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
                else:
                    stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
                LOGGER.info(f"[止损] stop_loss_pct={stop_loss_pct}, stop_loss_price={stop_loss_price}")
                if stop_loss_price is None:
                    LOGGER.error("stop_loss_price为None，跳过止损单下单。")
                else:
                    stop_order_params = {
                        'tdMode': self.margin_mode,
                        'stopPrice': stop_loss_price
                    }
                    if self.hedge_mode:
                        stop_order_params['posSide'] = pos_side
                    LOGGER.info(f"提交止损单，数量: {amount}, 类型: {type(amount)}")
                    stop_order = self.exchange.create_order(
                        symbol=self.trade_symbol,
                        type='market',
                        side='sell' if pos_side == 'long' else 'buy',
                        amount=amount,
                        params=stop_order_params
                    )
                    if isinstance(stop_order, dict) and 'code' in stop_order:
                        LOGGER.error("止损单下单失败，返回错误: {}", stop_order)
                        raise RuntimeError(f"止损单下单失败: {stop_order}")
                    LOGGER.info(f"止损订单已设置: 价格 ${stop_loss_price:.2f}")
            if take_profit_pct:
                if pos_side == 'long':
                    take_profit_price = entry_price * (1 + take_profit_pct / 100)
                else:
                    take_profit_price = entry_price * (1 - take_profit_pct / 100)
                LOGGER.info(f"[止盈] take_profit_pct={take_profit_pct}, take_profit_price={take_profit_price}")
                if take_profit_price is None:
                    LOGGER.error("take_profit_price为None，跳过止盈单下单。")
                else:
                    take_profit_order_params = {
                        'tdMode': self.margin_mode
                    }
                    if self.hedge_mode:
                        take_profit_order_params['posSide'] = pos_side
                    LOGGER.info(f"提交止盈单，数量: {amount}, 类型: {type(amount)}")
                    take_profit_order = self.exchange.create_order(
                        symbol=self.trade_symbol,
                        type='limit',
                        side='sell' if pos_side == 'long' else 'buy',
                        amount=amount,
                        price=take_profit_price,
                        params=take_profit_order_params
                    )
                    if isinstance(take_profit_order, dict) and 'code' in take_profit_order:
                        LOGGER.error("止盈单下单失败，返回错误: {}", take_profit_order)
                        raise RuntimeError(f"止盈单下单失败: {take_profit_order}")
                    LOGGER.info(f"止盈订单已设置: 价格 ${take_profit_price:.2f}")
        except Exception as e:
            LOGGER.error("设置止损/止盈订单时发生错误: {}", e)
            self.email_notifier.send_error_notification("止盈止损设置错误", str(e))

    def get_balance(self, currency: str = 'USDT'):
        """
        Return available margin (equity) for the futures account. For dashboard compatibility.
        """
        if self.demo_mode:
            LOGGER.info(f"Demo mode: returning fixed margin balance for {currency}: 500.0")
            return 500.0
        try:
            balance = self.exchange.fetch_balance()
            info = balance.get('info', {})
            usdt_info = info.get('USDT') if isinstance(info, dict) else None
            if usdt_info and isinstance(usdt_info, dict):
                equity = float(usdt_info.get('eq', 0))
                LOGGER.info(f"Futures margin (equity) for {currency}: {equity}")
                return equity
            # 备用方案：尝试从total获取
            if 'total' in balance and currency in balance['total']:
                equity = float(balance['total'][currency])
                LOGGER.info(f"Futures margin (equity) for {currency}: {equity}")
                return equity
            LOGGER.warning(f"Could not find margin (equity) for {currency} in balance response.")
            return 0
        except Exception as e:
            LOGGER.error("Error fetching margin (equity) for {}: {}", currency, e)
            return 0 