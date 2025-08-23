import ccxt
import json
import os
from typing import Optional, Dict, Any, Union
from decimal import Decimal
from datetime import datetime, timezone

from tenacity import retry, stop_after_attempt, wait_exponential

import config
from btc_predictor.utils import LOGGER
from utils.email_notifier import EmailNotifier

ONGOING_TRADES_FILE = 'ongoing_trades.json'
TRADE_LOG_FILE = 'trade_log.json'


class OKXTrader:
    """
    OKXTrader 类用于与 OKX 交易所进行期货交互。
    """
    def __init__(self, demo_mode: bool = True):
        """初始化OKXTrader。"""
        self.demo_mode = demo_mode
        
        # 从配置加载期货设置
        futures_config = config.FUTURES
        self.trade_symbols = futures_config['trade_symbols'] # 接收币种列表
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
            LOGGER.info(f"OKX Trader 已初始化 (模拟盘模式) - 交易对: {', '.join(self.trade_symbols)}")
        else:
            LOGGER.info(f"OKX Trader 已初始化 (实盘模式) - 交易对: {', '.join(self.trade_symbols)}")

        # 初始化邮件通知器
        self.email_notifier = EmailNotifier()

        # 加载进行中的交易
        self._ongoing_trades = self._load_json_db(ONGOING_TRADES_FILE, {})

    def _load_json_db(self, filepath: str, default_data: Any) -> Any:
        """加载JSON文件，如果不存在则创建。"""
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(default_data, f, ensure_ascii=False, indent=4)
            return default_data
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            LOGGER.error(f"无法读取或解析 {filepath}，将使用默认数据。")
            return default_data

    def _save_json_db(self, filepath: str, data: Any):
        """保存数据到JSON文件。"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except IOError as e:
            LOGGER.error(f"无法写入到 {filepath}: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取指定交易对的仓位信息。
        如果无仓位，返回 None。
        """
        if self.demo_mode:
            LOGGER.info(f"[{symbol}] 模拟模式：不获取真实仓位，返回 None。")
            return None
        try:
            # ccxt V4 接受符号列表
            positions = self.exchange.fetch_positions([symbol])
            
            # 过滤掉数量为0的仓位并返回第一个
            for p in positions:
                # OKX即使仓位已关闭也会返回仓位信息，所以我们需要检查合约/数量
                if float(p.get('contracts', 0) or p.get('info', {}).get('pos', 0)) != 0:
                    LOGGER.info(f"[{symbol}] 获取到仓位信息: {p['info']}")
                    return p['info'] # 返回原始info字典，包含更多信息
            return None
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            LOGGER.error(f"[{symbol}] 获取仓位失败: {e}")
            raise
        return None

    def get_positions(self, instId: Optional[str] = None):
        """
        Dashboard compatibility: returns a list with the current position (or empty list).
        """
        # 注意：此方法当前只获取第一个交易对的仓位，可能需要为多币种仪表盘进行调整
        if self.trade_symbols:
            pos = self.get_position(self.trade_symbols[0])
            return [pos] if pos else []
        return []


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def set_leverage(self, symbol: str, leverage: Optional[int] = None):
        """为交易对设置杠杆。"""
        target_leverage = leverage if leverage is not None else self.leverage
        if self.demo_mode:
            LOGGER.info(f"模拟模式：假装为 {symbol} 设置杠杆为 {target_leverage}x。")
            return

        try:
            self.exchange.set_leverage(target_leverage, symbol)
            LOGGER.success(f"已为 {symbol} 设置杠杆为 {target_leverage}x")
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            LOGGER.error(f"为 {symbol} 设置杠杆失败: {e}")
            raise

    def execute_decision(self, decision_data: Dict[str, Any]):
        """
        根据LLM决策字典智能执行期货交易。
        """
        decision = decision_data.get('decision', 'HOLD').upper()
        params = decision_data.get('trade_params', {})
        suggested_trade_size = decision_data.get('suggested_trade_size', 0.95)
        
        # 从决策数据中获取交易对，如果没有则返回错误
        symbol = decision_data.get('symbol')
        if not symbol or symbol not in self.trade_symbols:
            LOGGER.error(f"决策中缺少有效交易对'symbol'或该币种未被配置: {symbol}")
            return

        if self.demo_mode:
            LOGGER.warning(f"[{symbol}] 当前为模拟盘模式，所有交易操作将被记录但不会真实执行。")
            LOGGER.info(f"[{symbol}] 模拟执行决策: {decision}，参数: {params}")
            return

        try:
            position_info = self.get_position(symbol)
            
            # --- 决策执行逻辑 ---
            if decision == 'HOLD':
                LOGGER.success(f"[{symbol}] 决策为 HOLD，无需执行任何交易。")
                return

            # 如果需要开仓 (LONG or SHORT)
            if decision in ['LONG', 'SHORT']:
                if position_info:
                    LOGGER.warning(f"[{symbol}] 决策为 {decision}，但已存在仓位，请先平仓。操作已取消。")
                    return
                
                # 获取当前市场价格
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # 设置杠杆（使用决策中的杠杆参数）
                leverage = params.get('leverage', self.leverage)
                self.set_leverage(symbol, leverage)
                
                # 获取最大可开张数（OKX API）
                max_amount = None
                try:
                    market = self.exchange.market(symbol)
                    # fetch_positions_risk 返回每个合约的最大可开张数（maxOpenPos）
                    risk = self.exchange.fetch_positions_risk([symbol])
                    for r in risk:
                        if r.get('symbol') == symbol and 'maxOpenPos' in r:
                            max_amount = int(float(r['maxOpenPos']))
                            break
                except Exception as e:
                    LOGGER.warning(f"[{symbol}] 获取最大可开张数失败: {e}")
                if max_amount is None:
                    # fallback: 手动计算最大可开张数
                    try:
                        avail_eq = self.get_balance('USDT')
                        if avail_eq <= 0:
                            LOGGER.error(f"[{symbol}] 可用保证金为0，无法下单。")
                            return
                        ticker = self.exchange.fetch_ticker(symbol)
                        current_price = float(ticker['last'])
                        # TODO: 不同币种的合约面值可能不同，需要配置化
                        contract_value = 100  # BTC-USDT-SWAP 一张=100美元名义价值
                        max_amount = int(avail_eq * leverage / (contract_value * current_price / 100))
                        if max_amount < 1:
                            max_amount = 1
                        LOGGER.info(f"[{symbol}] 手动计算最大可开张数: availEq={avail_eq}, 杠杆={leverage}, 现价={current_price}, 合约面值={contract_value}，最大可开张数={max_amount}")
                    except Exception as e:
                        LOGGER.error(f"[{symbol}] 手动计算最大可开张数失败: {e}")
                        return
                else:
                    LOGGER.info(f"[{symbol}] 最大可开张数: {max_amount}")
                amount = max_amount
                LOGGER.info(f"[{symbol}] 将用最大可开张数下单: {amount} 张, 杠杆: {leverage}x")
                
                side = 'buy' if decision == 'LONG' else 'sell'
                pos_side = 'long' if decision == 'LONG' else 'short'
                
                LOGGER.info(f"[{symbol}] 准备开新仓: {decision} 保证金 {amount} USDT (杠杆: {leverage}x)...")
                
                # 创建市价单
                order_params = {'tdMode': self.margin_mode}
                if self.hedge_mode:
                    order_params['posSide'] = pos_side
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=side,
                    amount=amount,
                    params=order_params
                )
                if isinstance(order, dict) and 'code' in order:
                    LOGGER.error(f"[{symbol}] 下单失败，返回错误: {order}")
                    raise RuntimeError(f"下单失败: {order}")
                LOGGER.success(f"[{symbol}] 开仓 ({decision}) 订单已成功提交，订单ID: {order.get('id', 'N/A')}")
                
                # --- [学习闭环] 记录开仓信息 ---
                try:
                    # 等待片刻以确保仓位信息更新
                    import time
                    time.sleep(5) 
                    updated_position = self.get_position(symbol)
                    if updated_position:
                        self._ongoing_trades[symbol] = {
                            "entry_report": decision_data,
                            "entry_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "entry_price": float(updated_position.get('avgPx', 0)),
                            "entry_contracts": float(updated_position.get('pos', 0))
                        }
                        self._save_json_db(ONGOING_TRADES_FILE, self._ongoing_trades)
                        LOGGER.info(f"[{symbol}] 已记录开仓信息到 {ONGOING_TRADES_FILE}")
                    else:
                        LOGGER.error(f"[{symbol}] 开仓后未能获取到仓位信息，无法记录开仓日志。")
                except Exception as e:
                    LOGGER.error(f"[{symbol}] 记录开仓信息时发生错误: {e}")

                # 设置止损和止盈订单（如果提供了百分比参数）
                if 'stop_loss_pct' in params or 'take_profit_pct' in params:
                    if current_price is not None:
                        self._set_stop_orders(symbol, order, float(current_price), params, pos_side, amount)

            # 如果需要平仓 (CLOSE_LONG or CLOSE_SHORT)
            elif decision in ['CLOSE_LONG', 'CLOSE_SHORT']:
                if not position_info:
                    LOGGER.warning(f"[{symbol}] 决策为 {decision}，但当前无仓位。无需操作。")
                    return

                # --- [学习闭环] 在平仓前记录信息 ---
                entry_trade_info = self._ongoing_trades.pop(symbol, None)
                exit_price = 0
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    exit_price = float(ticker['last'])
                except Exception as e:
                    LOGGER.warning(f"[{symbol}] 无法获取平仓时的市价: {e}")

                current_pos_side = position_info.get('posSide')
                decision_pos_side = 'long' if decision == 'CLOSE_LONG' else 'short'
                
                # 兼容OKX的net持仓模式
                original_pos_side = current_pos_side  # 保存原始的posSide
                if current_pos_side == 'net':
                    pos_amount_str = position_info.get('pos')
                    if pos_amount_str and float(pos_amount_str) > 0:
                        effective_side = 'long'
                    elif pos_amount_str and float(pos_amount_str) < 0:
                        effective_side = 'short'
                    else:
                        effective_side = 'none' # 无有效持仓
                    LOGGER.info(f"[{symbol}] 检测到 'net' 持仓模式, 根据持仓量判断实际方向为: {effective_side}")
                    current_pos_side = effective_side

                if current_pos_side != decision_pos_side:
                    LOGGER.error(f"[{symbol}] 决策平仓方向 ({decision_pos_side}) 与实际持仓方向 ({current_pos_side}) 不符！操作取消。")
                    return
                
                pos_amount_str = position_info.get('pos')
                contracts = position_info.get('contracts')
                # 判断持仓模式 - 使用原始的posSide来判断是否为net模式
                if original_pos_side == 'net':
                    # 单向持仓模式下，直接使用 'pos' 字段作为平仓数量（单位为BTC）
                    if not pos_amount_str:
                        LOGGER.error(f"[{symbol}] 无法从仓位信息中获取持仓数量 ('pos')。平仓操作取消。")
                        return
                    
                    amount = abs(float(pos_amount_str))
                    notional_usd = position_info.get('notionalUsd')
                    notional_value = abs(float(notional_usd)) if notional_usd else 0.0

                    LOGGER.info(f"[{symbol}] 准备平仓: {decision} {amount} (名义价值 ${notional_value:.2f})...")
                else:
                    # 双向持仓，amount用张数
                    if contracts is not None and pos_amount_str is not None and float(contracts) > 0:
                        amount = int(float(contracts))
                        btc_amount = float(pos_amount_str)
                    elif pos_amount_str is not None:
                        btc_amount = float(pos_amount_str)
                        ticker = self.exchange.fetch_ticker(symbol)
                        mark_price = ticker.get('last')
                        if mark_price is None:
                            LOGGER.error(f"[{symbol}] 无法获取当前价格，无法计算平仓张数。平仓操作取消。")
                            return
                        mark_price = float(mark_price)
                        contract_value = 100
                        amount = max(1, round(btc_amount * mark_price / contract_value))
                    else:
                        LOGGER.error(f"[{symbol}] 无法从仓位信息中获取持仓数量 ('pos'/'contracts')。平仓操作取消。")
                        return
                    LOGGER.info(f"[{symbol}] 准备平仓: {decision} {amount}张 (约{btc_amount})...")
                side = 'sell' if current_pos_side == 'long' else 'buy'
                order_params = {'tdMode': self.margin_mode, 'reduceOnly': True}
                if self.hedge_mode:
                    order_params['posSide'] = current_pos_side
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=side,
                    amount=amount,
                    params=order_params
                )
                if isinstance(order, dict) and 'code' in order:
                    LOGGER.error(f"[{symbol}] 平仓下单失败，返回错误: {order}")
                    # 如果平仓失败，把进行中的交易信息加回去
                    if entry_trade_info:
                        self._ongoing_trades[symbol] = entry_trade_info
                    raise RuntimeError(f"平仓下单失败: {order}")
                LOGGER.success(f"[{symbol}] 平仓 ({decision}) 订单已成功提交，订单ID: {order.get('id', 'N/A')}")

                # --- [学习闭环] 记录完整交易日志 ---
                if entry_trade_info:
                    try:
                        pnl = (exit_price - entry_trade_info['entry_price']) * entry_trade_info['entry_contracts']
                        # 对于空头，PnL方向相反
                        if decision == 'CLOSE_SHORT':
                            pnl = -pnl
                        
                        trade_log_entry = {
                            "symbol": symbol,
                            "entry_report": entry_trade_info['entry_report'],
                            "entry_timestamp_utc": entry_trade_info['entry_timestamp_utc'],
                            "entry_price": entry_trade_info['entry_price'],
                            "exit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "exit_price": exit_price,
                            "exit_reason": f"LLM Decision: {decision}",
                            "pnl": pnl
                        }
                        
                        trade_logs = self._load_json_db(TRADE_LOG_FILE, [])
                        trade_logs.append(trade_log_entry)
                        self._save_json_db(TRADE_LOG_FILE, trade_logs)
                        self._save_json_db(ONGOING_TRADES_FILE, self._ongoing_trades) # 保存移除后的 ongoing_trades
                        LOGGER.success(f"[{symbol}] 完整交易已记录到 {TRADE_LOG_FILE}")
                    except Exception as e:
                        LOGGER.error(f"[{symbol}] 记录完整交易日志时发生错误: {e}")
                        # 即使日志记录失败，也要确保进行中的交易被移除
                        self._save_json_db(ONGOING_TRADES_FILE, self._ongoing_trades)

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            LOGGER.error(f"[{symbol}] 执行交易决策时发生交易所错误: {e}", exc_info=True)
            self.email_notifier.send_error_notification(f"OKX交易所错误 ({symbol})", str(e))
        except Exception as e:
            LOGGER.error(f"[{symbol}] 执行交易决策时发生未知错误: {e}", exc_info=True)
            self.email_notifier.send_error_notification(f"OKX交易器未知错误 ({symbol})", str(e))

    def _set_stop_orders(self, symbol: str, main_order: Union[Dict[str, Any], Any], entry_price: Union[float, Decimal], params: Dict[str, Any], pos_side: str, order_amount: float = None):
        """
        设置止损和止盈订单前，先撤销当前symbol和pos_side的所有未成交止盈止损委托单。
        """
        try:
            # 先撤销未成交止盈止损单
            open_orders = self.exchange.fetch_open_orders(symbol=symbol)
            for o in open_orders:
                # 只撤销本方向的止盈止损单（不撤reduceOnly的平仓单）
                o_type = o.get('type')
                o_params = o.get('info', {})
                o_pos_side = o_params.get('posSide') if self.hedge_mode else pos_side
                reduce_only = o_params.get('reduceOnly', False)
                # OKX止损单type为'stop'，止盈单type为'limit'但带止盈价，且都不是reduceOnly
                if o.get('status') in ('open', 'new') and o_pos_side == pos_side and not reduce_only:
                    if o_type in ('stop', 'trigger', 'conditional') or (o_type == 'limit' and ('takeProfit' in o_params or 'tpTriggerPx' in o_params)):
                        try:
                            self.exchange.cancel_order(o['id'], symbol=symbol)
                            LOGGER.info(f"[{symbol}] 已撤销未成交止盈止损单: {o['id']} {o_type} {o_pos_side}")
                        except Exception as ce:
                            LOGGER.warning(f"[{symbol}] 撤销止盈止损单失败: {o['id']} {ce}")
        except Exception as e:
            LOGGER.warning(f"[{symbol}] 获取/撤销未成交止盈止损单时出错: {e}")
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
            LOGGER.info(f"[{symbol}] [止盈止损] entry_price={entry_price}, params={params}, pos_side={pos_side}, order_amount={order_amount}")
            if entry_price is None:
                LOGGER.error(f"[{symbol}] entry_price为None，无法设置止盈止损单。直接返回。")
                return
            
            entry_price = float(entry_price)
            stop_loss_pct = params.get('stop_loss_pct')
            take_profit_pct = params.get('take_profit_pct')
            
            # 优先使用传入的order_amount，如果没有则尝试从main_order获取
            if order_amount is not None:
                amount = float(order_amount)
                LOGGER.info(f"[{symbol}] [仓位计算] 使用传入的order_amount: {amount}")
            else:
                # 尝试从订单中获取数量
                order_amount_from_order = None
                if isinstance(main_order, dict):
                    order_amount_from_order = main_order.get('amount')
                else:
                    order_amount_from_order = getattr(main_order, 'amount', None)
                
                if order_amount_from_order is not None:
                    amount = float(order_amount_from_order)
                    LOGGER.info(f"[{symbol}] [仓位计算] 从main_order获取数量: {amount}")
                else:
                    LOGGER.error(f"[{symbol}] 无法获取订单数量，无法设置止盈止损单。main_order={main_order}, order_amount={order_amount}")
                    return
            
            # 确保数量精度正确
            amount = round(amount, 4)
            LOGGER.info(f"[{symbol}] [仓位计算] 最终使用数量: {amount}, 仓位方向: {pos_side}")
            
            stop_loss_price = None
            take_profit_price = None
            if stop_loss_pct:
                if pos_side == 'long':
                    stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
                else:
                    stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
                LOGGER.info(f"[{symbol}] [止损] stop_loss_pct={stop_loss_pct}, stop_loss_price={stop_loss_price}")
                if stop_loss_price is None:
                    LOGGER.error(f"[{symbol}] stop_loss_price为None，跳过止损单下单。")
                else:
                    # 使用OKX标准计划委托（trigger单），防止下单即成交
                    stop_order_params = {
                        'tdMode': self.margin_mode,
                        'triggerPrice': stop_loss_price,
                        'orderType': 'market',  # 触发后市价
                        'orderPx': '',  # 市价计划委托必须传递orderPx字段
                    }
                    if self.hedge_mode:
                        stop_order_params['posSide'] = pos_side
                    LOGGER.info(f"[{symbol}] 提交止损计划委托单，数量: {amount}, 类型: {type(amount)} triggerPrice={stop_loss_price}")
                    stop_order = self.exchange.create_order(
                        symbol=symbol,
                        type='trigger',  # 计划委托
                        side='sell' if pos_side == 'long' else 'buy',
                        amount=amount,
                        params=stop_order_params
                    )
                    if isinstance(stop_order, dict) and 'code' in stop_order:
                        LOGGER.error(f"[{symbol}] 止损单下单失败，返回错误: {stop_order}")
                        raise RuntimeError(f"止损单下单失败: {stop_order}")
                    LOGGER.info(f"[{symbol}] 止损计划委托已设置: 触发价 ${stop_loss_price:.2f}")
            if take_profit_pct:
                if pos_side == 'long':
                    take_profit_price = entry_price * (1 + take_profit_pct / 100)
                else:
                    take_profit_price = entry_price * (1 - take_profit_pct / 100)
                LOGGER.info(f"[{symbol}] [止盈] take_profit_pct={take_profit_pct}, take_profit_price={take_profit_price}")
                if take_profit_price is None:
                    LOGGER.error(f"[{symbol}] take_profit_price为None，跳过止盈单下单。")
                else:
                    take_profit_order_params = {
                        'tdMode': self.margin_mode
                    }
                    if self.hedge_mode:
                        take_profit_order_params['posSide'] = pos_side
                    LOGGER.info(f"[{symbol}] 提交止盈单，数量: {amount}, 类型: {type(amount)}")
                    take_profit_order = self.exchange.create_order(
                        symbol=symbol,
                        type='limit',
                        side='sell' if pos_side == 'long' else 'buy',
                        amount=amount,
                        price=take_profit_price,
                        params=take_profit_order_params
                    )
                    if isinstance(take_profit_order, dict) and 'code' in take_profit_order:
                        LOGGER.error(f"[{symbol}] 止盈单下单失败，返回错误: {take_profit_order}")
                        raise RuntimeError(f"止盈单下单失败: {take_profit_order}")
                    LOGGER.info(f"[{symbol}] 止盈订单已设置: 价格 ${take_profit_price:.2f}")
        except Exception as e:
            LOGGER.error(f"[{symbol}] 设置止损/止盈订单时发生错误: {e}")
            self.email_notifier.send_error_notification(f"止盈止损设置错误 ({symbol})", str(e))

    def get_balance(self, currency: str = 'USDT'):
        """
        获取指定货币的可用保证金 (available equity)。
        OKX返回的是一个包含多个币种信息的列表，我们需要找到USDT并提取'availEq'。
        """
        if self.demo_mode:
            LOGGER.info(f"模拟盘模式: 返回固定可用保证金 {currency}: 500.0")
            return 500.0
        try:
            # 获取账户余额信息
            balance = self.exchange.fetch_balance()
            
            # OKX v5 API的响应结构通常在 'info' 字段中
            # 路径: info -> data -> [0] -> details -> [...]
            if 'info' in balance and 'data' in balance['info'] and balance['info']['data']:
                details = balance['info']['data'][0].get('details', [])
                for asset in details:
                    if asset.get('ccy') == currency:
                        # 'availEq' 是可用作保证金的权益（以USD计价），这是最准确的指标
                        available_equity = float(asset.get('availEq', 0))
                        if available_equity > 0:
                            LOGGER.info(f"获取到 {currency} 可用保证金 (availEq): {available_equity}")
                            return available_equity
                        else:
                            # 如果availEq为0，可能是因为没有仓位，此时用可用余额availBal
                            available_balance = float(asset.get('availBal', 0))
                            LOGGER.info(f"可用保证金为0，回退到可用余额 (availBal): {available_balance}")
                            return available_balance

            # 如果上述路径找不到，尝试备用方案
            if currency in balance:
                available_balance = float(balance[currency].get('free', 0))
                if available_balance > 0:
                    LOGGER.warning(f"备用方案: 获取到 {currency} 可用余额 (free): {available_balance}")
                    return available_balance

            LOGGER.error(f"无法在余额响应中找到 {currency} 的可用保证金。响应: {balance}")
            return 0
        except Exception as e:
            LOGGER.error("获取可用保证金时出错: {}", e, exc_info=True)
            return 0 