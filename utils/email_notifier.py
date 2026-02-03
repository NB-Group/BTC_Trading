import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import config
from btc_predictor.utils import LOGGER

class EmailNotifier:
    """邮件通知器，用于发送交易决策和错误通知"""
    
    def __init__(self):
        self.config = config.EMAIL_CONFIG
        self.enabled = self.config['enabled']
        
        if not self.enabled:
            LOGGER.info("邮件通知功能已禁用")
            return
            
        # 验证配置
        required_fields = ['smtp_server', 'smtp_port', 'from_email', 'auth_code']
        missing_fields = [field for field in required_fields if not self.config.get(field)]

        to_emails = self.config.get('to_emails') or []
        if not to_emails:
            missing_fields.append('to_emails')
        else:
            self.config['to_emails'] = to_emails
        
        if missing_fields:
            LOGGER.warning(f"邮件配置不完整，缺少字段: {missing_fields}")
            self.enabled = False
            return
            
        LOGGER.info(f"邮件通知器已初始化，发件人: {self.config['from_email']}")

    def send_decision_notification(self, decision_data: Dict[str, Any], execution_success: bool = True, error_msg: str = None, process_status: Dict[str, Any] = None):
        """发送交易决策通知"""
        if not self.enabled:
            return
            
        try:
            subject = self._get_decision_subject(decision_data, execution_success)
            html_content = self._create_decision_email_html(decision_data, execution_success, error_msg, process_status)
            
            self._send_email(subject, html_content)
            LOGGER.info("交易决策邮件通知已发送")
            
        except Exception as e:
            if (str(e) == "(-1, b'\x00\x00\x00')"):
                LOGGER.info("发送交易决策邮件成功")
            else:
                LOGGER.error(f"发送交易决策邮件失败: {e}")

    def send_error_notification(self, error_type: str, error_msg: str, context: Dict[str, Any] = None):
        """发送错误通知"""
        if not self.enabled:
            return
            
        try:
            subject = f"🚨 BTC交易系统错误 - {error_type}"
            html_content = self._create_error_email_html(error_type, error_msg, context)
            
            self._send_email(subject, html_content)
            LOGGER.info("错误通知邮件已发送")
            
        except Exception as e:
            if (str(e) == "(-1, b'\x00\x00\x00')"):
                LOGGER.info("发送错误通知邮件成功")
            else:
                LOGGER.error(f"发送错误通知邮件失败: {e}")

    def _get_decision_subject(self, decision_data: Dict[str, Any], execution_success: bool) -> str:
        """生成邮件主题"""
        decision = decision_data.get('decision', 'UNKNOWN').upper()
        current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        if execution_success:
            if decision in ['LONG', 'SHORT']:
                return f"📈 BTC交易开仓成功 - {decision} ({current_time})"
            elif decision in ['CLOSE_LONG', 'CLOSE_SHORT']:
                return f"📉 BTC交易平仓成功 - {decision} ({current_time})"
            else:
                return f"⏸️ BTC交易决策 - {decision} ({current_time})"
        else:
            return f"❌ BTC交易执行失败 - {decision} ({current_time})"

    def _create_decision_email_html(self, decision_data: Dict[str, Any], execution_success: bool, error_msg: str = None, process_status: Dict[str, Any] = None) -> str:
        """创建决策邮件的HTML内容"""
        decision = decision_data.get('decision', 'UNKNOWN').upper()
        reasoning = decision_data.get('reasoning', '')
        key_signals = decision_data.get('key_signals_detected', '')
        risk_assessment = decision_data.get('risk_assessment', '')
        trade_params = decision_data.get('trade_params', {})
        position_snapshot = decision_data.get('position_snapshot')
        
        current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # 决策状态图标（根据流程状态自动识别“部分失败”）
        flow_has_error = False
        if process_status:
            try:
                for _k, _v in process_status.items():
                    if isinstance(_v, dict) and _v.get('status') == 'error':
                        flow_has_error = True
                        break
            except Exception:
                pass

        if execution_success:
            if flow_has_error:
                status_icon = '<i class="fas fa-exclamation-triangle"></i>'
                status_text = "部分失败（详见流程状态）"
            else:
                status_icon = '<i class="fas fa-check-circle"></i>' if decision in ['LONG', 'SHORT', 'CLOSE_LONG', 'CLOSE_SHORT'] else '<i class="fas fa-pause-circle"></i>'
            status_text = "执行成功" if decision in ['LONG', 'SHORT', 'CLOSE_LONG', 'CLOSE_SHORT'] else "观望中"
        else:
            status_icon = '<i class="fas fa-times-circle"></i>'
            status_text = "执行失败"
            
        # 决策类型颜色
        decision_colors = {
            'LONG': '#28a745',      # 绿色
            'SHORT': '#dc3545',     # 红色
            'CLOSE_LONG': '#ffc107', # 黄色
            'CLOSE_SHORT': '#ffc107', # 黄色
            'HOLD': '#17a2b8'       # 蓝色
        }
        decision_color = decision_colors.get(decision, '#6c757d')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * {{ box-sizing: border-box; }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #495057;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    background: white;
                    border-radius: 16px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                    overflow: hidden;
                }}
                .header {{
                    background: #f8f9fa;
                    padding: 16px 20px;
                    border-bottom: 1px solid #dee2e6;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 18px;
                    font-weight: 600;
                    color: #495057;
                }}
                .header .subtitle {{
                    margin-top: 4px;
                    font-size: 12px;
                    color: #6c757d;
                }}
                .content {{
                    padding: 24px;
                }}
                .decision-card {{
                    background: #ffffff;
                    border: 1px solid #e9ecef;
                    border-radius: 12px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }}
                .decision-header {{
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    margin-bottom: 12px;
                }}
                .decision-badge {{
                    background: {decision_color};
                    color: white;
                    padding: 6px 12px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                .status-info {{
                    margin-left: auto;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 13px;
                    color: #6c757d;
                }}
                .decision-time {{
                    font-size: 12px;
                    color: #adb5bd;
                    margin-top: 8px;
                }}
                .info-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 16px;
                    margin-bottom: 24px;
                }}
                .info-card {{
                    background: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                    padding: 16px;
                }}
                .info-card h3 {{
                    margin: 0 0 8px 0;
                    font-size: 14px;
                    font-weight: 600;
                    color: #495057;
                }}
                .info-card p {{
                    margin: 0;
                    font-size: 13px;
                    color: #6c757d;
                }}
                .section {{
                    margin-bottom: 20px;
                }}
                .section h3 {{
                    font-size: 14px;
                    font-weight: 600;
                    color: #495057;
                    margin-bottom: 12px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}
                .section p {{
                    background: #ffffff;
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                    padding: 16px;
                    margin: 0;
                    line-height: 1.6;
                    color: #495057;
                }}
                .error-section {{
                    background: #fff5f5;
                    border: 1px solid #fed7d7;
                    border-radius: 8px;
                    padding: 16px;
                    margin-top: 16px;
                }}
                .error-section h3 {{
                    color: #c53030;
                    margin-top: 0;
                    font-size: 14px;
                }}
                .error-section p {{
                    color: #c53030;
                    margin: 0;
                }}
                .footer {{
                    background: #f8f9fa;
                    padding: 16px;
                    text-align: center;
                    color: #6c757d;
                    font-size: 12px;
                    border-top: 1px solid #dee2e6;
                }}
                /* 参数键值行样式 */
                .kv {{
                    display: grid;
                    gap: 8px;
                }}
                .kv .row {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    background: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 6px;
                    padding: 12px;
                }}
                .kv .row span {{
                    color: #6c757d;
                    font-size: 12px;
                    font-weight: 500;
                }}
                .kv .row strong {{
                    color: #495057;
                    font-size: 13px;
                    font-weight: 600;
                }}

                /* 移动端优化 */
                @media (max-width: 520px) {{
                    body {{ padding: 12px; }}
                    .content {{ padding: 16px; }}
                    .info-grid {{ grid-template-columns: 1fr; }}
                    .header h1 {{ font-size: 16px; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1><i class="fas fa-robot"></i> BTC 期货智能决策系统</h1>
                    <div class="subtitle">自动交易决策通知</div>
                </div>
                
                <div class="content">
                    <div class="decision-card">
                        <div class="decision-header">
                            <span class="decision-badge">{decision}</span>
                            <div class="status-info">
                                <span>{status_icon}</span>
                                <span>{status_text}</span>
                            </div>
                        </div>
                        <div class="decision-time">{current_time}</div>
                    </div>
                    
                    <div class="info-grid">
                        
                        <div class="info-card">
                            <h3><i class="fas fa-clock"></i> 决策时间</h3>
                            <p>{current_time}</p>
                        </div>
        """
        # 可选：持仓卡片（有持仓显示盈亏；无持仓显示状态）
        if position_snapshot and isinstance(position_snapshot, dict):
            try:
                if position_snapshot.get('status') == 'no_position' or position_snapshot.get('no_position'):
                    pos_desc = position_snapshot.get('desc', '当前无持仓')
                    html += f"""
                        <div class="info-card">
                            <h3><i class="fas fa-box-open"></i> 持仓状态</h3>
                            <p style="font-weight: 600;">{pos_desc}</p>
                        </div>
                    """
                else:
                    pnl_usd = float(position_snapshot.get('pnl_usd', 0.0))
                    pnl_color = '#28a745' if pnl_usd >= 0 else '#dc3545'
                    pnl_prefix = '+' if pnl_usd >= 0 else '-'
                    pos_desc = position_snapshot.get('desc', '')
                    html += f"""
                        <div class="info-card">
                            <h3><i class="fas fa-dollar-sign"></i> 持仓盈亏</h3>
                            <p style=\"font-weight: 600; color: {pnl_color};\">{pnl_prefix}${abs(pnl_usd):.2f} USDT</p>
                            <div style=\"font-size: 12px; color: #6c757d;\">{pos_desc}</div>
                        </div>
                    """
            except Exception:
                # 回退：若无法解析盈亏但有描述，仍显示持仓状态
                try:
                    pos_desc = position_snapshot.get('desc')
                    if pos_desc:
                        html += f"""
                        <div class=\"info-card\">
                            <h3><i class=\"fas fa-box-open\"></i> 持仓状态</h3>
                            <p style=\"font-weight: 600;\">{pos_desc}</p>
                        </div>
                        """
                except Exception:
                    pass
        html += f"""
                    </div>
                    
                    <div class="section">
                        <h3><i class="fas fa-cog"></i> 交易参数</h3>
                        <div class="kv">
                            <div class="row"><span>杠杆</span><strong>{trade_params.get('leverage', 'N/A')}x</strong></div>
                            <div class="row"><span>止盈</span><strong>{trade_params.get('take_profit_pct', 'N/A')}%</strong></div>
                            <div class="row"><span>止损</span><strong>{trade_params.get('stop_loss_pct', 'N/A')}%</strong></div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h3><i class="fas fa-brain"></i> 决策理由</h3>
                        <p>{reasoning}</p>
                    </div>
                    
                    <div class="section">
                        <h3><i class="fas fa-search"></i> 关键信号</h3>
                        <p>{key_signals}</p>
                    </div>
                    
                    <div class="section">
                        <h3><i class="fas fa-exclamation-triangle"></i> 风险评估</h3>
                        <p>{risk_assessment}</p>
                    </div>
        """
        
        # 添加流程运行状态部分
        if process_status:
            html += self._create_process_status_html(process_status)
        
        if error_msg:
            html += f"""
                    <div class="error-section">
                        <h3><i class="fas fa-times-circle"></i> 执行错误</h3>
                        <p>{error_msg}</p>
                    </div>
            """
            
        html += """
                </div>
                
                <div class="footer">
                    <p>此邮件由 BTC 期货智能决策系统自动发送</p>
                    <p>请勿回复此邮件</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html

    def _create_process_status_html(self, process_status: Dict[str, Any]) -> str:
        """创建流程运行状态的HTML内容"""
        html = """
                    <div class="section">
                        <h3><i class="fas fa-cogs"></i> 流程运行状态</h3>
        """
        
        # 流程状态映射
        status_icons = {
            'success': '<i class="fas fa-check-circle"></i>',
            'error': '<i class="fas fa-times-circle"></i>',
            'warning': '<i class="fas fa-exclamation-triangle"></i>',
            'info': '<i class="fas fa-info-circle"></i>',
            'pending': '<i class="fas fa-clock"></i>'
        }
        
        # 流程顺序
        process_order = [
            'data_collection',
            'vlm_analysis', 
            'news_intelligence',
            'llm_decision',
            'trade_execution'
        ]
        
        for process_key in process_order:
            if process_key in process_status:
                process_info = process_status[process_key]
                status = process_info.get('status', 'pending')
                icon = status_icons.get(status, '❓')
                
                # 流程名称映射
                process_names = {
                    'data_collection': '数据获取',
                    'vlm_analysis': 'VLM技术分析',
                    'news_intelligence': '新闻情报收集',
                    'llm_decision': 'LLM决策分析',
                    'trade_execution': '交易执行'
                }
                
                process_name = process_names.get(process_key, process_key)
                duration = process_info.get('duration', 'N/A')
                message = process_info.get('message', '')
                error = process_info.get('error', '')
                
                # 状态颜色
                status_colors = {
                    'success': '#28a745',
                    'error': '#dc3545', 
                    'warning': '#ffc107',
                    'info': '#17a2b8',
                    'pending': '#6c757d'
                }
                status_color = status_colors.get(status, '#6c757d')
                
                html += f"""
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 2px solid {status_color};">
                            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                <span style="font-size: 18px; margin-right: 10px;">{icon}</span>
                                <strong style="color: #495057;">{process_name}</strong>
                                <span style="margin-left: auto; color: {status_color}; font-size: 12px;">{duration}</span>
                            </div>
                """
                
                if message:
                    html += f'<div style="color: #6c757d; font-size: 14px; margin-left: 28px;">{message}</div>'
                
                if error:
                    html += f'<div style="color: #dc3545; font-size: 14px; margin-left: 28px; margin-top: 5px;"><i class="fas fa-times-circle"></i> {error}</div>'
                
                html += "</div>"
        
        html += """
                    </div>
        """
        
        return html

    def _create_error_email_html(self, error_type: str, error_msg: str, context: Dict[str, Any] = None) -> str:
        """创建错误通知邮件的HTML内容"""
        current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .container {{
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 28px;
                    font-weight: 300;
                }}
                .content {{
                    padding: 30px;
                }}
                .error-card {{
                    background: #f8d7da;
                    border: 1px solid #f5c6cb;
                    border-radius: 8px;
                    padding: 25px;
                    margin-bottom: 25px;
                }}
                .error-card h2 {{
                    color: #721c24;
                    margin-top: 0;
                    font-size: 24px;
                }}
                .error-card p {{
                    color: #721c24;
                    margin: 0;
                    font-size: 16px;
                    line-height: 1.8;
                }}
                .info-section {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                }}
                .info-section h3 {{
                    color: #495057;
                    margin-top: 0;
                }}
                .footer {{
                    background: #e9ecef;
                    padding: 20px;
                    text-align: center;
                    color: #6c757d;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1><i class="fas fa-exclamation-triangle"></i> 系统错误通知</h1>
                </div>
                
                <div class="content">
                    <div class="error-card">
                        <h2><i class="fas fa-times-circle"></i> {error_type}</h2>
                        <p>{error_msg}</p>
                    </div>
                    
                    <div class="info-section">
                        <h3><i class="fas fa-clock"></i> 错误时间</h3>
                        <p>{current_time}</p>
                    </div>
        """
        
        if context:
            html += f"""
                    <div class="info-section">
                        <h3><i class="fas fa-clipboard-list"></i> 错误上下文</h3>
                        <p>
            """
            for key, value in context.items():
                html += f"<strong>{key}:</strong> {value}<br>"
            html += """
                        </p>
                    </div>
            """
            
        html += """
                </div>
                
                <div class="footer">
                    <p>此邮件由 BTC 期货智能决策系统自动发送</p>
                    <p>请及时检查系统状态</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html

    def _send_email(self, subject: str, html_content: str):
        """发送邮件"""
        if not self.enabled:
            return
            
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.config['from_email']
            recipients = self.config.get('to_emails') or []
            msg['To'] = ', '.join(recipients)
            
            # 添加HTML内容
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
            
            # 创建SSL上下文
            context = ssl.create_default_context()
            
            # 连接SMTP服务器并发送邮件
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                if self.config['use_tls']:
                    server.starttls(context=context)
                server.login(self.config['from_email'], self.config['auth_code'])
                server.send_message(msg, from_addr=self.config['from_email'], to_addrs=recipients)
                
        except Exception as e:
            LOGGER.error(f"发送邮件失败: {e}")
            raise 