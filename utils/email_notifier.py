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
        required_fields = ['smtp_server', 'smtp_port', 'from_email', 'to_email', 'auth_code']
        missing_fields = [field for field in required_fields if not self.config.get(field)]
        
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
        confidence = decision_data.get('confidence', 0)
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
                status_icon = "⚠️"
                status_text = "部分失败（详见流程状态）"
            else:
                status_icon = "✅" if decision in ['LONG', 'SHORT', 'CLOSE_LONG', 'CLOSE_SHORT'] else "⏸️"
                status_text = "执行成功" if decision in ['LONG', 'SHORT', 'CLOSE_LONG', 'CLOSE_SHORT'] else "观望中"
        else:
            status_icon = "❌"
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
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 28px;
                    font-weight: 300;
                }}
                .header .subtitle {{
                    margin-top: 10px;
                    opacity: 0.9;
                    font-size: 16px;
                }}
                .content {{
                    padding: 30px;
                }}
                .decision-card {{
                    background: {decision_color};
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    margin-bottom: 25px;
                }}
                .decision-card h2 {{
                    margin: 0;
                    font-size: 32px;
                    font-weight: bold;
                }}
                .decision-card .status {{
                    margin-top: 10px;
                    font-size: 18px;
                    opacity: 0.9;
                }}
                .info-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-bottom: 25px;
                }}
                .info-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #007bff;
                }}
                .info-card h3 {{
                    margin: 0 0 10px 0;
                    color: #007bff;
                    font-size: 18px;
                }}
                .info-card p {{
                    margin: 0;
                    font-size: 14px;
                }}
                .section {{
                    margin-bottom: 25px;
                }}
                .section h3 {{
                    color: #495057;
                    border-bottom: 2px solid #e9ecef;
                    padding-bottom: 10px;
                    margin-bottom: 15px;
                }}
                .section p {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 6px;
                    margin: 0;
                    line-height: 1.8;
                }}
                .error-section {{
                    background: #f8d7da;
                    border: 1px solid #f5c6cb;
                    border-radius: 8px;
                    padding: 20px;
                    margin-top: 20px;
                }}
                .error-section h3 {{
                    color: #721c24;
                    margin-top: 0;
                }}
                .error-section p {{
                    color: #721c24;
                    margin: 0;
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
                    <h1>🤖 BTC 期货智能决策系统</h1>
                    <div class="subtitle">自动交易决策通知</div>
                </div>
                
                <div class="content">
                    <div class="decision-card">
                        <h2>{status_icon} {decision}</h2>
                        <div class="status">{status_text}</div>
                    </div>
                    
                    <div class="info-grid">
                        <div class="info-card">
                            <h3>📊 置信度</h3>
                            <p>{confidence:.1%}</p>
                        </div>
                        <div class="info-card">
                            <h3>⏰ 决策时间</h3>
                            <p>{current_time}</p>
                        </div>
        """
        # 可选：持仓盈亏卡片
        if position_snapshot and isinstance(position_snapshot, dict):
            try:
                pnl_usd = float(position_snapshot.get('pnl_usd', 0.0))
                pnl_color = '#28a745' if pnl_usd >= 0 else '#dc3545'
                pnl_prefix = '+' if pnl_usd >= 0 else '-'
                pos_desc = position_snapshot.get('desc', '')
                html += f"""
                        <div class="info-card">
                            <h3>💰 持仓盈亏</h3>
                            <p style="font-weight: 600; color: {pnl_color};">{pnl_prefix}${abs(pnl_usd):.2f} USDT</p>
                            <div style="font-size: 12px; color: #6c757d;">{pos_desc}</div>
                        </div>
                """
            except Exception:
                pass
        html += """
                    </div>
                    
                    <div class="section">
                        <h3>🎯 交易参数</h3>
                        <p>
                            <strong>杠杆:</strong> {trade_params.get('leverage', 'N/A')}x<br>
                            <strong>止盈:</strong> {trade_params.get('take_profit_pct', 'N/A')}%<br>
                            <strong>止损:</strong> {trade_params.get('stop_loss_pct', 'N/A')}%
                        </p>
                    </div>
                    
                    <div class="section">
                        <h3>🧠 决策理由</h3>
                        <p>{reasoning}</p>
                    </div>
                    
                    <div class="section">
                        <h3>🔍 关键信号</h3>
                        <p>{key_signals}</p>
                    </div>
                    
                    <div class="section">
                        <h3>⚠️ 风险评估</h3>
                        <p>{risk_assessment}</p>
                    </div>
        """
        
        # 添加流程运行状态部分
        if process_status:
            html += self._create_process_status_html(process_status)
        
        if error_msg:
            html += f"""
                    <div class="error-section">
                        <h3>❌ 执行错误</h3>
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
                        <h3>🔧 流程运行状态</h3>
        """
        
        # 流程状态映射
        status_icons = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️',
            'pending': '⏳'
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
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid {status_color};">
                            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                <span style="font-size: 18px; margin-right: 10px;">{icon}</span>
                                <strong style="color: #495057;">{process_name}</strong>
                                <span style="margin-left: auto; color: {status_color}; font-size: 12px;">{duration}</span>
                            </div>
                """
                
                if message:
                    html += f'<div style="color: #6c757d; font-size: 14px; margin-left: 28px;">{message}</div>'
                
                if error:
                    html += f'<div style="color: #dc3545; font-size: 14px; margin-left: 28px; margin-top: 5px;">❌ {error}</div>'
                
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
                    <h1>🚨 系统错误通知</h1>
                </div>
                
                <div class="content">
                    <div class="error-card">
                        <h2>❌ {error_type}</h2>
                        <p>{error_msg}</p>
                    </div>
                    
                    <div class="info-section">
                        <h3>⏰ 错误时间</h3>
                        <p>{current_time}</p>
                    </div>
        """
        
        if context:
            html += """
                    <div class="info-section">
                        <h3>📋 错误上下文</h3>
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
            msg['To'] = self.config['to_email']
            
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
                server.send_message(msg)
                
        except Exception as e:
            LOGGER.error(f"发送邮件失败: {e}")
            raise 