# 邮件通知功能配置指南

## 概述

BTC交易系统现在支持邮件通知功能，可以在以下情况自动发送精美的邮件通知：

- ✅ DeepSeek做出交易决策后
- ❌ 开平仓执行失败时
- 🚨 系统发生严重错误时

## 配置步骤

### 1. 复制环境变量模板

```bash
cp env.example .env
```

### 2. 编辑 .env 文件

在 `.env` 文件中配置以下邮件相关参数：

```env
# Email Notification Configuration
EMAIL_ENABLED=true
EMAIL_SMTP_SERVER=smtp.qq.com
EMAIL_SMTP_PORT=587
EMAIL_FROM=your_email@qq.com
EMAIL_TO=your_email@qq.com
EMAIL_AUTH_CODE=qlyoeszgeastbidh
EMAIL_USE_TLS=true
```

### 3. 配置说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `EMAIL_ENABLED` | 是否启用邮件通知 | `true` / `false` |
| `EMAIL_SMTP_SERVER` | SMTP服务器地址 | `smtp.qq.com` (QQ邮箱) |
| `EMAIL_SMTP_PORT` | SMTP端口 | `587` (QQ邮箱) |
| `EMAIL_FROM` | 发件人邮箱 | `your_email@qq.com` |
| `EMAIL_TO` | 收件人邮箱 | `your_email@qq.com` |
| `EMAIL_AUTH_CODE` | SMTP/IMAP授权码 | `qlyoeszgeastbidh` |
| `EMAIL_USE_TLS` | 是否使用TLS加密 | `true` |

### 4. 获取QQ邮箱SMTP/IMAP授权码

1. 登录QQ邮箱网页版
2. 点击"设置" → "账户"
3. 找到"POP3/IMAP/SMTP/Exchange/CardDAV/CalDAV服务"
4. 开启"POP3/SMTP服务"
5. 按照提示获取**授权码**（不是QQ密码）

### 5. 测试邮件功能

运行测试脚本验证配置是否正确：

```bash
python test_email.py
```

如果看到"✅ 邮件发送成功"的消息，说明配置正确。

## 邮件模板预览

### 决策通知邮件
- 📈 开仓成功通知（绿色主题）
- 📉 平仓成功通知（黄色主题）
- ⏸️ 观望决策通知（蓝色主题）
- ❌ 执行失败通知（红色主题）

### 错误通知邮件
- 🚨 系统错误通知（红色主题）
- 📋 包含错误上下文信息

## 邮件内容包含

### 决策通知包含：
- 决策类型和状态
- 置信度
- 决策时间
- 交易参数（杠杆、止盈、止损）
- 决策理由
- 关键信号
- 风险评估
- 执行错误信息（如果有）

### 错误通知包含：
- 错误类型
- 错误消息
- 错误时间
- 错误上下文信息

## 注意事项

1. **安全性**：授权码不要泄露给他人
2. **频率**：系统会在每次决策周期后发送邮件，避免过于频繁
3. **网络**：确保服务器能访问SMTP服务器
4. **配置**：修改配置后需要重启系统

## 故障排除

### 常见问题

1. **邮件发送失败**
   - 检查SMTP服务器和端口是否正确
   - 确认SMTP/IMAP授权码是否正确（不是邮箱密码）
   - 检查网络连接

2. **收不到邮件**
   - 检查垃圾邮件文件夹
   - 确认收件人邮箱地址正确
   - 检查邮件服务器设置

3. **配置不生效**
   - 确认 `.env` 文件在项目根目录
   - 重启应用程序
   - 检查环境变量是否正确加载

4. **授权码问题**
   - 确认使用的是SMTP/IMAP授权码，不是邮箱登录密码
   - 如果授权码过期，需要重新生成
   - 确保已开启POP3/SMTP服务

### 日志查看

邮件发送的详细日志会记录在系统日志中，可以通过以下方式查看：

```bash
# 查看最近的日志
tail -f logs/btc_trading.log
```

## 自定义配置

如需使用其他邮箱服务商，请参考以下配置：

### Gmail
```env
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USE_TLS=true
```

### 163邮箱
```env
EMAIL_SMTP_SERVER=smtp.163.com
EMAIL_SMTP_PORT=587
EMAIL_USE_TLS=true
```

### 企业邮箱
请咨询您的IT管理员获取正确的SMTP配置。 