# BTC Trading System

一个基于机器学习和人工智能的比特币期货智能交易系统，集成了量化模型、视觉语言模型(VLM)分析和社交媒体情报分析。

## 🚀 主要特性

- **🤖 多模型融合决策**: 结合量化模型、VLM图表分析和社交媒体情报
- **📊 期货交易支持**: 支持做多、做空、杠杆交易和仓位管理
- **🔍 智能信号识别**: "大海捞针"式关键信号检测，识别地缘政治风险和市场转折点
- **📈 技术指标丰富**: K线图包含MA、布林带、RSI等多种技术指标
- **🔄 实时监控**: 支持定时任务和实时决策
- **🛡️ 风险管理**: 内置风险控制和仓位管理机制

## 📋 系统架构

```
BTC Trading System
├── btc_predictor/          # 量化模型模块
│   ├── model.py           # 机器学习模型
│   ├── predict.py         # 预测引擎
│   ├── kline_plot.py      # K线图生成
│   └── ...
├── data_ingestion/        # 数据获取模块
│   └── news_feeds.py      # 新闻数据获取
├── decision_engine/       # 决策引擎
│   ├── deepseek_analyzer.py  # LLM决策分析
│   ├── vlm_analyzer.py    # 视觉语言模型
│   └── vlm_cache.py       # VLM缓存管理
├── execution_engine/      # 交易执行模块
│   └── okx_trader.py      # OKX交易所接口
└── main.py               # 主程序入口
```

## 🛠️ 安装指南

### 1. 克隆仓库
```bash
git clone https://github.com/yourusername/btc-trading-system.git
cd btc-trading-system
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量
```bash
# 复制环境变量模板
cp env.example .env

# 编辑 .env 文件，填入您的API密钥
```

### 4. 配置API密钥
在 `.env` 文件中配置以下API密钥：

```env
# OKX交易所API
OKX_API_KEY=your_okx_api_key
OKX_SECRET_KEY=your_okx_secret_key
OKX_PASSPHRASE=your_okx_passphrase

# DeepSeek API (用于LLM分
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.siliconflow.cn/v1
DEEPSEEK_MODEL=deepseek-ai/DeepSeek-R1

# 代理设置 (可选)
PROXY_URL=http://127.0.0.1:7890

# 运行模式
DEMO_MODE=true  # true=模拟盘, false=实盘
```

## 🎯 使用方法

### 立即运行一次决策
```bash
python main.py --now
```

### 跳过LLM分析（用于调试）
```bash
python main.py --now --skip-llm
```

### 启动定时任务（每小时执行一次）
```bash
python main.py
```

### 启动仪表板
```bash
python run_dashboard.py
```

## 🔧 配置说明

### 期货交易设置
在 `config.py` 中修改期货交易参数：

```python
FUTURES = {
    'trade_symbol': 'BTC-USDT-SWAP',  # 交易对
    'leverage': 5,                    # 杠杆倍数
    'margin_mode': 'isolated',        # 保证金模式
}
```

### 监控设置
配置重点监控的Twitter用户和宏观经济关键词：

```python
INFLUENTIAL_TWITTER_USERS = [
    "elonmusk",
    "realDonaldTrump",
    # ... 更多用户
]

MACRO_ECONOMIC_KEYWORDS = [
    '"Federal Reserve" OR "Fed" interest rates',
    '"CPI report" OR "inflation data"',
    # ... 更多关键词
]
```

## 📊 决策流程

1. **数据获取**: 获取价格数据、新闻情报
2. **量化分析**: 运行机器学习模型生成信号
3. **K线分析**: VLM模型分析技术图表
4. **情报整合**: 分析社交媒体和新闻信息
5. **综合决策**: LLM综合所有信息做出交易决策
6. **执行交易**: 根据决策执行期货交易

## 🛡️ 安全注意事项

- ⚠️ **请勿将 `.env` 文件提交到版本控制系统**
- ⚠️ **实盘交易前请充分测试模拟盘**
- ⚠️ **请合理设置杠杆倍数，控制风险**
- ⚠️ **定期备份重要数据和配置**

## 📈 性能监控

系统提供详细的日志记录和性能分析：

- 交易决策报告保存在 `decision_report.json`
- 运行日志保存在 `logs/` 目录
- 缓存文件保存在 `cache/` 目录

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## ⚠️ 免责声明

本软件仅供学习和研究使用。加密货币交易存在高风险，可能导致资金损失。使用本软件进行实际交易的风险由用户自行承担。作者不对任何交易损失承担责任。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue: [GitHub Issues](https://github.com/yourusername/btc-trading-system/issues)
- 邮箱: your.email@example.com

---

**⭐ 如果这个项目对您有帮助，请给个星标！** 