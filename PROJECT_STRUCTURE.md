# 项目结构说明

## 📁 目录结构

```
BTC-Trading/
├── 📄 README.md                    # 项目说明文档
├── 📄 LICENSE                      # MIT许可证
├── 📄 requirements.txt             # Python依赖包
├── 📄 config.py                    # 系统配置文件
├── 📄 main.py                      # 主程序入口
├── 📄 env.example                  # 环境变量模板
├── 📄 .gitignore                   # Git忽略文件
├── 📄 PROJECT_STRUCTURE.md         # 项目结构说明
│
├── 📁 btc_predictor/               # 量化预测模块
│   ├── 📄 __init__.py
│   ├── 📄 model.py                 # 机器学习模型定义
│   ├── 📄 predict.py               # 预测引擎
│   ├── 📄 train.py                 # 模型训练
│   ├── 📄 data.py                  # 数据获取和处理
│   ├── 📄 features.py              # 特征工程
│   ├── 📄 kline_plot.py            # K线图生成
│   ├── 📄 backtest.py              # 回测功能
│   ├── 📄 evaluate.py              # 模型评估
│   ├── 📄 optimize.py              # 超参数优化
│   ├── 📄 config.py                # 模型配置
│   └── 📄 utils.py                 # 工具函数
│
├── 📁 data_ingestion/              # 数据获取模块
│   ├── 📄 __init__.py
│   └── 📄 news_feeds.py            # 新闻数据获取
│
├── 📁 decision_engine/             # 决策引擎
│   ├── 📄 __init__.py
│   ├── 📄 deepseek_analyzer.py     # LLM决策分析
│   ├── 📄 vlm_analyzer.py          # 视觉语言模型
│   └── 📄 vlm_cache.py             # VLM缓存管理
│
├── 📁 execution_engine/            # 交易执行模块
│   ├── 📄 __init__.py
│   └── 📄 okx_trader.py            # OKX交易所接口
│
├── 📁 models/                      # 训练好的模型文件
│   ├── 📄 final_model_btc-*.joblib # 量化模型文件
│   └── 📁 risk_model/              # 风险模型
│
├── 📁 results/                     # 回测和优化结果
│   ├── 📄 best_params_*.json       # 最佳参数
│   ├── 📄 final_evaluation_*.json  # 评估报告
│   ├── 📄 *.png                    # 图表文件
│   └── 📁 risk_model/              # 风险模型结果
│
├── 📁 cache/                       # 缓存文件
│   ├── 📄 vlm_kline_cache.json     # VLM缓存
│   ├── 📄 vlm_tweet_cache.json     # 推文缓存
│   └── 📄 *.png                    # 生成的K线图
│
├── 📁 logs/                        # 日志文件
│   └── 📄 *.log                    # 运行日志
│
├── 📁 common/                      # 公共工具
│   └── 📄 utils.py                 # 通用工具函数
│
├── 📄 dashboard.py                 # 仪表板应用
├── 📄 run_dashboard.py             # 仪表板启动脚本
└── 📄 performance_analysis.py      # 性能分析工具
```

## 🔧 核心模块说明

### btc_predictor/ - 量化预测模块
- **model.py**: 定义机器学习模型架构
- **predict.py**: 实时预测引擎，生成交易信号
- **kline_plot.py**: 生成包含技术指标的K线图
- **backtest.py**: 历史数据回测功能
- **optimize.py**: 使用Optuna进行超参数优化

### decision_engine/ - 决策引擎
- **deepseek_analyzer.py**: 使用DeepSeek LLM进行综合决策分析
- **vlm_analyzer.py**: 视觉语言模型，分析K线图
- **vlm_cache.py**: VLM分析结果缓存管理

### execution_engine/ - 交易执行
- **okx_trader.py**: OKX交易所API接口，支持期货交易

### data_ingestion/ - 数据获取
- **news_feeds.py**: 从CoinDesk获取市场新闻

## 📊 数据流

```
市场数据 → 量化模型 → 预测信号
    ↓
新闻数据 → VLM分析 → 图表分析
    ↓
综合决策 ← LLM分析 ← 所有信息
    ↓
交易执行 → OKX交易所
```

## 🔐 安全配置

- **环境变量**: 所有API密钥通过 `.env` 文件管理
- **配置文件**: `config.py` 包含系统设置，不包含敏感信息
- **Git忽略**: `.gitignore` 确保敏感文件不被提交

## 🚀 部署说明

1. **开发环境**: 使用 `DEMO_MODE=true` 进行测试
2. **生产环境**: 设置 `DEMO_MODE=false` 进行实盘交易
3. **监控**: 通过日志文件和决策报告监控系统运行状态

## 📈 扩展指南

- **新交易所**: 在 `execution_engine/` 中添加新的交易接口
- **新数据源**: 在 `data_ingestion/` 中添加新的数据获取模块
- **新模型**: 在 `btc_predictor/` 中添加新的机器学习模型
- **新指标**: 在 `btc_predictor/kline_plot.py` 中添加新的技术指标 