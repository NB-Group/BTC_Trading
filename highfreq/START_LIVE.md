# 🚀 实盘交易快速启动

## 直接运行（默认配置）

```bash
python highfreq/run_live_trading.py
```

**默认参数：**
- 最大资金：$500
- 运行时长：60分钟
- 检查间隔：1秒
- 订单类型：Maker（限价单）

---

## 自定义参数

### 运行2小时，使用500美元

```bash
python highfreq/run_live_trading.py --duration 120
```

### 使用Taker订单（立即成交）

```bash
python highfreq/run_live_trading.py --use-taker
```

### 自定义资金和时长

```bash
python highfreq/run_live_trading.py --capital 300 --duration 90
```

---

## ⚠️ 重要提示

1. **确认API Key已配置**
   - 检查 `.env` 文件中的 OKX API Key
   - 确保有交易权限

2. **确认账户余额**
   - 建议至少 $500 USDT
   - 程序会自动计算最优杠杆

3. **运行前会要求确认**
   - 输入 `yes` 开始实盘交易
   - 输入其他内容会取消

4. **停止程序**
   - 按 `Ctrl+C` 停止
   - 程序会自动平仓（如果有仓位）

---

## 📊 实时监控

程序会输出详细的交易日志：

```
[HF-LIVE] 开仓: LONG, 5张, 杠杆=5x
[HF-LIVE] 开仓成功: LONG, 入场价=100000.00
[HF-LIVE] 触发止盈: LONG, 收益=0.1000%
[HF-LIVE] 平仓: LONG, 入场=100000.00, 出场=100100.00, 收益=0.1000%
```

---

## 🛑 紧急停止

如果遇到问题，立即按 `Ctrl+C`：
- 程序会尝试平仓
- 等待平仓完成后再退出
- 如果无法自动平仓，请手动在交易所平仓

---

**准备好了吗？运行命令开始交易！** 🚀



