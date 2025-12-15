# 📊 高频策略原理详解

## 🎯 策略概述

这是一个基于**订单簿微观结构**的高频交易策略，通过分析订单簿的**不平衡性**和**价格偏移**来捕捉短期价格波动，实现快速进出场。

**核心思想**：订单簿的不平衡反映了买卖双方的力量对比，这种不平衡往往预示着短期价格移动方向。

---

## 🔍 核心指标

### 1. **Order Book Imbalance (OBI) - 订单簿不平衡度**

**定义**：
```
buy_depth = 前3档买单总量
sell_depth = 前3档卖单总量
ratio = buy_depth / sell_depth
```

**含义**：
- `ratio > 1`：买单力量强，可能上涨
- `ratio < 1`：卖单力量强，可能下跌
- `ratio ≈ 1`：买卖平衡

**代码位置**：`highfreq/hf_orderbook.py:compute_obi()`

---

### 2. **Microprice - 微观价格**

**定义**：
```
microprice = (ask × bid_depth + bid × ask_depth) / (bid_depth + ask_depth)
```

**含义**：
- 这是一个**加权平均价格**，考虑了订单簿的深度
- 如果买单深度大，microprice 会偏向 ask（卖价）
- 如果卖单深度大，microprice 会偏向 bid（买价）

**示例**：
```
bid = 90000, ask = 90010, mid = 90005
buy_depth = 10 BTC, sell_depth = 5 BTC

microprice = (90010 × 10 + 90000 × 5) / (10 + 5) = 90006.67
```

---

### 3. **Microprice Bias - 微观价格偏移**

**定义**：
```
micro_bias = (microprice - mid) / mid
```

**含义**：
- `micro_bias > 0`：microprice 高于中间价，**买单力量强**，可能上涨 → **买入信号**
- `micro_bias < 0`：microprice 低于中间价，**卖单力量强**，可能下跌 → **卖出信号**

**当前阈值**：
- `bias_long = 4e-7`（0.0000004）：买入阈值
- `bias_short = -4e-7`（-0.0000004）：卖出阈值

**代码位置**：`highfreq/hf_orderbook.py:_enrich_micro_ofi_features()`

---

### 4. **OFI (Order Flow Imbalance) - 订单流不平衡**

**定义**：
```
ofi_raw = Δbuy_depth - Δsell_depth
ofi_ema = EMA(ofi_raw, span=8)
```

**含义**：
- `ofi_raw`：买单深度变化 - 卖单深度变化
- `ofi_ema`：平滑后的订单流不平衡（指数移动平均）
- `ofi_ema > 0`：买单在增加，可能上涨
- `ofi_ema < 0`：卖单在增加，可能下跌

**当前设置**：`ofi_long = 0.0, ofi_short = 0.0`（暂时关闭OFI过滤，只用micro_bias）

**代码位置**：`highfreq/hf_orderbook.py:_enrich_micro_ofi_features()`

---

## 🚦 信号生成逻辑

### 步骤1：数据积累

策略需要**至少8条订单簿数据**才能计算特征（OFI需要历史数据）：

```python
if len(orderbook_history) < 8:
    return None  # 继续积累数据
```

**当前状态**：每1秒获取一次订单簿，8秒后开始计算信号。

---

### 步骤2：过滤条件

在生成信号前，先检查市场条件：

1. **深度过滤**：
   ```
   total_depth = buy_depth + sell_depth
   if total_depth < 3.0:  # 最小深度要求
       return None  # 流动性不足，不交易
   ```

2. **点差过滤**：
   ```
   spread_bps = (ask - bid) / mid × 10000
   if spread_bps > 1.5:  # 点差过大
       return None  # 交易成本太高，不交易
   ```

---

### 步骤3：信号判断

**买入信号（BUY）**：
```python
if micro_bias > 4e-7:  # 买单力量强
    if ofi_long == 0.0 or ofi_ema > ofi_long:  # OFI确认（当前关闭）
        signal = 'BUY'
```

**卖出信号（SELL）**：
```python
if micro_bias < -4e-7:  # 卖单力量强
    if ofi_short == 0.0 or ofi_ema < ofi_short:  # OFI确认（当前关闭）
        signal = 'SELL'
```

**代码位置**：`highfreq/hf_live_trader.py:_generate_signal_from_obi()`

---

### 步骤4：信号确认机制

**目的**：减少假信号，提高信号质量

**逻辑**：
```python
# 需要连续2个相同信号才开仓
if current_signal == last_signal:
    signal_confirmation_count += 1
    if signal_confirmation_count >= 2:  # 连续2次
        return current_signal  # 确认信号，可以开仓
else:
    signal_confirmation_count = 1  # 重置计数
```

**示例**：
```
时刻1: micro_bias = 3e-7 → 未达到阈值 → HOLD
时刻2: micro_bias = 5e-7 → BUY信号（第1次）→ 确认中
时刻3: micro_bias = 6e-7 → BUY信号（第2次）→ 确认！开仓
```

**代码位置**：`highfreq/hf_live_trader.py:_generate_signal_from_obi()`

---

## 💰 交易执行

### 开仓

**条件**：
1. 信号确认（连续2次相同信号）
2. 无持仓
3. 不在冷却期（上次交易后15秒内不开新仓）

**执行**：
```python
# 计算仓位：只开最小仓位（1张）
contracts = 1  # 固定1张
leverage = 5x  # 自动计算的最优杠杆

# 使用Taker订单（立即成交）
execute_decision({
    'decision': 'LONG' or 'SHORT',
    'suggested_trade_size': 1,
    'leverage': 5
})
```

**代码位置**：`highfreq/hf_live_trader.py:_open_position()`

---

### 平仓条件

策略有三种平仓方式：

#### 1. **止盈（Take Profit）**
```python
if move_pct >= 0.12%:  # 收益达到0.12%
    close_position()  # 平仓
```

**原因**：覆盖交易成本（手续费约0.10%），确保盈利。

#### 2. **止损（Stop Loss）**
```python
if move_pct <= -0.08%:  # 亏损达到0.08%
    close_position()  # 平仓
```

**原因**：快速止损，避免大亏。

#### 3. **时间止损（Time Stop）**
```python
if hold_time >= 120秒:  # 持仓超过120秒
    close_position()  # 强制平仓
```

**原因**：如果价格在120秒内没有朝预期方向移动，说明信号可能失效。

**代码位置**：`highfreq/hf_live_trader.py:_check_exit_conditions()`

---

## 🛡️ 风险控制

### 1. **资金管理**

- **最大资金**：$300（可配置）
- **仓位大小**：固定1张（最小开仓）
- **杠杆**：自动计算（基于最小开仓价格和可用资金）

**计算逻辑**：
```python
最小开仓价格 = 1张 × $100 = $100
可用资金 = $300
最优杠杆 = 300 / 100 = 3x → 使用5x（OKX最小要求）
所需保证金 = 100 / 5 = $20
```

**代码位置**：`highfreq/hf_live_trader.py:_calculate_position_size()`

---

### 2. **冷却期（Cooldown）**

**目的**：避免过度交易，减少手续费成本

**逻辑**：
```python
cooldown_sec = 15  # 上次交易后15秒内不开新仓
```

**代码位置**：`highfreq/hf_live_trader.py:run_once()`

---

### 3. **过滤条件**

- **深度过滤**：`min_depth_total = 3.0`（确保流动性）
- **点差过滤**：`max_spread_bps = 1.5`（避免高成本）

---

## 📈 策略流程

### 完整执行流程

```
1. 初始化
   ├─ 加载市场数据
   ├─ 获取合约信息（面值、最小开仓）
   ├─ 计算最优杠杆
   └─ 记录初始余额

2. 主循环（每1秒）
   ├─ 获取订单簿
   ├─ 计算OBI（buy_depth, sell_depth, ratio）
   ├─ 添加到历史记录
   │
   ├─ 数据积累检查
   │   └─ 如果 < 8条 → 继续积累
   │
   ├─ 计算特征
   │   ├─ microprice = (ask×bid_depth + bid×ask_depth) / (bid_depth+ask_depth)
   │   ├─ micro_bias = (microprice - mid) / mid
   │   └─ ofi_ema = EMA(Δbuy_depth - Δsell_depth, span=8)
   │
   ├─ 过滤检查
   │   ├─ 深度过滤：total_depth >= 3.0
   │   └─ 点差过滤：spread_bps <= 1.5
   │
   ├─ 信号生成
   │   ├─ 如果 micro_bias > 4e-7 → BUY信号
   │   └─ 如果 micro_bias < -4e-7 → SELL信号
   │
   ├─ 信号确认
   │   └─ 需要连续2次相同信号才开仓
   │
   ├─ 开仓检查
   │   ├─ 无持仓？
   │   ├─ 信号确认？
   │   └─ 不在冷却期？
   │       └─ 开仓（1张，5x杠杆）
   │
   └─ 平仓检查（如果有持仓）
       ├─ 止盈：move_pct >= 0.12%？
       ├─ 止损：move_pct <= -0.08%？
       └─ 时间止损：hold_time >= 120秒？
           └─ 平仓
```

---

## 📊 当前参数配置

### 信号参数
- `bias_long = 4e-7`：买入阈值
- `bias_short = -4e-7`：卖出阈值
- `signal_confirmation_threshold = 2`：需要连续2次信号确认

### 风险控制
- `tp_pct = 0.12%`：止盈
- `sl_pct = 0.08%`：止损
- `time_stop_sec = 120`：时间止损（2分钟）

### 过滤条件
- `min_depth_total = 3.0`：最小深度
- `max_spread_bps = 1.5`：最大点差
- `cooldown_sec = 15`：冷却期

### 交易设置
- `use_taker = True`：使用Taker订单（立即成交）
- `max_capital_usd = 300`：最大资金
- `contracts = 1`：固定1张

---

## 🎯 策略优势

1. **基于订单簿微观结构**：捕捉市场微观不平衡，信号质量高
2. **快速进出场**：持仓时间短（通常<120秒），降低风险
3. **多重过滤**：深度、点差、信号确认，减少假信号
4. **严格风控**：止盈止损+时间止损，保护资金

---

## ⚠️ 策略风险

1. **手续费成本高**：频繁交易，手续费累积
2. **滑点影响**：快速进出场，可能遇到滑点
3. **市场环境变化**：策略可能在不同市场环境下失效
4. **API限流**：高频请求可能触发API限流

---

## 📝 总结

这是一个**基于订单簿微观结构的高频交易策略**，通过分析：
- **订单簿不平衡**（OBI）
- **微观价格偏移**（micro_bias）
- **订单流变化**（OFI）

来捕捉短期价格波动，实现快速进出场。策略采用**多重过滤**和**信号确认机制**来提高信号质量，并通过**严格的止盈止损**来控制风险。

**核心公式**：
```
microprice = (ask × bid_depth + bid × ask_depth) / (bid_depth + ask_depth)
micro_bias = (microprice - mid) / mid

信号：micro_bias > 4e-7 → BUY
      micro_bias < -4e-7 → SELL
```

