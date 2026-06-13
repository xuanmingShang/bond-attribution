# Bond PnL Attribution System

> 本文用于帮助理解系统设计。
> 目标读者可能熟悉投资研究，但不一定熟悉债券、收益率曲线和债券阶梯策略。

---

## 1. 系统面向的用户问题

本项目不是只计算一只债券价格，而是把债券投资分析拆成几个用户可以理解和操作的问题：

1. 市场利率环境发生了什么变化？
2. 一只债券为什么赚钱或亏钱？
3. 盈亏来自利率变化、时间流逝、曲线形变，还是现金流？
4. 如果把多只不同期限债券组成债券阶梯（Bond Ladder），组合价值和现金流会如何变化？
5. Classic、Withdrawal、Immunized 三类债券阶梯策略分别对应什么投资目标？

系统设计围绕这些问题展开：先构建收益率曲线数据，再建立债券定价能力，然后做单债归因，最后扩展到组合层面的 Ladder 策略回测，并通过 Dashboard 做现场交互展示。

---

## 2. 整体系统流程

系统主流程可以理解为一条数据流：

```text
FRED Treasury yield data
  -> Yield curve construction
  -> Bond pricing and risk measures
  -> Single-bond PnL attribution
  -> PCA yield-curve factor attribution
  -> Bond ladder strategy backtest
  -> Streamlit dashboard presentation
```

其中每一层都有明确职责：

- 数据层：读取 1M 到 30Y 的美国国债收益率数据，并处理交易日、缺失值和缓存。
- 曲线层：把离散期限的收益率转换成可插值的收益率曲线。
- 定价层：用债券未来现金流折现得到价格，并计算久期、DV01、凸度等风险指标。
- 归因层：解释债券 PnL 的来源。
- 策略层：把单债分析扩展为多期限债券组合和投资策略。
- 展示层：把模型能力组织成 Field、Attribution、Ladder 三个用户入口。

一个极简贯穿案例是：选定 `2023-06-01` 到 `2024-06-30` 的收益率数据，对一只 10 年期、3.5% 票息债券做归因，再用同一段利率环境运行不同 Ladder 策略。这个案例只用于说明系统运行路径，重点仍然是系统设计。

---

## 3. 数据与定价基础

### 3.1 收益率曲线数据

系统使用 FRED 的美国国债 Constant Maturity Treasury 数据，覆盖：

```text
1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y
```

这些期限点组成每天的一条收益率曲线。系统会把数据缓存到本地，避免重复请求 API。

收益率曲线的作用是给任意期限债券提供折现率。例如一只 6.5 年后到期的债券，市场没有直接给出 6.5Y 利率，系统会通过插值得到对应折现率。

### 3.2 债券定价模型

系统把债券看成一组未来现金流：

- 每期票息（Coupon）
- 到期本金（Principal）

债券价格等于这些未来现金流按收益率曲线折现后的现值。系统同时区分：

- 全价（Dirty Price）：包含已经累计但尚未支付的利息。
- 净价（Clean Price）：全价减去应计利息，更接近市场报价口径。

这些定价能力是后续归因和 Ladder 回测的基础。没有统一的定价模型，系统就无法比较不同日期、不同期限、不同策略下的价值变化。

---

## 4. 单债 PnL 归因：解释为什么赚钱或亏钱

单债归因回答的问题是：给定一只债券和一段利率变化，实际盈亏来自哪里？

核心恒等式是：

```text
Actual PnL = Market Impact + Time Impact
```

系统进一步把 Market Impact 拆成：

```text
Market Impact = Duration + Convexity + Curve Reshape
```

把 Time Impact 拆成：

```text
Time Impact = Accrual + Rolldown - Funding
```

其中：

- Duration：利率变化带来的主要价格影响。
- Convexity：利率变化较大时，对 Duration 线性近似的修正。
- Curve Reshape：收益率曲线不是整体平移时，由曲线形状变化带来的影响。
- Accrual：持有债券期间自然累积的票息收入。
- Rolldown：债券剩余期限缩短后，在收益率曲线上自然滑动带来的价格变化。
- Funding：持有头寸所需的融资成本。
- Residual：模型近似、非线性和舍入带来的剩余项。

系统设计上，归因不是直接套公式，而是通过三个价格状态来保持解释清晰：

```text
Step 0: 昨日债券 + 昨日曲线
Step 1: 昨日债券 + 今日曲线
Step 2: 今日债券 + 今日曲线
```

Step 0 到 Step 1 解释市场变化；Step 1 到 Step 2 解释时间流逝和现金流。这样用户可以看到一只债券的 PnL 到底是被利率冲击主导，还是被持有收益和期限滑动主导。

---

## 5. PCA 归因：用曲线因子解释市场变化

PCA 归因回答的问题是：收益率曲线的变化能否用少数几个主要因子解释？

系统把每日收益率曲线变化分解成三个主要方向：

```text
Yield Curve Change ~= PC1 Level + PC2 Slope + PC3 Curvature
```

直观解释：

- PC1 Level：整条收益率曲线整体上移或下移。
- PC2 Slope：短端和长端相对变化，曲线变陡或变平。
- PC3 Curvature：中段相对短端和长端的弯曲变化。

PCA 不是替代传统归因，而是提供另一种观察市场冲击的角度。传统归因使用 Duration、Convexity、Curve Reshape 这类经济含义明确的分解；PCA 归因使用 Level、Slope、Curvature 这类从历史数据中学习出来的曲线因子。

系统实现上，PCA 归因也采用完整重定价：每次只施加一个 PC 对应的曲线冲击，然后重新计算债券价格。这样可以避免只用单一 DV01 近似导致的解释偏差。

---

## 6. Bond Ladder：系统的重点投资策略能力

Ladder 是本系统最接近真实投资决策的部分。单债归因解释的是“一只债券为什么涨跌”，而 Ladder 回测解释的是“用户如何组织一组债券来满足投资目标”。

债券阶梯（Bond Ladder）是一种把不同到期期限的债券组合在一起的策略。系统不是只看组合最终收益，而是持续记录：

- 组合价值
- 现金余额
- 债券市值
- 持仓明细
- 票息现金流
- 本金到期现金流
- 策略交易日志
- 组合归因
- 目标久期匹配情况

### 6.1 Ladder 的共同建模基础

三种 Ladder 策略共享同一套组合机制：

1. 在初始日期构建多个期限的合成平价债券（Synthetic Par Bond）。
2. 每个 rung 表示一个目标期限，例如 1Y、2Y、3Y、4Y、5Y。
3. 每个交易日重新估值组合，记录现金、债券市值和组合总价值。
4. 票息进入现金账户，到期本金进入现金账户。
5. 组合 PnL 被拆成 Income、Rolldown、Rate Movement、Residual。

这里的 Synthetic Par Bond 是系统设计中的重要抽象：系统根据当天收益率曲线反推出一个接近平价发行的债券，使不同期限的 rung 可以在同一规则下构建和比较。

### 6.2 Classic Roll：基础滚动债券阶梯

用户场景：用户想维持一个简单、规则化、持有到期的债券阶梯。

系统设计：

- 默认使用 `1Y, 2Y, 3Y, 4Y, 5Y` 五个 rung。
- 初始资金在这些 rung 之间等权配置。
- 债券不到期时不主动卖出。
- 某个 rung 到期后，本金滚入最长端 rung，重新买入新的长期债券。
- 票息保留在现金账户中。

这个策略用于展示最基础的 Ladder 思想：通过错开到期日，使组合不是一次性暴露在单一到期点上，同时通过到期再投资维持阶梯结构。

### 6.3 Withdrawal：带提款需求的现金流策略

用户场景：用户不是只关心组合收益，还需要定期从组合中取出现金。

系统设计：

- 用户设置提款金额、提款频率和首次提款日期。
- 系统根据交易日生成提款计划。
- 票息和到期本金先进入现金账户。
- 提款只从现金账户支付。
- 如果现金不足，系统记录 Shortfall，而不是强行卖出未到期债券。

这种设计刻意保留了现金流缺口，而不是用卖债来掩盖问题。它回答的是：在当前 Ladder 和利率环境下，组合自然产生的现金流能否支持提款需求？

关键输出包括：

- Withdrawal Due：应提款金额。
- Withdrawal Paid：实际支付金额。
- Withdrawal Shortfall：现金不足产生的缺口。
- Cumulative Withdrawn：累计已提款金额。
- Coupon / Principal Cashflow：现金来源。

Withdrawal 策略是面向用户设计中最直观的一部分，因为它把债券组合从“收益率分析”推进到“现金流能否满足需求”的问题。

### 6.4 Immunized：面向目标风险或负债的配置策略

用户场景：用户有一个目标期限、目标久期或未来负债，希望组合的利率风险与目标更接近。

系统设计：

- 使用更宽的候选 rung：`1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y`。
- 每个候选 rung 都先生成对应的合成平价债券。
- 系统计算各 rung 的久期、DV01 和凸度。
- 通过 long-only 权重求解，让组合久期尽量匹配目标久期。
- 如果选择 Liability 模式，系统会根据负债日期和负债金额计算对应的目标暴露。

Immunized Ladder 的核心思想不是追求最高收益，而是控制组合对利率变化的敏感度，使组合更接近用户指定的风险目标或负债期限。

关键输出包括：

- Portfolio Duration：组合当前久期。
- Target Duration：目标久期。
- Duration Gap：组合久期与目标久期的差距。
- Funding Ratio：资产价值相对负债现值的覆盖情况。
- Rung Analytics：每个候选 rung 的权重、久期、DV01、凸度。
- Target Match Log：每次建仓或再投资时的目标匹配记录。

### 6.5 三种 Ladder 策略的系统定位

```text
Classic Roll
  -> 维持基础阶梯结构，观察组合收益和再投资过程。

Withdrawal
  -> 在阶梯组合上叠加现金流支出需求，观察提款覆盖和缺口。

Immunized
  -> 在阶梯组合上叠加目标久期或负债约束，观察风险匹配效果。
```

因此，Ladder 模块不是一个附加图表功能，而是系统从“债券解释工具”扩展到“投资策略分析工具”的关键部分。

---

## 7. Dashboard：最终交互展示层

Dashboard 是系统最终面向用户和评委老师展示的入口。它把底层模型组织成三个清晰的分析区域：

- Field：观察收益率曲线数据本身，包括曲面和指定日期曲线快照。
- Attribution：运行单债 Core 归因、PCA 归因和两种归因结果对比。
- Ladder：运行 Classic Roll、Withdrawal、Immunized 三类债券阶梯策略。

Dashboard 的设计重点是让用户在每个分析场景内直接设置参数并点击运行。它不要求用户理解底层代码，也不需要在命令行中手动拼接多个模块。

---

## 8. 代码索引

| 系统层 | 相关文件 | 极简说明 |
| --- | --- | --- |
| 数据与曲线 | `bond_pnl/yield_curve.py` | FRED 数据读取、缓存、收益率曲线插值 |
| 债券定价 | `bond_pnl/bond.py` | 债券现金流、Dirty/Clean Price、DV01、久期、凸度 |
| 单债归因 | `bond_pnl/attribution.py` | Campisi-style Daily PnL Attribution |
| PCA 归因 | `bond_pnl/pca.py` | 收益率曲线 PCA 与基于 PC 的债券 PnL 归因 |
| Ladder 策略 | `bond_pnl/ladder.py` | Classic、Withdrawal、Immunized 债券阶梯回测 |
| 交互展示 | `dashboard.py` | Streamlit Dashboard，提供 Field / Attribution / Ladder 入口 |
| CLI 分析 | `main.py` | 命令行运行核心分析、PCA 和 Ladder |
| 跨区间比较 | `run_multi_year.py` | 多市场环境下的归因和 Ladder 对比 |
| 测试 | `tests/` | 定价、归因、PCA、Ladder 和集成测试 |

---

## 9. 系统特点总结

先用收益率曲线和债券定价建立统一计算基础，再用 Core 和 PCA 归因解释单只债券的盈亏来源，最后用 Ladder 把分析扩展到组合投资策略。尤其是 Ladder 模块，使系统能够覆盖基础滚动持有、现金流提款和目标久期匹配三类投资场景。

因此，本系统可以被理解为一个完整的债券投资分析系统：它既能解释市场变化如何影响单只债券，也能展示不同债券阶梯策略如何服务于不同用户目标。
