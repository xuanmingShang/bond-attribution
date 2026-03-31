# Bond PnL Attribution System — 从零开始的完整解释文档

> **目标读者**：没有金融背景，但有机器学习代码经验的工程学生。
> 本文从最基础的金融概念讲起，逐步深入到项目的每一个关键实现细节，帮助你理解"为什么这么写"和"怎么手动验证"。

---

## 目录

1. [基础金融概念](#1-基础金融概念)
2. [债券定价原理](#2-债券定价原理)
3. [收益率曲线](#3-收益率曲线)
4. [PnL 归因框架（Campisi）](#4-pnl-归因框架campisi)
5. [PCA 分析与归因](#5-pca-分析与归因)
6. [债券阶梯回测](#6-债券阶梯回测)
7. [代码模块详解](#7-代码模块详解)
8. [手动验证指南](#8-手动验证指南)
9. [关键公式速查表](#9-关键公式速查表)

---

## 1. 基础金融概念

### 1.1 什么是债券？

**类比**：你借钱给政府，政府承诺按约定时间还本付息。

- **面值（Face Value / Par）**：到期时偿还的本金，通常是 $100
- **票息率（Coupon Rate）**：每年支付的利息占面值的百分比
- **票息频率（Frequency）**：一年支付几次利息（美国国债通常 2 次，即半年付息）
- **到期日（Maturity Date）**：本金偿还的日期
- **发行日（Issue Date）**：债券发行的日期

**举例**：一张面值 $100、票息率 4%、半年付息、5 年到期的债券，意味着：
- 每半年收到 $100 × 4% ÷ 2 = **$2 的利息**
- 5 年内共收 10 次利息
- 第 10 次额外拿回 **$100 本金**

### 1.2 什么是 PnL（盈亏）？

$$
\text{PnL} = V_t - V_0
$$

- $V_0$ = 成本（上一个计算时点的价值）
- $V_t$ = 当前价值
- PnL > 0 → 赚钱；PnL < 0 → 亏钱

### 1.3 什么是收益率（Yield）？

收益率是市场参与者对持有该债券所要求的回报率。

**关键反向关系**：

$$
\text{利率上升} \Rightarrow \text{债券价格下降}
$$

**直觉**：如果市场新发行的债券利率更高（如 5%），而你手中持有的旧债券只有 4%，那么旧债券的吸引力下降，价格自然下跌。

### 1.4 什么是日计利息（Accrued Interest）？

两次票息支付之间的某一天，虽然还没到支付日，但利息在"积累"中。

**类比**：你的工资是月结发放，但到月中的时候，你已经赚了半个月的工资。

$$
\text{应计利息} = \frac{\text{距上次付息天数}}{\text{付息周期总天数}} \times \text{每期票息}
$$

### 1.5 净价（Clean Price）与全价（Dirty Price）

- **全价（Dirty Price）** = 真实交易价格 = 债券所有未来现金流的现值
- **净价（Clean Price）** = 全价 − 应计利息
- 市场报价通常是**净价**（避免锯齿形波动），但实际结算用**全价**

$$
\text{Dirty} = \text{Clean} + \text{Accrued Interest}
$$

### 1.6 什么是 DV01？

DV01 = Dollar Value of 01 = 利率变动 1 个基点（0.01%）时，债券价格变化多少美元。

$$
\text{DV01} = \frac{P(r - 1\text{bp}) - P(r + 1\text{bp})}{2}
$$

**直觉**：DV01 就像机器学习中的"梯度"——它告诉你价格对利率的敏感度。
- DV01 = 0.05 意味着利率每上升 1bp，价格下跌约 $0.05
- 5 年期债券的 DV01 约 0.04–0.05
- 30 年期债券的 DV01 约 0.15–0.20（更敏感）

### 1.7 什么是凸度（Convexity）？

如果说 DV01/久期是一阶导数（梯度），凸度就是**二阶导数**（海森矩阵）。

$$
\text{Convexity} = \frac{P(r+\Delta r) + P(r-\Delta r) - 2P(r)}{(\Delta r)^2}
$$

凸度的作用：当利率大幅变动时，仅用 DV01 线性近似不够精确，凸度提供修正。

**类比 ML**：
- DV01 ≈ 梯度（一阶泰勒展开）
- Convexity ≈ 海森矩阵（二阶泰勒展开）
- Residual ≈ 高阶截断误差

---

## 2. 债券定价原理

### 2.1 现金流折现（DCF）

债券的价格等于所有未来现金流的**现值**之和：

$$
P = \sum_{i=1}^{n} CF_i \times e^{-r(t_i) \times t_i}
$$

其中：
- $CF_i$ = 第 $i$ 期现金流（中间期为票息，最后一期为 票息 + 本金）
- $r(t_i)$ = 期限 $t_i$ 对应的即期收益率（从收益率曲线插值得到）
- $t_i$ = 距离第 $i$ 期现金流的年份数

**代码对应**：`bond.py → dirty_price()`

```python
def dirty_price(self, settle, curve):
    pv = 0.0
    for dt, cf in self.cashflow_schedule(settle):
        t = self._dcf(settle, dt)  # 计算年份差
        if t <= 0: continue
        pv += cf * np.exp(-curve.rate(t) * t)  # 连续复利折现
    return pv
```

### 2.2 连续复利 vs. 离散复利

本项目使用**连续复利**（continuous compounding）：

$$
\text{折现因子} = e^{-r \times t}
$$

而非教科书常见的离散复利 $\frac{1}{(1+r)^t}$。两者在短期限差异很小，但连续复利数学上更方便（指数函数的链式法则更简单）。

### 2.3 年分数计算（Day Count）

"半年"到底是多少天？需要一个约定。

本项目使用**简化的 ACT/ACT**：

$$
\text{年份差} = \frac{\text{实际天数}}{365.25}
$$

`365.25` 是因为每4年有一个闰年。这是简化版——真正的 ISDA ACT/ACT 更复杂，但对于教学项目差异可忽略。

**代码对应**：`bond.py → _act_act()`

---

## 3. 收益率曲线

### 3.1 什么是收益率曲线？

收益率曲线描述了**不同期限**的利率水平，横轴是期限（1个月 ~ 30年），纵轴是年化收益率。

```
    利率
    ^
5%  |        ╱─────────
    |      ╱
4%  |    ╱
    |  ╱
3%  |╱
    └──────────────────> 期限
    1M 3M 6M 1Y 2Y 3Y 5Y 7Y 10Y 20Y 30Y
```

- **正常曲线（Normal）**：长期利率 > 短期利率（常见）
- **倒挂曲线（Inverted）**：短期利率 > 长期利率（衰退信号）
- **平坦曲线（Flat）**：长短期利率接近

### 3.2 FRED 数据源

我们使用美联储经济数据库（FRED）的 **CMT（Constant Maturity Treasury）** 系列：

| 期限 | FRED 代码 | 年份值 |
|------|-----------|--------|
| 1月  | DGS1MO    | 0.083  |
| 3月  | DGS3MO    | 0.250  |
| 6月  | DGS6MO    | 0.500  |
| 1年  | DGS1      | 1.000  |
| 2年  | DGS2      | 2.000  |
| 3年  | DGS3      | 3.000  |
| 5年  | DGS5      | 5.000  |
| 7年  | DGS7      | 7.000  |
| 10年 | DGS10     | 10.000 |
| 20年 | DGS20     | 20.000 |
| 30年 | DGS30     | 30.000 |

**重要建模假设**：CMT 利率本质上是"面值收益率"（par yield），而我们在定价时**将其当作零息即期利率（zero-coupon spot rate）** 使用。这是一种简化——真正的做法需要 bootstrap 来提取零息利率。但对于教学项目，该近似在数值上是可行的，且已在代码中显式声明。

### 3.3 收益率插值

对于期限不在标准网格上的点（如 4.7 年），我们用**三次样条插值**：

```python
self._spline = CubicSpline(tenors, yields, bc_type="natural")
```

**类比 ML**：这就像用样条函数做连续的特征插值，而不是只取离散的网格点。

**代码对应**：`yield_curve.py → YieldCurve.rate()`

---

## 4. PnL 归因框架（Campisi）

### 4.1 核心恒等式

这是整个项目最核心的公式：

$$
\text{Actual PnL} = \text{Market Impact} + \text{Time Impact}
$$

**这个等式是精确的，不是近似**。所有误差（Residual）来自于进一步分解 Market Impact 和 Time Impact 时的近似。

### 4.2 三步分解法

想象从"昨天"到"今天"的变化分两步走：

| 步骤 | 债券状态 | 收益率曲线 | 含义 |
|------|---------|-----------|------|
| Step 0 | T-1 时刻 | 昨天的曲线 | 起点 |
| Step 1 | T-1 时刻 | 今天的曲线 | 仅曲线变了 → **Market Impact** |
| Step 2 | T 时刻   | 今天的曲线 | 时间过了一天 → **Time Impact** |

$$
\text{Market} = PV(T{-}1, \text{Curve}_T) - PV(T{-}1, \text{Curve}_{T{-}1})
$$

$$
\text{Time} = PV(T, \text{Curve}_T) - PV(T{-}1, \text{Curve}_T) + \text{CashFlow} - \text{Funding}
$$

**直觉**：
- Market Impact = "如果时间冻结，只有利率变了，价格会怎样变？"
- Time Impact = "如果利率不变，只是时间过了一天，价格会怎样变？"

### 4.3 Market Impact 分解

Market Impact 进一步分解为：

#### a) Duration Effect（久期效应）

$$
\text{Duration} = -\text{DV01} \times \Delta r_{\text{local}} \quad (\text{bp})
$$

- $\Delta r_{\text{local}}$ 是以债券到期期限为中心的**高斯加权**平均利率变化
- 负号反映利率和价格的反向关系

**为什么用"local shift"而不是"parallel shift"？**

"Parallel shift"（平行移动）假设整条曲线等量移动，但现实中不同期限的利率变化量不同。我们用高斯加权让靠近债券期限的利率权重更大：

$$
w_i = \exp\left(-\frac{(t_i - \text{TTM})^2}{2\sigma^2}\right), \quad \sigma = \max(0.3 \times \text{TTM}, 1.0)
$$

**代码对应**：`attribution.py → _local_shift_bp()`

#### b) Convexity Effect（凸度效应）

$$
\text{Convexity} = \frac{1}{2} \times \text{ConvDollar} \times (\Delta r)^2
$$

二阶修正项。利率变动越大，此项越重要。

#### c) Curve Reshape（曲线形变）

$$
\text{Reshape} = \text{Market}_{\text{exact}} - \text{Market}_{\text{local shift}}
$$

曲线并非等量移动，而是有扭曲（twist）/ 蝶式（butterfly）变化。Reshape 捕获了这部分。

#### d) Rate Residual（利率残差）

$$
\text{Rate Residual} = \text{Market} - (\text{Duration} + \text{Convexity} + \text{Reshape})
$$

泰勒展开的截断误差 + 高阶项。如果 DV01/凸度计算正确，此项非常小。

### 4.4 Time Impact 分解

#### a) Accrual（票息累计）

$$
\text{Accrual} = \text{Face} \times \text{Coupon Rate} \times \Delta t
$$

每天"赚到"的利息，无论利率怎么变。

#### b) Rolldown（期限滑坡）

$$
\text{Rolldown} = \text{Clean Price}(T, \text{Curve}_T) - \text{Clean Price}(T{-}1, \text{Curve}_T)
$$

**直觉**：时间过了一天，债券的剩余期限缩短了，在相同曲线上重新定价。如果曲线是上倾的（正常形态），期限缩短意味着折现率降低，价格上升，产生正的 Rolldown。

**注意**：用净价（Clean Price）而非全价，因为全价变化中包含了应计利息的变化，已被 Accrual 项捕获。

#### c) Funding（融资成本）

$$
\text{Funding} = \text{Dirty PV}(T{-}1) \times r_{\text{financing}} \times \Delta t
$$

持有债券需要占用资金（或借钱买债券），所以必须扣除资金成本。融资利率取 3 个月国库券利率。

#### d) Carry（持有收益）

$$
\text{Carry} = \text{Accrual} + \text{Rolldown} - \text{Funding}
$$

**直觉**：Carry 是"什么都不变"时你每天赚多少。

#### e) Time Residual

$$
\text{Time Residual} = \text{Time Impact} - \text{Carry}
$$

应该非常小（接近零），如果大了说明模型分解有问题。

### 4.5 完整分解图

```
Actual PnL
├── Market Impact
│   ├── Duration Effect        (一阶利率敏感度)
│   ├── Convexity Effect       (二阶利率敏感度)
│   ├── Curve Reshape          (非平行移动)
│   └── Rate Residual          (高阶截断误差)
└── Time Impact
    ├── Accrual                (票息收入)
    ├── Rolldown               (期限滑坡)
    ├── Funding                (融资成本, 负项)
    └── Time Residual          (分解余项)
```

### 4.6 到期处理

当债券在归因期间到期时：
- `dirty_price()` 返回 0（因为所有现金流已经发完）
- 本金偿还放入 `coupon_cf`（包含票息 + 本金）
- Rolldown 使用 `face` 值而非 `clean_price = 0`

---

## 5. PCA 分析与归因

### 5.1 收益率变化的 PCA

**类比 ML**：PCA 对你来说应该很熟悉——把高维数据降到几个主成分。

这里的"高维数据"是每天的收益率变化向量（11 个期限 → 11 维）：

$$
\Delta Y_t = [\Delta r_{1M}, \Delta r_{3M}, \ldots, \Delta r_{30Y}]
$$

PCA 把这些 11 维向量分解为 3 个正交主成分：

| PC  | 方差解释 | 金融解释 | 含义 |
|-----|---------|---------|------|
| PC1 | ~86%    | Level   | 整条曲线同时上下移动 |
| PC2 | ~9%     | Slope   | 短端和长端反向移动（变陡/变平） |
| PC3 | ~1.4%   | Curvature | 中间凸起或凹下（蝶式） |

**观察 loadings 判断**：
- PC1 loadings 全部同号且幅度接近 → Level
- PC2 loadings 短端正、长端负（或反过来）→ Slope
- PC3 loadings 中间与两端反号 → Curvature

### 5.2 PCA 归因 vs. 传统归因

**传统方法**（Core）：
- 用单一标量 $\Delta r$（DV01 × 利率变化）解释 Market Impact
- Reshape 捕获非平行移动

**PCA 方法**：
- 把 Market Impact 分解为 PC1（Level）、PC2（Slope）、PC3（Curvature）各自的贡献
- 每个 PC 通过**完整重定价**（不是 DV01 近似）计算，确保因子一致性

### 5.3 归因计算流程

对于每一天：
1. 计算日收益率变化 $\Delta Y$
2. 去均值化：$\Delta Y_{\text{centered}} = \Delta Y - \mu$
3. 投影：$\text{score}_k = \text{loading}_k \cdot \Delta Y_{\text{centered}}$
4. 逐层重定价：
   - Mean 层：curve + μ → 重定价 → Mean PnL
   - PC1 层：curve + μ + score₁ × loading₁ → 重定价 → PC1 PnL
   - PC2 层：curve + μ + score₁×l₁ + score₂×l₂ → 重定价 → PC2 PnL
   - ...
5. Residual = Actual − Carry − PC Total

**代码对应**：`pca.py → pca_attribution()`

### 5.4 为什么用全价重定价而不是 DV01？

如果用 DV01（一阶近似）来估算每个 PC 的贡献，**Slope 和 Curvature 因子不会被正确捕获**——因为 DV01 假设整条曲线等量移动（平行移动）。

全价重定价意味着：对于每个 PC 冲击，真正重新计算一次债券价格。虽然慢一些，但保证了因子间的正交性和总量一致性。

---

## 6. 债券阶梯回测

### 6.1 什么是债券阶梯？

**债券阶梯（Bond Ladder）** 是一种简单的投资策略：把资金等分投入不同期限的债券。

```
初始投组（每档 $200,000）：
├── 2 年期 ──▪ 2025 年到期
├── 5 年期 ──▪ 2028 年到期
├── 7 年期 ──▪ 2030 年到期
├── 10 年期 ─▪ 2033 年到期
└── 30 年期 ─▪ 2053 年到期
```

当最短期限的债券到期后收回本金，再投入最长期限的新债券，保持阶梯结构。

### 6.2 合成面值债券（Synthetic Par Bond）

因为我们无法获取真实的国债市场数据（价格、CUSIP 等），所以使用**合成债券**：

对于每个档位（rung），在当前收益率曲线下，通过**二分法**求解票息率 $c$，使得：

$$
\text{Dirty Price}(c) = 100 \quad (\text{即面值})
$$

这就是"面值债券"——发行时价格等于面值，意味着票息率恰好等于市场对该期限的要求回报。

**代码对应**：`ladder.py → _make_bond()`

### 6.3 阶梯归因

对于投资组合的每日 PnL：

$$
\text{Total PnL} = \text{Income} + \text{Rolldown} + \text{Rate Movement} + \text{Residual}
$$

- **Income**（收入）= ΔAI（应计利息变化）+ 收到的票息
- **Rolldown**（期限滑坡）= 用净价在前一天曲线上，期限缩短引起的价值变化
- **Rate Movement**（利率变动）= 新旧曲线下净价的差异
- **Residual**（残差）= 总 PnL 减去以上三项

**我们的 Ladder Residual = 0.00（精确值）**，说明归因是完美的。

### 6.4 再平衡（Rebalance）

每 12 个月执行一次再平衡：
1. 卖出所有持仓（按当前市价）
2. 用全部现金重新等分购买 5 个档位的新面值债券

---

## 7. 代码模块详解

### 7.1 模块关系图

```
main.py                    ← CLI 入口 + 绘图 + 输出
  │
  ├── bond_pnl/
  │   ├── yield_curve.py   ← FRED 数据加载 + 收益率曲线(CubicSpline)
  │   ├── bond.py          ← 债券定义 + 定价 + DV01 + 凸度
  │   ├── attribution.py   ← Campisi 日度 PnL 归因
  │   ├── pca.py           ← PCA 分析 + PCA 归因
  │   └── ladder.py        ← 债券阶梯回测
  │
  └── tests/
      ├── test_bond.py     ← 15 个测试
      ├── test_attribution.py ← 7 个测试
      ├── test_ladder.py   ← 5 个测试
      └── test_pca.py      ← 4 个测试
```

### 7.2 yield_curve.py

**职责**：从 FRED 下载数据、缓存为 CSV、提供插值。

关键类：
- `YieldCurve`：单日曲线快照，支持 `rate(tenor_years)` 插值
- `YieldCurveHistory`：时间序列容器，支持 `[date]` 索引和 `changes()` 方法

```python
# 示例：获取 2024-01-15 的 5 年期利率
curve = yield_curve_history["2024-01-15"]
rate_5y = curve.rate(5.0)   # 返回小数形式，如 0.042
```

### 7.3 bond.py

**职责**：债券规格、现金流调度、定价、风险指标。

核心方法：
- `cashflow_schedule(settle)` — 从结算日之后的所有现金流
- `dirty_price(settle, curve)` — 连续折现定价
- `clean_price(settle, curve)` — 全价 − 应计利息
- `dv01(settle, curve)` — 中心差分法（±1bp）
- `convexity_dollar(settle, curve)` — 二阶差分

**到期后行为**：
- `dirty_price()` → 0（所有现金流已支付）
- `clean_price()` → 0
- `accrued_interest()` → 0

### 7.4 attribution.py

**职责**：Campisi 风格日度归因。

核心函数：
- `compute_daily_attribution()` — 计算一天的完整归因
- `run_attribution()` — 对整个日期范围逐日运行
- `attribution_summary()` — 汇总所有天的归因

**精确性**：Market + Time = Actual 是完全精确的（恒等式），残差仅来自进一步的泰勒分解。

### 7.5 pca.py

**职责**：PCA 分析和基于 PCA 的另一种归因视角。

核心函数：
- `fit_pca()` — 对收益率变化矩阵做 PCA
- `pca_attribution()` — 用 PC 冲击重定价做归因

### 7.6 ladder.py

**职责**：5 档债券阶梯的历史回测。

核心类 `LadderBacktest`：
- `_make_bond()` — 二分法求解面值债券
- `run()` — 主循环：初始化 → 日度跟踪 → 再平衡
- 输出：投资组合时间序列 + 归因 + 再平衡日志 + 持仓

---

## 8. 手动验证指南

### 8.1 验证债券定价

最简单的情况：**面值债券**。

如果票息率 = 市场利率，且在付息日结算（AI=0），则价格应等于面值 $100。

```python
import pandas as pd
import numpy as np
from bond_pnl.bond import BondSpec
from bond_pnl.yield_curve import YieldCurve

# 平坦收益率曲线 = 4%（YieldCurve 接受百分数形式）
curve = YieldCurve(pd.Timestamp("2024-01-01"),
                   np.full(11, 4.0))  # 4% 全部期限

bond = BondSpec(maturity="2029-01-01", coupon=0.04, freq=2,
                issue_date="2024-01-01")

price = bond.dirty_price(pd.Timestamp("2024-01-01"), curve)
print(f"Price: {price:.4f}")  # 应该接近 100
```

### 8.2 验证 DV01

手动计算：
1. 在原始曲线上定价：$P_0$
2. 曲线整体上移 1bp：$P_{\text{up}}$
3. 曲线整体下移 1bp：$P_{\text{down}}$
4. $\text{DV01} = (P_{\text{down}} - P_{\text{up}}) / 2$

```python
# 承接 8.1 的 bond 和 curve（如果单独运行，需先执行 8.1 的代码）
# 或者直接复制下面完整版：
import pandas as pd, numpy as np
from bond_pnl.bond import BondSpec
from bond_pnl.yield_curve import YieldCurve

curve = YieldCurve(pd.Timestamp("2024-01-01"), np.full(11, 4.0))
bond = BondSpec(maturity="2029-01-01", coupon=0.04, freq=2, issue_date="2024-01-01")

dv01 = bond.dv01(pd.Timestamp("2024-01-01"), curve)
print(f"DV01: {dv01:.6f}")  # 5 年期约 0.04–0.05
```

### 8.3 验证归因恒等式

最关键的检查点：**Market + Time 必须精确等于 Actual**。

```python
import pandas as pd
import numpy as np
from bond_pnl.bond import BondSpec
from bond_pnl.yield_curve import YieldCurve
from bond_pnl.attribution import compute_daily_attribution

# 构造两日的曲线和债券
curve_prev = YieldCurve(pd.Timestamp("2024-01-02"), np.full(11, 4.0))
curve_curr = YieldCurve(pd.Timestamp("2024-01-03"), np.full(11, 4.05))
d_prev = pd.Timestamp("2024-01-02")
d_curr = pd.Timestamp("2024-01-03")
financing_rate = 0.05  # 5% 融资利率

bond = BondSpec(maturity="2029-01-02", coupon=0.04, freq=2,
                issue_date="2024-01-02")

a = compute_daily_attribution(bond, curve_prev, curve_curr,
                              d_prev, d_curr, financing_rate)

# 验证：Market + Time 必须精确等于 Actual
print(f"Actual: {a.actual_pnl:.8f}")
print(f"Market + Time: {a.market_impact + a.time_impact:.8f}")
assert abs(a.actual_pnl - (a.market_impact + a.time_impact)) < 1e-10
print("✓ 归因恒等式验证通过")
```

### 8.4 验证 Ladder 残差

Ladder 归因的残差应为 0.00：

```
Total PnL = Income + Rolldown + Rate Movement + Residual
```

在输出中检查 `Residual` 列是否全为 0.00。

### 8.5 运行自动化测试

```bash
python -m pytest tests/ -v
```

31 个测试全部通过。

---

## 9. 关键公式速查表

| 概念 | 公式 | 代码位置 |
|------|------|---------|
| 全价 | $PV = \sum CF_i \cdot e^{-r(t_i) \cdot t_i}$ | `bond.py:dirty_price()` |
| 应计利息 | $AI = \frac{d - d_{\text{prev}}}{d_{\text{next}} - d_{\text{prev}}} \times \text{Coupon}$ | `bond.py:accrued_interest()` |
| 净价 | $\text{Clean} = \text{Dirty} - AI$ | `bond.py:clean_price()` |
| DV01 | $\frac{P(r-1bp) - P(r+1bp)}{2}$ | `bond.py:dv01()` |
| 凸度 | $\frac{P(r+\Delta r) + P(r-\Delta r) - 2P(r)}{(\Delta r)^2}$ | `bond.py:convexity_dollar()` |
| 市场影响 | $PV(T{-}1, C_T) - PV(T{-}1, C_{T{-}1})$ | `attribution.py` |
| 时间影响 | $PV(T, C_T) - PV(T{-}1, C_T) + CF - F$ | `attribution.py` |
| 持久效应 | $-\text{DV01} \times \Delta r$ | `attribution.py` |
| 凸度效应 | $\frac{1}{2} \text{Conv} \times (\Delta r)^2$ | `attribution.py` |
| Carry | $\text{Accrual} + \text{Rolldown} - \text{Funding}$ | `attribution.py` |
| PCA 得分 | $s_k = l_k \cdot (\Delta Y - \mu)$ | `pca.py` |
| 面值求解 | 二分法使 $P(c) = 100$ | `ladder.py:_make_bond()` |

---

## 附录 A：ML 类比速查

| 金融概念 | ML 类比 |
|---------|--------|
| DV01（一阶敏感度） | 梯度 $\nabla_x f$ |
| 凸度（二阶敏感度） | 海森矩阵 $H$ |
| Duration Effect | 线性近似 $f(x+\delta) \approx f(x) + \nabla f \cdot \delta$ |
| Convexity Effect | 二次修正 $+ \frac{1}{2} \delta^T H \delta$ |
| Rate Residual | 截断误差 $O(\delta^3)$ |
| PCA on Yield Changes | 对 feature matrix 做 PCA |
| PC1 = Level | 第一主成分（最大方差方向） |
| Loadings | 特征向量 / 权重 |
| Scores | 投影坐标 |
| Bond Pricing (DCF) | 加权求和（attention-like） |
| Yield Curve Interpolation | 样条回归 |

---

## 附录 B：单位与转换

| 量 | 单位 | 转换 |
|----|------|------|
| 1 bp | 0.01% = 0.0001 (decimal) | |
| 100 bp | 1% | |
| coupon = 0.04 | 4% | 代码中用 decimal |
| yields_pct = 4.0 | 4% | FRED 数据 / YieldCurve 输入 |
| yields = 0.04 | 4% | YieldCurve 内部 |
| DV01 | $ per bp per $100 face | |
| Convexity | $ per bp² per $100 face | |
