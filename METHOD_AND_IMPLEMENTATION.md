# 两市商透明市场 Q-Learning 实验：方法说明

## 1. 共同设定

两个实验均使用：

```text
市商数量：2
价格动作：low = 2.0, high = 2.6
单位成本：1.0
初始动作：[0, 1]
训练轮数：50,000
透明观测
```

两个 Agent 都能看到上一轮双方的动作、需求和收益，因此上一轮市场状态是公共状态。

## 2. 收益记号

第一个字母表示自己，第二个字母表示对手：

```text
LL：自己低价，对手低价
HL：自己高价，对手低价
LH：自己低价，对手高价
HH：自己高价，对手高价
```

两组实验都严格满足：

```text
LH > HH > LL > HL
```

其经济含义是：

- 对手高价时，自己降价有短期诱惑：`LH > HH`；
- 双方高价比双方低价更赚钱：`HH > LL`；
- 对手低价时，自己也应选择低价：`LL > HL`。

## 3. 高价收敛组

```text
gamma = 0.99
LL = 410
HL = 200
LH = 560
HH = 500
```

需求参数：

```text
demand_all_low = 410.0
demand_all_high = 312.5
demand_low_when_undercutting = 560.0
demand_high_when_undercut = 125.0
```

由于 `gamma` 很高，Agent 重视偏离后可能出现的未来低价惩罚。透明公共历史使惩罚和恢复策略可以被学习，因此最终接近双方高价。

结果：

```text
final price = 2.594
final profit = 498.49
all-high = 0.986
```

## 4. 低价收敛组

```text
gamma = 0.30
LL = 410
HL = 50
LH = 700
HH = 430
```

需求参数：

```text
demand_all_low = 410.0
demand_all_high = 268.75
demand_low_when_undercutting = 700.0
demand_high_when_undercut = 31.25
```

这里高价合作相对低价只增加 `20`，而当期降价收益达到 `700`。较低的 `gamma` 使未来惩罚不足以抵消当前偏离收益，因此最终收敛到双方低价。

结果：

```text
final price = 2.001
final profit = 409.85
all-low = 0.996
```

## 5. Q-Learning

```text
y_i = r_i + gamma * max_{a'} Q_i(s_i', a')
delta_i = y_i - Q_i(s_i, a_i)
Q_i(s_i, a_i) <- Q_i(s_i, a_i) + alpha_i * delta_i
```

代码使用 hysteretic learning rate：

```text
alpha_i = alpha_pos, if delta_i >= 0
alpha_i = alpha_neg, if delta_i < 0
```

## 6. 结论

在商品价格、初始状态和透明观测结构相同，并且两组都满足 `LH > HH > LL > HL` 的情况下，需求收益差距和 `gamma` 的不同可以导致：

```text
高 gamma -> 高价协调
低 gamma + 强降价诱惑 -> 低价竞争
```
