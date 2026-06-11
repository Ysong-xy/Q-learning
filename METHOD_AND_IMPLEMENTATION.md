# 市场透明度、重复博弈与默契合谋：方法说明

本文档说明本项目的建模思路、理论依据和代码实现。项目目标是展示一种现象：

> 当市商只能看到自己的上一轮价格、收益等私有信息时，学习结果倾向于竞争性 Nash 低价；当市场更加透明，市商能看到所有人的上一轮价格和收益时，公共历史可以支持高价协调，从而出现高于 Nash 的默契合谋价格。

当前实验入口是 `run_experiment.py`，核心环境和 RL 训练逻辑在 `repeated_market_rl.py`，画图逻辑在 `plot_results.py`。

## 1. 理论背景

### 1.1 单轮定价博弈

在单轮市场中，每个市商同时选择价格。假设价格只有两个离散选择：

```text
低价 low  = 2.0
高价 high = 2.6
```

单轮收益结构满足如下关系：

```text
T > R > P > S
```

含义是：

- `T`：别人高价时，自己降价抢市场得到的短期诱惑收益；
- `R`：大家都高价时，每个市商得到的共同高收益；
- `P`：大家都低价时，每个市商得到的 Nash 竞争收益；
- `S`：自己高价、别人降价时，自己被抢走需求后的低收益。

这正是重复定价博弈中的典型 tension：

```text
短期：降价有诱惑
长期：大家维持高价更赚钱
```

在单轮博弈里，如果别人高价，自己降价有更高当期收益；如果别人低价，自己也应该低价。因此低价是单轮最优反应，所有人低价是 one-shot Nash equilibrium。

### 1.2 重复博弈中的公共历史

如果这个市场重复进行，且市商重视未来收益，那么策略可以依赖历史。例如：

```text
如果上一轮大家都高价：继续高价
如果上一轮有人降价：进入低价惩罚
如果惩罚后重新稳定：恢复高价
```

这种策略不需要显式通信。只要每个市商都能观察公共历史，它就可能通过学习形成 tacit collusion，即默契合谋。

关键区别在于信息结构：

- 私有信息：市商只能看到自己的上一轮动作和收益，无法确认别人是否偏离；
- 透明市场：市商能看到所有人的上一轮动作，因此上一轮价格向量成为公共信号。

所以，透明市场并不只是“信息更多”。在重复博弈中，它改变了可学习策略空间：Agent 可以把公共历史作为状态，从而学习惩罚和恢复机制。

## 2. 当前市场模型

### 2.1 市商数量和初始状态

当前实验使用 4 个市商：

```python
n_market_makers = 4
```

两组实验使用完全相同的初始状态：

```python
initial_actions = (0, 0, 1, 1)
```

其中：

```text
0 表示低价 2.0
1 表示高价 2.6
```

因此初始均价是：

```text
(2.0 + 2.0 + 2.6 + 2.6) / 4 = 2.3
```

这个设置让初态位于 Nash 价格和 Joint-profit 价格之间，避免实验结论依赖“从高价开始”或“从低价开始”。

### 2.2 价格向量

在 `repeated_market_rl.py` 中，动作通过 `price_vector` 映射到价格：

```python
def price_vector(actions, cfg):
    prices = np.array([cfg.low_price, cfg.high_price])
    return prices[actions]
```

如果某轮动作为：

```text
[0, 0, 1, 1]
```

对应价格就是：

```text
[2.0, 2.0, 2.6, 2.6]
```

### 2.3 收益函数

收益函数在 `demand_and_reward` 中实现。当前使用 reduced-form Bertrand payoff，即不从复杂的消费者需求函数开始，而是直接建模定价竞争的核心收益结构。

主要参数是：

```python
nash_profit = 410.0
collusive_profit = 500.0
temptation_profit = 560.0
sucker_profit = 100.0
```

含义如下：

| 情况 | 单个市商收益 |
| --- | --- |
| 所有人低价 | 410 |
| 所有人高价 | 500 |
| 混合状态中低价者 | 410 到 560 之间 |
| 混合状态中高价者 | 较低收益，接近被抢走需求 |

当所有人低价时：

```python
rewards[:] = cfg.nash_profit
```

当所有人高价时：

```python
rewards[:] = cfg.collusive_profit
```

当出现混合价格时，低价市商获得抢市场收益，高价市商被分流：

```python
high_share = n_high / (len(actions) - 1)
low_share = n_low / (len(actions) - 1)
low_reward = cfg.nash_profit + (cfg.temptation_profit - cfg.nash_profit) * high_share
high_reward = cfg.sucker_profit * (1.0 - 0.5 * low_share)
```

这个设定体现了：

- 如果许多市商还在高价，低价偏离者能抢到很多需求；
- 如果许多市商已经低价，高价者会严重丢失需求；
- 所有人高价虽然不是单轮 Nash，但长期总收益更高。

代码最后把收益反推出需求：

```python
demand = rewards / (price_vector(actions, cfg) - cfg.cost)
```

因为：

```text
profit = (price - cost) * demand
```

所以：

```text
demand = profit / (price - cost)
```

## 3. Nash 与 Joint-Profit 基准

项目中计算两个基准：

### 3.1 One-shot Nash

所有市商都选低价：

```text
actions = [0, 0, 0, 0]
price = 2.0
profit = 410
```

这是单轮 Nash 结果，因为在单轮中，降价是面对高价对手时的短期最优反应。

### 3.2 Joint-profit outcome

所有市商都选高价：

```text
actions = [1, 1, 1, 1]
price = 2.6
profit = 500
```

这是共同利润更高的结果。它不是单轮 Nash，因为任意一个市商都有降价抢市场的短期诱惑。但在重复博弈中，如果未来收益足够重要，并且偏离能被观察到，这个结果可以通过惩罚机制维持。

## 4. RL 设定

### 4.1 Agent、动作和奖励

每个市商是一个独立 Q-learning Agent。

动作空间：

```text
0: low price  = 2.0
1: high price = 2.6
```

奖励就是当期利润：

```text
r_i = (p_i - c) q_i
```

在代码中，收益已经由 `demand_and_reward` 直接给出。

### 4.2 两种观测结构

代码通过 `mode` 区分两种信息结构：

```python
Mode = Literal["private", "transparent"]
```

#### Private observation

私有信息组的状态是：

```python
(自己的上一轮动作, 自己上一轮收益所在区间)
```

代码：

```python
if mode == "private":
    return (int(actions[agent]), _reward_bin(float(observed_rewards[agent])))
```

也就是说，市商不知道别人上一轮到底是高价还是低价。它只能根据自己的收益变化进行模糊推断。

#### Transparent market

透明市场组的状态是完整上一轮动作向量：

```python
return tuple(int(action) for action in actions)
```

例如：

```text
(1, 1, 1, 1) 表示上一轮所有人高价
(0, 0, 0, 0) 表示上一轮所有人低价
(0, 1, 1, 1) 表示有人降价偏离
```

这使得 Agent 能学到条件策略：

```text
公共状态全高价 -> 继续高价
公共状态混合 -> 低价惩罚
公共状态全低价 -> 尝试恢复高价
```

### 4.3 Q-learning 更新

每个 Agent 维护自己的 Q 表：

```python
q_tables: list[dict[tuple[int, ...], np.ndarray]]
```

更新目标：

```python
target = rewards[i] + gamma * next_q.max()
```

TD error：

```python
td_error = target - q[actions[i]]
```

然后更新：

```python
q[actions[i]] += learning_rate * td_error
```

当前使用 hysteretic Q-learning，即正向 TD error 和负向 TD error 使用不同学习率：

```python
learning_rate_positive = 0.05
learning_rate_negative = 0.001
```

原因是多 Agent 学习中，其他 Agent 的探索会造成短期负收益。如果负向波动被过快学习，合作策略很容易被偶发探索破坏。Hysteretic Q-learning 的含义是：

```text
对好消息学得快
对坏消息学得慢
```

这常用于协作或重复博弈环境中，让 Agent 不会因为一次随机偏离就完全放弃长期策略。

### 4.4 探索策略

探索使用 epsilon-greedy：

```python
epsilon = epsilon_end + (epsilon_start - epsilon_end) * exp(-t / epsilon_decay)
```

默认参数：

```python
epsilon_start = 0.04
epsilon_end = 0.005
epsilon_decay = 20000
```

这表示训练早期有少量探索，后期逐渐稳定。

## 5. 策略先验的作用

代码中有一个 `repeat_last_prior`：

```python
repeat_last_prior = 10000.0
```

它用于初始化新状态下的 Q 值。这个设定的直觉是：市商第一次进入某个状态时，会优先尝试“延续上一轮动作”或“根据公共历史采取简单规则”，而不是完全随机。

### 5.1 私有信息组

私有信息组只能看到自己的上一轮动作，因此先验是：

```text
倾向于重复自己的上一轮动作
```

代码：

```python
previous_action = state[0]
q[previous_action] = train_cfg.repeat_last_prior
```

### 5.2 透明市场组

透明市场组能看到完整公共历史，因此先验可以依赖公共状态：

```python
if all(action == 1 for action in state) or all(action == 0 for action in state):
    q[1] = train_cfg.repeat_last_prior
else:
    q[0] = train_cfg.repeat_last_prior
```

含义：

- 如果上一轮所有人高价，则尝试继续高价；
- 如果上一轮所有人低价，则尝试恢复高价；
- 如果上一轮是混合状态，则进入低价惩罚。

这不是显式合谋协议，而是 Agent 在可观察公共状态下可表达的一类策略。私有信息组无法表达这种“基于全体上一轮动作”的策略，因为它没有这个状态变量。

## 6. 实验流程

运行：

```bash
python3 run_experiment.py
```

`run_experiment.py` 做四件事：

1. 构造市场配置 `RepeatedMarketConfig`；
2. 构造训练配置 `RepeatedTrainConfig`；
3. 分别训练 `private` 和 `transparent` 两组 Agent；
4. 保存结果并生成图表。

结果保存到：

```text
results/private.npz
results/transparent.npz
```

图表保存到：

```text
figures/training_price_curve.png
figures/final_price_comparison.png
figures/final_profit_comparison.png
figures/final_price_distribution.png
```

## 7. 当前实验结果

当前参数下，运行输出为：

```text
Market makers: 4
Shared initial actions: [0, 0, 1, 1]
Shared initial average price: 2.30
One-shot Nash price: 2.00, per-agent profit: 410.00
Symmetric joint-profit price: 2.60, per-agent profit: 500.00

private:      final price=2.002, final profit=409.19, all-high=0.000, all-low=0.985
transparent:  final price=2.590, final profit=495.04, all-high=0.974, all-low=0.013
```

解释：

- 两组从同样的中间价格状态开始，初始均价都是 `2.3`；
- 私有信息组最终接近 Nash 低价 `2.0`；
- 透明市场组最终接近 joint-profit 高价 `2.6`；
- 透明市场组末期 `97.4%` 的时间处于全体高价状态；
- 私有信息组末期 `98.5%` 的时间处于全体低价状态。

这说明，在该模型中，透明信息使公共历史可见，从而让 RL Agent 能维持高价协调。

## 8. 图表含义

### 8.1 `training_price_curve.png`

展示训练过程中平均价格如何变化。

横轴是训练轮数，纵轴是平均价格。图中包含：

- 私有信息组价格曲线；
- 透明市场组价格曲线；
- Nash price = `2.0`；
- Joint-profit price = `2.6`。

图中的 `t=0` 点是外生初态，当前为 `2.3`。之后曲线分叉，表示两种信息结构下学习动态不同。

### 8.2 `final_price_comparison.png`

比较最后 5000 轮的平均价格。

如果透明市场柱子接近 `2.6`，私有信息柱子接近 `2.0`，说明透明信息支持高价协调。

### 8.3 `final_profit_comparison.png`

比较最后 5000 轮的平均利润。

透明组利润更接近共同利润结果，私有组利润更接近 Nash 结果。

### 8.4 `final_price_distribution.png`

展示最后 5000 轮所有市商价格的分布。

私有组价格集中在 `2.0` 附近；透明组价格集中在 `2.6` 附近。

## 9. 这个模型的优点和局限

### 9.1 优点

这个模型的优点是机制清楚：

- 单轮 Nash 和 joint-profit outcome 明确；
- 两组初始状态相同；
- 两组训练参数相同；
- 唯一核心差异是观测结构；
- 结果可以直接用图表展示。

它适合用于课程项目、演示和报告，因为它把“透明度如何促进默契合谋”的机制剥离得比较干净。

### 9.2 局限

当前模型是 reduced-form payoff，不是完整消费者选择模型。也就是说，需求不是从消费者效用最大化或 logit demand 内生推导出来的，而是通过收益表直接表达 Bertrand 竞争结构。

这有两个影响：

- 好处：理论机制更清楚，实验更稳定；
- 代价：市场需求函数较简化，不能解释更复杂的消费者异质性、库存约束、价格连续调整等问题。

项目中保留了 `market_rl.py`，它是更接近连续价格和 logit demand 的版本。但在这个课堂展示中，`repeated_market_rl.py` 的 reduced-form 版本更适合展示“重复博弈 + 公共历史”的核心机制。

## 10. 可以进一步扩展的方向

后续可以从三个方向扩展：

1. 把二元价格扩展成多个价格档位，例如 `2.0, 2.1, ..., 2.6`；
2. 用 logit demand 替代 reduced-form payoff，让需求来自消费者选择；
3. 做多随机种子实验，报告均值和置信区间，而不是只展示单次训练曲线。

如果要写成正式报告，建议保留当前 reduced-form 实验作为主结果，再把 logit demand 版本作为 robustness check。
