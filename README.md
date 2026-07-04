# Multi-Agent Q-Learning Pricing Experiments

本项目使用多智能体 Q-learning 模拟重复定价市场，比较卖家数量、价格动作数和折扣因子 $\gamma$ 对价格、利润与策略稳定性的影响。

当前主实验包含三种市场：

| 市场 | 卖家数 | 价格动作 |
|---|---:|---|
| 多人两价 | 4 | $\{2.0,2.6\}$ |
| 两人多价 | 2 | $\{2.0,2.1,\ldots,2.6\}$ |
| 多人多价 | 4 | $\{2.0,2.1,\ldots,2.6\}$ |

三种市场使用相同的价格范围 $[2.0,2.6]$，避免将价格范围差异误当成市场结构差异。

## 需求与利润

通用多人/多价实验使用包含外部选项的 Logit 需求。卖家 $i$ 选择价格 $p_i$ 后，

$$
u_i=\frac{v-\beta p_i}{\mu},
\qquad
s_i=\frac{e^{u_i}}{1+\sum_{j=1}^{N}e^{u_j}},
$$

$$
q_i=M s_i,
\qquad
r_i=(p_i-c)q_i.
$$

默认市场参数为

$$
M=1000,quad c=1,quad v=5,quad \beta=1.25,quad \mu=0.80.
$$

## Q-learning

每个 Agent 观察上一轮所有卖家的动作、需求和利润分箱。更新目标为

$$
y_i=r_i+\gamma\max_{a'}Q_i(s',a'),
\qquad
\delta_i=y_i-Q_i(s,a_i).
$$

代码使用 hysteretic learning rate：

$$
Q_i(s,a_i)\leftarrow Q_i(s,a_i)+
\begin{cases}
\alpha_+\delta_i,&\delta_i\ge0,\\
\alpha_-\delta_i,&\delta_i<0,
\end{cases}
$$

其中 $\alpha_+=0.05$、$\alpha_-=0.01$。Gamma sweep 实验使用零初始化 $Q_0(s,a)=0$，而不使用高价先验。

## Gamma sweep 主实验

每种市场分别测试

$$
\gamma\in\{0.01,0.30,0.60,0.95,0.99\}.
$$

共运行 15 个实验，每个训练 $1,000,000$ 轮，探索率由 $0.20$ 衰减至 $0.001$。收敛检查比较最后两个各 $50,000$ 轮的窗口，要求：

$$
|\Delta\bar p|<0.01,
\qquad
\operatorname{TV}(\pi_{t-1},\pi_t)<0.03.
$$

运行完整实验：

```bash
python3 run_gamma_sweep.py --episodes 1000000 --workers 3
```

最后 $50,000$ 轮的结果摘要：

| 市场 | 主要结果 |
|---|---|
| 多人两价 | $\gamma\le0.95$ 时几乎全员低价；$\gamma=0.99$ 未通过收敛检查 |
| 两人多价 | $\gamma\le0.60$ 收敛到低价附近；$0.95/0.99$ 价格更高但未收敛 |
| 多人多价 | 五组均通过经验收敛检查，稳定在非对称价格分布 |

完整表格、数值和讨论见 [`GAMMA_SWEEP_RESULTS_CN.md`](GAMMA_SWEEP_RESULTS_CN.md) 与 [`results/gamma_sweep/summary.csv`](results/gamma_sweep/summary.csv)。

## 图像

每种市场分开绘图：

```text
figures/gamma_sweep/
├── many_agents_two_prices/
├── two_agents_many_prices/
└── many_agents_many_prices/
```

每个目录包含：

- `training_curves_by_gamma.png`：价格、利润、需求和同价率的完整训练曲线；
- `final_metrics_by_gamma.png`：最后价格、利润和同价率对 $\gamma$ 的比较。

## 两人两价初始化对照

`run_experiment.py` 是一个独立的两人两价对照实验，用来检验原始状态相关 Q 先验在 $\gamma=0.99$ 和 $\gamma=0.01$ 下的表现：

```bash
python3 run_experiment.py
```

该实验与上述零初始化 Gamma sweep 不应混为一组。详细说明见 [`GAMMA_CONTROL_EXPERIMENT_CN.md`](GAMMA_CONTROL_EXPERIMENT_CN.md)。

## 安装

```bash
python3 -m pip install -r requirements.txt
```

主要代码：

- `repeated_market_rl.py`：市场、需求、状态、Q 表和训练循环；
- `run_gamma_sweep.py`：15 组主实验与汇总；
- `plot_gamma_sweep.py`：Gamma sweep 绘图；
- `run_experiment.py`：两人两价原始 Q 先验对照。

> 注：多智能体独立 Q-learning 面对的环境并非平稳，因此即使训练很长，也不存在一般性的理论收敛保证。本项目中的“收敛”特指通过上述尾部窗口稳定性检查。
