# Transparent Two-Seller Pricing Experiment

This project compares two fully transparent repeated Q-learning pricing markets. Both experiments use:

- two sellers;
- the same prices: `2.0` and `2.6`;
- the same initial actions: `[0, 1]`;
- the same public observation structure.

Each seller observes both sellers' previous actions, demands, and rewards. The experiments differ only in demand allocation and `gamma`.

For one seller, the first letter is its own action and the second is the rival's action:

```text
LL: own low, rival low
HL: own high, rival low
LH: own low, rival high
HH: own high, rival high
```

Both experiments satisfy:

```text
LH > HH > LL > HL
```

## High-Convergence Market

```text
gamma = 0.99
LL = 410
HL = 200
LH = 560
HH = 500
```

## Low-Convergence Market

```text
gamma = 0.30
LL = 410
HL = 50
LH = 700
HH = 430
```

Profit is:

```text
r_i = (p_i - c) q_i
```

Q-learning update:

```text
y_i = r_i + gamma * max_{a'} Q_i(s_i', a')
delta_i = y_i - Q_i(s_i, a_i)
Q_i(s_i, a_i) <- Q_i(s_i, a_i) + alpha_i * delta_i
```

Run:

```bash
python3 run_experiment.py
```

Current results:

```text
Original initialization, gamma=0.99:
final price=2.594, final profit=498.49, all-high=0.986, all-low=0.007

Original initialization, gamma=0.01 (all other parameters unchanged):
final price=2.001, final profit=409.87, all-high=0.000, all-low=0.996
```

Both cases use the exact original state-dependent Q prior (`1000` on high after
unanimous play, `1000` on low after mismatched play).  With `gamma=0.01`, that
prior delays but does not prevent convergence to low prices.  See
`GAMMA_CONTROL_EXPERIMENT_CN.md` for the controlled comparison.

For the generalized many-seller/many-price Logit-demand experiments, run:

```bash
python3 run_generalized_experiment.py
```

Parameters, payoff equations, and validated results are documented in `GENERALIZED_EXPERIMENT_CN.md`.
