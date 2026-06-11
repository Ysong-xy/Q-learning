# Transparent Markets and Tacit Collusion

This project simulates a repeated pricing market with reinforcement-learning sellers.

The experiment compares two information structures with the same initial state:

- **Private observation**: each seller observes only its own previous price, demand, and reward.
- **Transparent market**: each seller observes every seller's previous price, demand, and reward.

The point is to show that market transparency can make previous actions a public signal. Independent RL agents can then learn high-price coordination and punishment-like responses without explicit communication.

## Model

Each round, four market makers simultaneously choose one of two prices:

```text
low price  = 2.0  # one-shot Nash / competitive price
high price = 2.6  # symmetric joint-profit price
```

All experiments start from the same state: two market makers charged the low price and two charged the high price in the previous round. The initial average price is therefore:

```text
(2.0 + 2.0 + 2.6 + 2.6) / 4 = 2.3
```

Demand is implemented through a reduced-form Bertrand payoff table:

```text
all low:   each market maker earns 410
all high:  each market maker earns 500
mixed:     low-price market makers earn more, high-price market makers lose demand
```

Profit is:

```text
r_i = (p_i - c) q_i
```

This payoff structure has the repeated-pricing tension we want:

- If the other seller prices high, undercutting gives a larger one-round reward.
- If the other seller prices low, low price is also the best one-round response.
- Therefore the one-shot Nash outcome is low-low.
- If both sellers can condition on public history, high-high gives higher long-run profit.

With four market makers, the same logic becomes:

- all-low is the one-shot Nash outcome;
- all-high maximizes joint profit;
- a single low-price deviation is profitable in the current round;
- transparent public history lets agents learn a punishment-and-recovery pattern.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 run_experiment.py
```

Outputs:

- `results/private.npz`
- `results/transparent.npz`
- `figures/training_price_curve.png`
- `figures/final_price_comparison.png`
- `figures/final_profit_comparison.png`
- `figures/final_price_distribution.png`

## Interpretation

If the transparent-market average price is above the one-shot Nash benchmark and closer to the joint-profit price, the simulation exhibits tacit collusion. The agents are not told to collude; they only optimize discounted reward under different observation structures.

Both groups use the same initial state, discount factor, learning rates, and exploration schedule. The only structural difference is the observation:

- private agents get a repeat-last-action prior because they know their own previous action;
- transparent agents can use the public previous joint action, so their Q-table can represent: maintain all-high, punish mixed deviations, and recover from all-low.
