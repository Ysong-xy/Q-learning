from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class RepeatedMarketConfig:
    n_market_makers: int = 2
    price_levels: tuple[float, ...] = (2.0, 2.6)
    cost: float = 1.0
    seed: int = 2
    initial_actions: tuple[int, ...] = (0, 1)

    # ``calibrated_2x2`` reproduces the original payoff table. ``logit`` is
    # defined for any number of sellers and price levels.
    demand_model: Literal["calibrated_2x2", "logit"] = "calibrated_2x2"
    market_size: float = 1_000.0
    consumer_value: float = 5.0
    price_sensitivity: float = 1.25
    differentiation: float = 0.80

    demand_all_low: float = 410.0
    demand_all_high: float = 312.5
    demand_low_when_undercutting: float = 560.0
    demand_high_when_undercut: float = 125.0

    @property
    def n_actions(self) -> int:
        return len(self.price_levels)

    @property
    def low_action(self) -> int:
        return 0

    @property
    def high_action(self) -> int:
        return self.n_actions - 1

    @property
    def low_price(self) -> float:
        return self.price_levels[self.low_action]

    @property
    def high_price(self) -> float:
        return self.price_levels[self.high_action]


@dataclass(frozen=True)
class RepeatedTrainConfig:
    episodes: int = 50_000
    gamma: float = 0.99
    learning_rate_positive: float = 0.05
    learning_rate_negative: float = 0.001
    epsilon_start: float = 0.04
    epsilon_end: float = 0.002
    epsilon_decay: float = 20_000.0
    log_every: int = 200
    repeat_last_prior: float = 1_000.0
    state_bins: int = 10
    q_initialization: Literal["repeat", "zero", "legacy_consensus_high", "theoretical_trigger"] = "repeat"


def price_vector(actions: np.ndarray, cfg: RepeatedMarketConfig) -> np.ndarray:
    prices = np.array(cfg.price_levels)
    return prices[actions]


def demand_and_reward(actions: np.ndarray, cfg: RepeatedMarketConfig) -> tuple[np.ndarray, np.ndarray]:
    if np.any(actions < 0) or np.any(actions >= cfg.n_actions):
        raise ValueError("actions must be valid price-level indexes.")
    if len(actions) != cfg.n_market_makers:
        raise ValueError("actions must contain one action per market maker.")

    prices = price_vector(actions, cfg)
    if cfg.demand_model == "logit":
        if cfg.market_size <= 0 or cfg.differentiation <= 0:
            raise ValueError("market_size and differentiation must be positive.")
        utilities = (cfg.consumer_value - cfg.price_sensitivity * prices) / cfg.differentiation
        shift = max(0.0, float(utilities.max()))
        exp_utilities = np.exp(utilities - shift)
        outside_option = np.exp(-shift)
        shares = exp_utilities / (outside_option + exp_utilities.sum())
        demand = cfg.market_size * shares
        return demand, (prices - cfg.cost) * demand

    if cfg.n_market_makers != 2 or cfg.n_actions != 2:
        raise ValueError("calibrated_2x2 demand requires exactly two sellers and two prices; use demand_model='logit' otherwise.")

    low_action = cfg.low_action
    high_action = cfg.high_action
    demand = np.zeros(cfg.n_market_makers, dtype=float)

    if np.all(actions == low_action):
        demand[:] = cfg.demand_all_low
    elif np.all(actions == high_action):
        demand[:] = cfg.demand_all_high
    else:
        demand[actions == low_action] = cfg.demand_low_when_undercutting
        demand[actions == high_action] = cfg.demand_high_when_undercut

    rewards = (prices - cfg.cost) * demand
    return demand, rewards


def benchmarks(cfg: RepeatedMarketConfig) -> dict[str, float]:
    low = np.full(cfg.n_market_makers, cfg.low_action, dtype=int)
    high = np.full(cfg.n_market_makers, cfg.high_action, dtype=int)
    _, low_rewards = demand_and_reward(low, cfg)
    _, high_rewards = demand_and_reward(high, cfg)
    return {
        "nash_price": cfg.low_price,
        "collusive_price": cfg.high_price,
        "nash_profit": float(low_rewards.mean()),
        "collusive_profit": float(high_rewards.mean()),
    }


def _value_bin(value: float, upper: float, bins: int) -> int:
    if upper <= 0:
        return 0
    scaled = np.clip(value / upper, 0.0, 0.999999)
    return int(scaled * bins)


def _state(
    actions: np.ndarray,
    demands: np.ndarray,
    rewards: np.ndarray,
    market_cfg: RepeatedMarketConfig,
    train_cfg: RepeatedTrainConfig,
) -> tuple[int, ...]:
    max_demand = market_cfg.market_size if market_cfg.demand_model == "logit" else max(
        market_cfg.demand_all_low,
        market_cfg.demand_all_high,
        market_cfg.demand_low_when_undercutting,
        market_cfg.demand_high_when_undercut,
    )
    max_reward = max((max(market_cfg.price_levels) - market_cfg.cost) * max_demand, 1.0)
    state: list[int] = []
    state.extend(int(action) for action in actions)
    state.extend(_value_bin(float(demand), max_demand, train_cfg.state_bins) for demand in demands)
    state.extend(_value_bin(float(reward), max_reward, train_cfg.state_bins) for reward in rewards)
    return tuple(state)


def _initial_q(
    state: tuple[int, ...],
    agent_id: int,
    market_cfg: RepeatedMarketConfig,
    train_cfg: RepeatedTrainConfig,
) -> np.ndarray:
    q = np.zeros(market_cfg.n_actions)
    if train_cfg.q_initialization == "zero":
        return q

    previous_actions = state[: market_cfg.n_market_makers]
    if train_cfg.q_initialization == "legacy_consensus_high":
        # Exact behaviour of the original project: every unanimous state,
        # including all-low, receives a prior on the highest-price action.
        if len(set(previous_actions)) == 1:
            q[market_cfg.high_action] = train_cfg.repeat_last_prior
        else:
            q[market_cfg.low_action] = train_cfg.repeat_last_prior
        return q

    if train_cfg.q_initialization == "theoretical_trigger":
        if market_cfg.demand_model != "calibrated_2x2" or market_cfg.n_market_makers != 2 or market_cfg.n_actions != 2:
            raise ValueError("theoretical_trigger initialization requires the calibrated two-seller/two-price model.")
        if not 0 <= agent_id < market_cfg.n_market_makers:
            raise ValueError("agent_id is out of range.")

        low = market_cfg.low_action
        high = market_cfg.high_action
        _, reward_ll = demand_and_reward(np.array([low, low]), market_cfg)
        _, reward_hh = demand_and_reward(np.array([high, high]), market_cfg)
        deviation = np.full(2, high, dtype=int)
        deviation[agent_id] = low
        _, reward_deviation = demand_and_reward(deviation, market_cfg)
        punished = np.full(2, low, dtype=int)
        punished[agent_id] = high
        _, reward_punished = demand_and_reward(punished, market_cfg)

        gamma = train_cfg.gamma
        cooperative_value = reward_hh[agent_id] / (1.0 - gamma)
        if len(set(previous_actions)) == 1:
            # Cooperation: remain high forever. A deviation earns the
            # undercutting payoff, one all-low punishment round, then recovers.
            q[high] = cooperative_value
            q[low] = reward_deviation[agent_id] + gamma * reward_ll[agent_id] + gamma**2 * cooperative_value
        else:
            # One-round punishment: both choose low, then return to cooperation.
            q[low] = reward_ll[agent_id] + gamma * cooperative_value
            q[high] = reward_punished[agent_id] + gamma * cooperative_value
        return q

    if len(set(previous_actions)) == 1:
        # The old code always favoured the highest action here, even after an
        # all-low round. Honour the parameter name and repeat the actual action.
        q[previous_actions[0]] = train_cfg.repeat_last_prior
    else:
        q[market_cfg.low_action] = train_cfg.repeat_last_prior
    return q


def train(market_cfg: RepeatedMarketConfig, train_cfg: RepeatedTrainConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(market_cfg.seed)
    q_tables: list[dict[tuple[int, ...], np.ndarray]] = [dict() for _ in range(market_cfg.n_market_makers)]

    if len(market_cfg.initial_actions) != market_cfg.n_market_makers:
        raise ValueError("initial_actions must have one entry per market maker.")
    last_actions = np.array(market_cfg.initial_actions, dtype=int)
    if np.any(last_actions < 0) or np.any(last_actions >= market_cfg.n_actions):
        raise ValueError("initial_actions must contain valid price-level indexes.")
    last_demands, last_rewards = demand_and_reward(last_actions, market_cfg)

    action_history = []
    demand_history = []
    price_history = []
    reward_history = []
    initial_prices = price_vector(last_actions, market_cfg)
    logs = [
        [
            0,
            train_cfg.epsilon_start,
            float(initial_prices.mean()),
            float(last_rewards.mean()),
            float((last_actions == market_cfg.high_action).all()),
            float((last_actions == market_cfg.low_action).all()),
        ]
    ]

    for t in range(train_cfg.episodes):
        epsilon = train_cfg.epsilon_end + (train_cfg.epsilon_start - train_cfg.epsilon_end) * np.exp(-t / train_cfg.epsilon_decay)

        public_state = _state(last_actions, last_demands, last_rewards, market_cfg, train_cfg)
        states = [public_state for _ in range(market_cfg.n_market_makers)]
        actions = np.zeros(market_cfg.n_market_makers, dtype=int)

        for i, state in enumerate(states):
            q = q_tables[i].setdefault(state, _initial_q(state, i, market_cfg, train_cfg))
            actions[i] = (
                rng.integers(0, market_cfg.n_actions)
                if rng.random() < epsilon
                else int(rng.choice(np.flatnonzero(q == q.max())))
            )

        demands, rewards = demand_and_reward(actions, market_cfg)
        next_public_state = _state(actions, demands, rewards, market_cfg, train_cfg)
        next_states = [next_public_state for _ in range(market_cfg.n_market_makers)]

        for i, state in enumerate(states):
            q = q_tables[i].setdefault(state, _initial_q(state, i, market_cfg, train_cfg))
            next_q = q_tables[i].setdefault(
                next_states[i], _initial_q(next_states[i], i, market_cfg, train_cfg)
            )
            target = rewards[i] + train_cfg.gamma * next_q.max()
            td_error = target - q[actions[i]]
            learning_rate = train_cfg.learning_rate_positive if td_error >= 0 else train_cfg.learning_rate_negative
            q[actions[i]] += learning_rate * td_error

        last_actions = actions
        last_demands = demands
        last_rewards = rewards

        action_history.append(actions.copy())
        demand_history.append(demands.copy())
        price_history.append(price_vector(actions, market_cfg))
        reward_history.append(rewards.copy())

        if (t + 1) % train_cfg.log_every == 0:
            recent_actions = np.array(action_history[-train_cfg.log_every :])
            recent_prices = np.array(price_history[-train_cfg.log_every :])
            recent_rewards = np.array(reward_history[-train_cfg.log_every :])
            high_high = (recent_actions == market_cfg.high_action).all(axis=1).mean()
            low_low = (recent_actions == market_cfg.low_action).all(axis=1).mean()
            logs.append([t + 1, epsilon, recent_prices.mean(), recent_rewards.mean(), high_high, low_low])

    return {
        "logs": np.array(logs),
        "actions": np.array(action_history),
        "demands": np.array(demand_history),
        "prices": np.array(price_history),
        "rewards": np.array(reward_history),
    }


def save_results(results: dict[str, dict[str, np.ndarray]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for scenario_key, data in results.items():
        np.savez_compressed(out_dir / f"{scenario_key}.npz", **data)
