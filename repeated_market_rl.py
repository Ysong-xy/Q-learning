from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


Mode = Literal["private", "transparent"]


@dataclass(frozen=True)
class RepeatedMarketConfig:
    n_market_makers: int = 4
    low_price: float = 2.0
    high_price: float = 2.6
    cost: float = 1.0
    seed: int = 2
    demand_noise: float = 0.35
    initial_actions: tuple[int, ...] = (0, 0, 1, 1)

    nash_profit: float = 410.0
    collusive_profit: float = 500.0
    temptation_profit: float = 560.0
    sucker_profit: float = 100.0


@dataclass(frozen=True)
class RepeatedTrainConfig:
    episodes: int = 70_000
    gamma: float = 0.99
    learning_rate_positive: float = 0.05
    learning_rate_negative: float = 0.001
    epsilon_start: float = 0.04
    epsilon_end: float = 0.005
    epsilon_decay: float = 20_000.0
    log_every: int = 200
    repeat_last_prior: float = 10_000.0


def price_vector(actions: np.ndarray, cfg: RepeatedMarketConfig) -> np.ndarray:
    prices = np.array([cfg.low_price, cfg.high_price])
    return prices[actions]


def demand_and_reward(actions: np.ndarray, cfg: RepeatedMarketConfig) -> tuple[np.ndarray, np.ndarray]:
    n_low = int((actions == 0).sum())
    n_high = len(actions) - n_low
    rewards = np.zeros(len(actions), dtype=float)

    if n_low == len(actions):
        rewards[:] = cfg.nash_profit
    elif n_high == len(actions):
        rewards[:] = cfg.collusive_profit
    else:
        high_share = n_high / (len(actions) - 1)
        low_share = n_low / (len(actions) - 1)
        low_reward = cfg.nash_profit + (cfg.temptation_profit - cfg.nash_profit) * high_share
        high_reward = cfg.sucker_profit * (1.0 - 0.5 * low_share)
        rewards[actions == 0] = low_reward
        rewards[actions == 1] = high_reward

    demand = rewards / (price_vector(actions, cfg) - cfg.cost)
    return demand, rewards


def benchmarks(cfg: RepeatedMarketConfig) -> dict[str, float]:
    low = np.zeros(cfg.n_market_makers, dtype=int)
    high = np.ones(cfg.n_market_makers, dtype=int)
    _, low_rewards = demand_and_reward(low, cfg)
    _, high_rewards = demand_and_reward(high, cfg)
    return {
        "nash_price": cfg.low_price,
        "collusive_price": cfg.high_price,
        "nash_profit": float(low_rewards.mean()),
        "collusive_profit": float(high_rewards.mean()),
    }


def _reward_bin(reward: float) -> int:
    return int(np.clip(reward // 80, 0, 9))


def _observed_rewards(rewards: np.ndarray, rng: np.random.Generator, cfg: RepeatedMarketConfig) -> np.ndarray:
    noise = rng.normal(0.0, cfg.demand_noise, size=len(rewards))
    return np.maximum(0.0, rewards * (1.0 + noise))


def _state(agent: int, actions: np.ndarray, observed_rewards: np.ndarray, mode: Mode) -> tuple[int, ...]:
    if mode == "private":
        return (int(actions[agent]), _reward_bin(float(observed_rewards[agent])))
    return tuple(int(action) for action in actions)


def _initial_q(agent: int, state: tuple[int, ...], mode: Mode, train_cfg: RepeatedTrainConfig) -> np.ndarray:
    q = np.zeros(2)
    if mode == "transparent":
        if all(action == 1 for action in state) or all(action == 0 for action in state):
            q[1] = train_cfg.repeat_last_prior
        else:
            q[0] = train_cfg.repeat_last_prior
        return q

    previous_action = state[0]
    q[previous_action] = train_cfg.repeat_last_prior
    return q


def train(mode: Mode, market_cfg: RepeatedMarketConfig, train_cfg: RepeatedTrainConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(market_cfg.seed + (10_000 if mode == "transparent" else 0))
    q_tables: list[dict[tuple[int, ...], np.ndarray]] = [dict() for _ in range(market_cfg.n_market_makers)]

    if len(market_cfg.initial_actions) != market_cfg.n_market_makers:
        raise ValueError("initial_actions must have one entry per market maker.")
    last_actions = np.array(market_cfg.initial_actions, dtype=int)
    _, last_rewards = demand_and_reward(last_actions, market_cfg)
    last_observed_rewards = _observed_rewards(last_rewards, rng, market_cfg)

    action_history = []
    price_history = []
    reward_history = []
    initial_prices = price_vector(last_actions, market_cfg)
    logs = [
        [
            0,
            train_cfg.epsilon_start,
            float(initial_prices.mean()),
            float(last_rewards.mean()),
            float((last_actions == 1).all()),
            float((last_actions == 0).all()),
        ]
    ]

    for t in range(train_cfg.episodes):
        epsilon = train_cfg.epsilon_end + (train_cfg.epsilon_start - train_cfg.epsilon_end) * np.exp(-t / train_cfg.epsilon_decay)

        states = [_state(i, last_actions, last_observed_rewards, mode) for i in range(market_cfg.n_market_makers)]
        actions = np.zeros(market_cfg.n_market_makers, dtype=int)

        for i, state in enumerate(states):
            q = q_tables[i].setdefault(state, _initial_q(i, state, mode, train_cfg))
            actions[i] = rng.integers(0, 2) if rng.random() < epsilon else int(rng.choice(np.flatnonzero(q == q.max())))

        demands, rewards = demand_and_reward(actions, market_cfg)
        observed_rewards = _observed_rewards(rewards, rng, market_cfg)
        next_states = [_state(i, actions, observed_rewards, mode) for i in range(market_cfg.n_market_makers)]

        for i, state in enumerate(states):
            q = q_tables[i].setdefault(state, _initial_q(i, state, mode, train_cfg))
            next_q = q_tables[i].setdefault(next_states[i], _initial_q(i, next_states[i], mode, train_cfg))
            target = rewards[i] + train_cfg.gamma * next_q.max()
            td_error = target - q[actions[i]]
            learning_rate = train_cfg.learning_rate_positive if td_error >= 0 else train_cfg.learning_rate_negative
            q[actions[i]] += learning_rate * td_error

        last_actions = actions
        last_rewards = rewards
        last_observed_rewards = observed_rewards

        action_history.append(actions.copy())
        price_history.append(price_vector(actions, market_cfg))
        reward_history.append(rewards.copy())

        if (t + 1) % train_cfg.log_every == 0:
            recent_actions = np.array(action_history[-train_cfg.log_every :])
            recent_prices = np.array(price_history[-train_cfg.log_every :])
            recent_rewards = np.array(reward_history[-train_cfg.log_every :])
            high_high = (recent_actions == 1).all(axis=1).mean()
            low_low = (recent_actions == 0).all(axis=1).mean()
            logs.append([t + 1, epsilon, recent_prices.mean(), recent_rewards.mean(), high_high, low_low])

    return {
        "logs": np.array(logs),
        "actions": np.array(action_history),
        "prices": np.array(price_history),
        "rewards": np.array(reward_history),
    }


def save_results(results: dict[str, dict[str, np.ndarray]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode, data in results.items():
        np.savez_compressed(out_dir / f"{mode}.npz", **data)
