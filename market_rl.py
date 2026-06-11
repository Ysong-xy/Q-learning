from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


ObservationMode = Literal["private", "transparent"]


@dataclass(frozen=True)
class MarketConfig:
    n_agents: int = 2
    market_size: float = 1000.0
    value: float = 5.0
    alpha: float = 1.25
    mu: float = 0.55
    cost: float = 1.0
    prices: tuple[float, ...] = tuple(np.round(np.arange(1.0, 5.05, 0.1), 2))
    seed: int = 7


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 80_000
    gamma: float = 0.95
    learning_rate: float = 0.08
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay: float = 24_000.0
    log_every: int = 200
    state_bins: int = 10


def demand_and_reward(price_vector: np.ndarray, cfg: MarketConfig) -> tuple[np.ndarray, np.ndarray]:
    """Logit demand with an outside option, then profit rewards."""
    utilities = (cfg.value - cfg.alpha * price_vector) / cfg.mu
    exp_u = np.exp(utilities - np.max(np.r_[0.0, utilities]))
    outside = np.exp(0.0 - np.max(np.r_[0.0, utilities]))
    shares = exp_u / (outside + exp_u.sum())
    demand = cfg.market_size * shares
    rewards = (price_vector - cfg.cost) * demand
    return demand, rewards


def find_symmetric_collusive_price(cfg: MarketConfig) -> tuple[float, float]:
    best_price = cfg.prices[0]
    best_profit = -np.inf
    for price in cfg.prices:
        p = np.full(cfg.n_agents, price)
        _, rewards = demand_and_reward(p, cfg)
        total_profit = rewards.sum()
        if total_profit > best_profit:
            best_profit = total_profit
            best_price = price
    return float(best_price), float(best_profit)


def find_pure_nash_prices(cfg: MarketConfig) -> tuple[np.ndarray, np.ndarray]:
    prices = np.array(cfg.prices)
    n_actions = len(prices)
    n = cfg.n_agents

    if n != 2:
        raise ValueError("Pure Nash search is implemented for the two-agent experiment.")

    rewards = np.zeros((n_actions, n_actions, n))
    for i, p0 in enumerate(prices):
        for j, p1 in enumerate(prices):
            _, r = demand_and_reward(np.array([p0, p1]), cfg)
            rewards[i, j] = r

    equilibria: list[tuple[int, int]] = []
    for i in range(n_actions):
        for j in range(n_actions):
            best0 = rewards[:, j, 0].max()
            best1 = rewards[i, :, 1].max()
            if np.isclose(rewards[i, j, 0], best0) and np.isclose(rewards[i, j, 1], best1):
                equilibria.append((i, j))

    if not equilibria:
        raise RuntimeError("No pure Nash equilibrium found on the price grid.")

    chosen = min(equilibria, key=lambda ij: abs(prices[ij[0]] - prices[ij[1]]))
    p = np.array([prices[chosen[0]], prices[chosen[1]]], dtype=float)
    _, r = demand_and_reward(p, cfg)
    return p, r


def _reward_bin(value: float, cfg: MarketConfig, train_cfg: TrainConfig) -> int:
    max_reward = (max(cfg.prices) - cfg.cost) * cfg.market_size
    scaled = np.clip(value / max_reward, 0.0, 0.999999)
    return int(scaled * train_cfg.state_bins)


def encode_observation(
    agent_id: int,
    last_actions: np.ndarray,
    last_demands: np.ndarray,
    last_rewards: np.ndarray,
    mode: ObservationMode,
    cfg: MarketConfig,
    train_cfg: TrainConfig,
) -> tuple[int, ...]:
    demand_bins = np.clip((last_demands / cfg.market_size * train_cfg.state_bins).astype(int), 0, train_cfg.state_bins - 1)
    reward_bins = np.array([_reward_bin(r, cfg, train_cfg) for r in last_rewards], dtype=int)

    if mode == "private":
        i = agent_id
        return (int(last_actions[i]), int(demand_bins[i]), int(reward_bins[i]))

    obs: list[int] = []
    obs.extend(int(x) for x in last_actions)
    obs.extend(int(x) for x in demand_bins)
    obs.extend(int(x) for x in reward_bins)
    return tuple(obs)


def train_agents(mode: ObservationMode, market_cfg: MarketConfig, train_cfg: TrainConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(market_cfg.seed + (0 if mode == "private" else 10_000))
    prices = np.array(market_cfg.prices)
    n_actions = len(prices)
    n = market_cfg.n_agents
    q_tables: list[dict[tuple[int, ...], np.ndarray]] = [dict() for _ in range(n)]

    last_actions = rng.integers(0, n_actions, size=n)
    last_demands, last_rewards = demand_and_reward(prices[last_actions], market_cfg)

    logs = []
    action_history = []
    reward_history = []

    for t in range(train_cfg.episodes):
        epsilon = train_cfg.epsilon_end + (train_cfg.epsilon_start - train_cfg.epsilon_end) * np.exp(
            -t / train_cfg.epsilon_decay
        )

        states = [
            encode_observation(i, last_actions, last_demands, last_rewards, mode, market_cfg, train_cfg)
            for i in range(n)
        ]

        actions = np.zeros(n, dtype=int)
        for i, state in enumerate(states):
            q = q_tables[i].setdefault(state, np.zeros(n_actions, dtype=float))
            if rng.random() < epsilon:
                actions[i] = rng.integers(0, n_actions)
            else:
                actions[i] = int(rng.choice(np.flatnonzero(q == q.max())))

        demands, rewards = demand_and_reward(prices[actions], market_cfg)
        next_states = [
            encode_observation(i, actions, demands, rewards, mode, market_cfg, train_cfg)
            for i in range(n)
        ]

        for i, state in enumerate(states):
            q = q_tables[i].setdefault(state, np.zeros(n_actions, dtype=float))
            next_q = q_tables[i].setdefault(next_states[i], np.zeros(n_actions, dtype=float))
            target = rewards[i] + train_cfg.gamma * next_q.max()
            q[actions[i]] += train_cfg.learning_rate * (target - q[actions[i]])

        last_actions = actions
        last_demands = demands
        last_rewards = rewards
        action_history.append(actions.copy())
        reward_history.append(rewards.copy())

        if (t + 1) % train_cfg.log_every == 0:
            recent_actions = np.array(action_history[-train_cfg.log_every :])
            recent_rewards = np.array(reward_history[-train_cfg.log_every :])
            logs.append(
                [
                    t + 1,
                    float(epsilon),
                    float(prices[recent_actions].mean()),
                    float(recent_rewards.mean()),
                    float((recent_actions[:, 0] == recent_actions[:, 1]).mean()),
                ]
            )

    actions_arr = np.array(action_history)
    rewards_arr = np.array(reward_history)
    return {
        "logs": np.array(logs),
        "actions": actions_arr,
        "prices": prices[actions_arr],
        "rewards": rewards_arr,
    }


def save_results(results: dict[str, dict[str, np.ndarray]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode, data in results.items():
        np.savez_compressed(out_dir / f"{mode}.npz", **data)
