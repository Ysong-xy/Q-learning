from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from repeated_market_rl import RepeatedMarketConfig, RepeatedTrainConfig, save_results, train
from plot_generalized_results import plot_generalized_results


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    agents: int
    prices: tuple[float, ...]
    seed: int


def build_scenarios() -> list[Scenario]:
    """The three requested market sizes, with all else held fixed."""
    multi_prices = tuple(float(x) for x in np.round(np.arange(2.0, 2.61, 0.1), 2))
    return [
        Scenario("many_agents_two_prices", "多人两价", 4, (2.0, 2.6), 11),
        Scenario("two_agents_many_prices", "两人多价", 2, multi_prices, 12),
        Scenario("many_agents_many_prices", "多人多价", 4, multi_prices, 13),
    ]


def market_config(scenario: Scenario) -> RepeatedMarketConfig:
    return RepeatedMarketConfig(
        n_market_makers=scenario.agents,
        price_levels=scenario.prices,
        initial_actions=tuple(i % len(scenario.prices) for i in range(scenario.agents)),
        seed=scenario.seed,
        demand_model="logit",
        market_size=1_000.0,
        consumer_value=5.0,
        price_sensitivity=1.25,
        differentiation=0.55,
        cost=1.0,
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "results" / "generalized"
    train_cfg = RepeatedTrainConfig(
        episodes=1_000_000,
        gamma=0.95,
        learning_rate_positive=0.05,
        learning_rate_negative=0.01,
        epsilon_start=0.20,
        epsilon_end=0.001,
        epsilon_decay=80_000.0,
        repeat_last_prior=0.0,
        state_bins=8,
        q_initialization="zero",
    )
    results: dict[str, dict[str, np.ndarray]] = {}

    for scenario in build_scenarios():
        cfg = market_config(scenario)
        data = train(cfg, train_cfg)
        results[scenario.key] = data
        window = 50_000
        tail = slice(-window, None)
        previous = slice(-2 * window, -window)
        coordination = np.all(data["actions"][tail] == data["actions"][tail, :1], axis=1).mean()
        price_change = abs(data["prices"][tail].mean() - data["prices"][previous].mean())
        profit_change = abs(data["rewards"][tail].mean() - data["rewards"][previous].mean())
        tail_distribution = np.bincount(data["actions"][tail].ravel(), minlength=cfg.n_actions) / data["actions"][tail].size
        previous_distribution = np.bincount(data["actions"][previous].ravel(), minlength=cfg.n_actions) / data["actions"][previous].size
        action_tv = 0.5 * np.abs(tail_distribution - previous_distribution).sum()
        empirically_converged = price_change < 0.01 and action_tv < 0.03
        print(f"{scenario.label}: N={cfg.n_market_makers}, K={cfg.n_actions}")
        print(f"  prices={cfg.price_levels}")
        print(f"  final mean price={data['prices'][tail].mean():.3f}")
        print(f"  final mean profit per seller={data['rewards'][tail].mean():.3f}")
        print(f"  same-price rate={coordination:.3f}")
        print(f"  adjacent-window price change={price_change:.4f}")
        print(f"  adjacent-window profit change={profit_change:.3f}")
        print(f"  action-distribution TV change={action_tv:.4f}")
        print(f"  empirical convergence check={'PASS' if empirically_converged else 'FAIL'}")

    save_results(results, out_dir)
    plot_generalized_results(results, build_scenarios(), Path(__file__).resolve().parent / "figures" / "generalized")
    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
