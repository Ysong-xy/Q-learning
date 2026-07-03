from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plot_results import plot_experiment
from plot_generalized_results import plot_generalized_results
from repeated_market_rl import price_vector
from repeated_market_rl import RepeatedMarketConfig, RepeatedTrainConfig, benchmarks, save_results, train


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    market_cfg: RepeatedMarketConfig
    train_cfg: RepeatedTrainConfig

    @property
    def prices(self) -> tuple[float, ...]:
        return self.market_cfg.price_levels


def build_scenarios() -> list[Scenario]:
    shared_initial_actions = (0, 1)
    shared_market = {
        "n_market_makers": 2,
        "price_levels": (2.0, 2.6),
        "initial_actions": shared_initial_actions,
        "seed": 2,
        "demand_all_low": 410.0,
        "demand_all_high": 312.5,
        "demand_low_when_undercutting": 560.0,
        "demand_high_when_undercut": 125.0,
    }
    return [
        Scenario(
            key="legacy_gamma_099",
            label="Original initialization, gamma=0.99",
            market_cfg=RepeatedMarketConfig(**shared_market),
            train_cfg=RepeatedTrainConfig(
                gamma=0.99,
                repeat_last_prior=1_000.0,
                q_initialization="legacy_consensus_high",
            ),
        ),
        Scenario(
            key="legacy_gamma_001",
            label="Original initialization, gamma=0.01",
            market_cfg=RepeatedMarketConfig(**shared_market),
            train_cfg=RepeatedTrainConfig(
                gamma=0.01,
                repeat_last_prior=1_000.0,
                q_initialization="legacy_consensus_high",
            ),
        ),
    ]


def main() -> None:
    root = Path(__file__).resolve().parent
    result_dir = root / "results"
    figure_dir = root / "figures"

    scenarios = build_scenarios()
    results = {}
    plot_specs = []

    print("Transparent two-seller pricing experiment")
    for scenario in scenarios:
        market_cfg = scenario.market_cfg
        train_cfg = scenario.train_cfg
        bench = benchmarks(market_cfg)
        initial_actions = np.array(market_cfg.initial_actions)
        initial_prices = price_vector(initial_actions, market_cfg)

        print(f"\n{scenario.label}")
        print(f"  Market makers: {market_cfg.n_market_makers}")
        print(f"  Price levels: {list(market_cfg.price_levels)}")
        print(f"  Gamma: {train_cfg.gamma:.2f}")
        print(f"  Shared initial actions: {initial_actions.tolist()}")
        print(f"  Initial average price: {initial_prices.mean():.2f}")
        print(f"  Low-price profit: {bench['nash_profit']:.2f} at price {bench['nash_price']:.2f}")
        print(f"  High-price joint profit: {bench['collusive_profit']:.2f} at price {bench['collusive_price']:.2f}")
        print(
            "  Mixed-price profits: "
            f"low seller={(market_cfg.low_price - market_cfg.cost) * market_cfg.demand_low_when_undercutting:.2f}, "
            f"high seller={(market_cfg.high_price - market_cfg.cost) * market_cfg.demand_high_when_undercut:.2f}"
        )

        data = train(market_cfg, train_cfg)
        results[scenario.key] = data
        plot_specs.append(
            {
                "key": scenario.key,
                "label": scenario.label,
                "gamma": train_cfg.gamma,
                "nash_price": bench["nash_price"],
                "collusive_price": bench["collusive_price"],
                "payoff_low_low": bench["nash_profit"],
                "payoff_high_high": bench["collusive_profit"],
                "payoff_low_high": (market_cfg.low_price - market_cfg.cost)
                * market_cfg.demand_low_when_undercutting,
                "payoff_high_low": (market_cfg.high_price - market_cfg.cost)
                * market_cfg.demand_high_when_undercut,
            }
        )

        final_actions = data["actions"][-5000:]
        final_price = data["prices"][-5000:].mean()
        final_profit = data["rewards"][-5000:].mean()
        high_high = (final_actions == market_cfg.high_action).all(axis=1).mean()
        low_low = (final_actions == market_cfg.low_action).all(axis=1).mean()
        print(
            f"  result: final price={final_price:.3f}, final profit={final_profit:.2f}, "
            f"all-high={high_high:.3f}, all-low={low_low:.3f}"
        )

    save_results(results, result_dir)
    plot_experiment(result_dir, figure_dir, plot_specs)
    plot_generalized_results(results, scenarios, figure_dir / "two_price_dynamics")

    print(f"Saved figures to: {figure_dir}")


if __name__ == "__main__":
    main()
