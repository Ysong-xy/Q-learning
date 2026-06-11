from __future__ import annotations

from pathlib import Path

import numpy as np

from plot_results import plot_experiment
from repeated_market_rl import price_vector
from repeated_market_rl import RepeatedMarketConfig, RepeatedTrainConfig, benchmarks, save_results, train


def main() -> None:
    root = Path(__file__).resolve().parent
    result_dir = root / "results"
    figure_dir = root / "figures"

    market_cfg = RepeatedMarketConfig()
    train_cfg = RepeatedTrainConfig()

    bench = benchmarks(market_cfg)

    print("Market benchmark")
    print(f"  Market makers: {market_cfg.n_market_makers}")
    initial_actions = np.array(market_cfg.initial_actions)
    initial_prices = price_vector(initial_actions, market_cfg)
    print(f"  Shared initial actions: {initial_actions.tolist()}")
    print(f"  Shared initial average price: {initial_prices.mean():.2f}")
    print(f"  One-shot Nash price: {bench['nash_price']:.2f}, per-agent profit: {bench['nash_profit']:.2f}")
    print(
        f"  Symmetric joint-profit price: {bench['collusive_price']:.2f}, "
        f"per-agent profit: {bench['collusive_profit']:.2f}"
    )

    results = {
        "private": train("private", market_cfg, train_cfg),
        "transparent": train("transparent", market_cfg, train_cfg),
    }
    save_results(results, result_dir)
    plot_experiment(result_dir, figure_dir, bench["nash_price"], bench["collusive_price"])

    for mode, data in results.items():
        final_price = data["prices"][-5000:].mean()
        final_profit = data["rewards"][-5000:].mean()
        high_high = (data["actions"][-5000:] == 1).all(axis=1).mean()
        low_low = (data["actions"][-5000:] == 0).all(axis=1).mean()
        print(
            f"{mode:>12}: final price={final_price:.3f}, final profit={final_profit:.2f}, "
            f"all-high={high_high:.3f}, all-low={low_low:.3f}"
        )

    print(f"Saved figures to: {figure_dir}")


if __name__ == "__main__":
    main()
