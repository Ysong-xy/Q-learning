from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plot_gamma_sweep import plot_market_sweep
from repeated_market_rl import RepeatedMarketConfig, RepeatedTrainConfig, train


GAMMAS = (0.01, 0.30, 0.60, 0.95, 0.99)


@dataclass(frozen=True)
class MarketCase:
    key: str
    label: str
    agents: int
    prices: tuple[float, ...]
    seed: int


def market_cases() -> tuple[MarketCase, ...]:
    many_prices = tuple(float(x) for x in np.round(np.arange(2.0, 2.61, 0.1), 2))
    return (
        MarketCase("many_agents_two_prices", "Four sellers, two prices", 4, (2.0, 2.6), 11),
        MarketCase("two_agents_many_prices", "Two sellers, seven prices", 2, many_prices, 12),
        MarketCase("many_agents_many_prices", "Four sellers, seven prices", 4, many_prices, 13),
    )


def _run_one(case: MarketCase, gamma: float, episodes: int, root: Path) -> dict[str, object]:
    market_cfg = RepeatedMarketConfig(
        n_market_makers=case.agents,
        price_levels=case.prices,
        initial_actions=tuple(i % len(case.prices) for i in range(case.agents)),
        seed=case.seed,
        demand_model="logit",
        market_size=1_000.0,
        consumer_value=5.0,
        price_sensitivity=1.25,
        differentiation=0.55,
        cost=1.0,
    )
    train_cfg = RepeatedTrainConfig(
        episodes=episodes,
        gamma=gamma,
        learning_rate_positive=0.05,
        learning_rate_negative=0.01,
        epsilon_start=0.20,
        epsilon_end=0.001,
        epsilon_decay=80_000.0,
        repeat_last_prior=0.0,
        state_bins=8,
        q_initialization="zero",
    )
    data = train(market_cfg, train_cfg)
    out_dir = root / "results" / "gamma_sweep" / case.key
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"gamma_{gamma:.2f}.npz", **data)

    window = min(50_000, episodes // 4)
    tail, previous = slice(-window, None), slice(-2 * window, -window)
    price_change = abs(data["prices"][tail].mean() - data["prices"][previous].mean())
    profit_change = abs(data["rewards"][tail].mean() - data["rewards"][previous].mean())
    tail_dist = np.bincount(data["actions"][tail].ravel(), minlength=market_cfg.n_actions) / data["actions"][tail].size
    prev_dist = np.bincount(data["actions"][previous].ravel(), minlength=market_cfg.n_actions) / data["actions"][previous].size
    action_tv = 0.5 * np.abs(tail_dist - prev_dist).sum()
    same = np.all(data["actions"][tail] == data["actions"][tail, :1], axis=1).mean()
    return {
        "market": case.key,
        "gamma": gamma,
        "mean_price": float(data["prices"][tail].mean()),
        "mean_profit": float(data["rewards"][tail].mean()),
        "same_price_rate": float(same),
        "price_change": float(price_change),
        "profit_change": float(profit_change),
        "action_tv": float(action_tv),
        "converged": bool(price_change < 0.01 and action_tv < 0.03),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cases = market_cases()
    rows: list[dict[str, object]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_one, case, gamma, args.episodes, root): (case, gamma)
            for case in cases for gamma in GAMMAS
        }
        for future in as_completed(futures):
            case, gamma = futures[future]
            row = future.result()
            rows.append(row)
            print(f"finished {case.key}, gamma={gamma:.2f}, converged={row['converged']}", flush=True)

    rows.sort(key=lambda row: (str(row["market"]), float(row["gamma"])))
    summary = root / "results" / "gamma_sweep" / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for case in cases:
        plot_market_sweep(
            case.key,
            case.label,
            GAMMAS,
            root / "results" / "gamma_sweep" / case.key,
            root / "figures" / "gamma_sweep" / case.key,
        )
    print(f"Saved summary to: {summary}")


if __name__ == "__main__":
    main()
