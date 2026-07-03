from __future__ import annotations

from pathlib import Path
from typing import Protocol

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class ScenarioLike(Protocol):
    key: str
    label: str
    prices: tuple[float, ...]


def _label(scenario: ScenarioLike) -> str:
    labels = {
        "many_agents_two_prices": "Four sellers, two prices",
        "two_agents_many_prices": "Two sellers, seven prices",
        "many_agents_many_prices": "Four sellers, seven prices",
        "legacy_gamma_099": "Original initialization, gamma=0.99",
        "legacy_gamma_001": "Original initialization, gamma=0.01",
    }
    return labels.get(scenario.key, scenario.key.replace("_", " ").title())


def _block_mean(values: np.ndarray, block: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if block is None:
        block = max(200, len(values) // 300)
    n = len(values) // block
    trimmed = values[: n * block]
    return np.arange(1, n + 1) * block, trimmed.reshape(n, block, *values.shape[1:]).mean(axis=1)


def plot_generalized_results(
    results: dict[str, dict[str, np.ndarray]], scenarios: list[ScenarioLike], figure_dir: Path
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#2563eb", "#dc2626", "#059669"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for scenario, color in zip(scenarios, colors):
        data = results[scenario.key]
        x, mean_price = _block_mean(data["prices"].mean(axis=1))
        _, mean_profit = _block_mean(data["rewards"].mean(axis=1))
        _, mean_demand = _block_mean(data["demands"].mean(axis=1))
        same = np.all(data["actions"] == data["actions"][:, :1], axis=1).astype(float)
        _, same_rate = _block_mean(same)
        label = _label(scenario)
        axes[0, 0].plot(x, mean_price, color=color, label=label)
        axes[0, 1].plot(x, mean_profit, color=color, label=label)
        axes[1, 0].plot(x, mean_demand, color=color, label=label)
        axes[1, 1].plot(x, same_rate, color=color, label=label)

    titles = ("Average price", "Average profit per seller", "Average demand per seller", "All sellers choose same price")
    ylabels = ("Price", "Profit", "Demand", "Rate")
    for ax, title, ylabel in zip(axes.ravel(), titles, ylabels):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[1, 0].set_xlabel("Training round")
    axes[1, 1].set_xlabel("Training round")
    fig.tight_layout()
    fig.savefig(figure_dir / "all_metrics.png", dpi=180)
    plt.close(fig)

    for scenario, color in zip(scenarios, colors):
        actions = results[scenario.key]["actions"]
        x, frequencies = _block_mean(
            np.stack([(actions == action).mean(axis=1) for action in range(len(scenario.prices))], axis=1)
        )
        fig, ax = plt.subplots(figsize=(11, 6))
        palette = plt.cm.viridis(np.linspace(0.08, 0.92, len(scenario.prices)))
        for action, (price, action_color) in enumerate(zip(scenario.prices, palette)):
            ax.plot(x, frequencies[:, action], color=action_color, label=f"p={price:.1f}", linewidth=1.6)
        ax.set_title(f"Price-action frequencies: {_label(scenario)}")
        ax.set_xlabel("Training round")
        ax.set_ylabel("Fraction of sellers choosing action")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(ncol=min(5, len(scenario.prices)))
        fig.tight_layout()
        fig.savefig(figure_dir / f"{scenario.key}_action_frequencies.png", dpi=180)
        plt.close(fig)
