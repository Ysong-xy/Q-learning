from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _block_mean(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    block = max(200, len(values) // 300)
    n = len(values) // block
    values = values[: n * block]
    return np.arange(1, n + 1) * block, values.reshape(n, block).mean(axis=1)


def plot_market_sweep(
    market_key: str,
    market_label: str,
    gammas: tuple[float, ...],
    result_dir: Path,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(gammas)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    final_price, final_profit, final_same = [], [], []

    for gamma, color in zip(gammas, colors):
        with np.load(result_dir / f"gamma_{gamma:.2f}.npz") as data:
            mean_price = data["prices"].mean(axis=1)
            mean_profit = data["rewards"].mean(axis=1)
            mean_demand = data["demands"].mean(axis=1)
            same = np.all(data["actions"] == data["actions"][:, :1], axis=1).astype(float)
            label = rf"$\gamma={gamma:.2f}$"
            for ax, values in zip(axes.ravel(), (mean_price, mean_profit, mean_demand, same)):
                x, y = _block_mean(values)
                ax.plot(x, y, color=color, linewidth=1.5, label=label)
            tail = slice(-50_000, None)
            final_price.append(float(mean_price[tail].mean()))
            final_profit.append(float(mean_profit[tail].mean()))
            final_same.append(float(same[tail].mean()))

    titles = ("Average price", "Average profit per seller", "Average demand per seller", "All sellers choose same price")
    ylabels = ("Price", "Profit", "Demand", "Rate")
    for ax, title, ylabel in zip(axes.ravel(), titles, ylabels):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
    axes[1, 0].set_xlabel("Training round")
    axes[1, 1].set_xlabel("Training round")
    fig.suptitle(f"Gamma sweep: {market_label}")
    fig.tight_layout()
    fig.savefig(figure_dir / "training_curves_by_gamma.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, values, title, ylabel in zip(
        axes,
        (final_price, final_profit, final_same),
        ("Final average price", "Final average profit", "Final same-price rate"),
        ("Price", "Profit", "Rate"),
    ):
        ax.plot(gammas, values, marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Final 50,000 rounds: {market_label}")
    fig.tight_layout()
    fig.savefig(figure_dir / "final_metrics_by_gamma.png", dpi=180)
    plt.close(fig)
