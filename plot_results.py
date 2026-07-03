from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def _short_label(label: object) -> str:
    return str(label).replace(" transparent market", "").replace("-convergence", "").title()


def _scenario_box(scenario: dict[str, object]) -> str:
    return (
        f"gamma={float(scenario['gamma']):.2f}\n"
        f"LL={float(scenario['payoff_low_low']):.0f}, "
        f"HL={float(scenario['payoff_high_low']):.0f}\n"
        f"LH={float(scenario['payoff_low_high']):.0f}, "
        f"HH={float(scenario['payoff_high_high']):.0f}"
    )


def plot_experiment(result_dir: Path, figure_dir: Path, scenarios: list[dict[str, object]]) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    loaded = [(scenario, np.load(result_dir / f"{scenario['key']}.npz")) for scenario in scenarios]
    colors = ["#2b6cb0", "#c2410c", "#047857", "#7c3aed"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, ((scenario, data), color) in enumerate(zip(loaded, colors)):
        logs = data["logs"]
        ax.plot(
            logs[:, 0],
            logs[:, 2],
            label=f"{_short_label(scenario['label'])} (gamma={float(scenario['gamma']):.2f})",
            color=color,
            linewidth=2,
        )
        ax.axhline(float(scenario["nash_price"]), color=color, linestyle="--", linewidth=1, alpha=0.45)
        ax.axhline(float(scenario["collusive_price"]), color=color, linestyle=":", linewidth=1.4, alpha=0.6)
        final_x = logs[-1, 0]
        final_y = logs[-1, 2]
        text_x = final_x * 0.58
        text_y = final_y + (0.08 if idx == 0 else 0.12)
        ax.annotate(
            _scenario_box(scenario),
            xy=(final_x, final_y),
            xytext=(text_x, text_y),
            color=color,
            fontsize=8.5,
            ha="left",
            va="center",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": color, "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.1},
        )
    ax.set_title("Average Transaction Price During Learning, Transparent Markets")
    ax.set_xlabel("Training round")
    ax.set_ylabel("Average price")
    ax.set_ylim(1.9, 2.78)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "training_price_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [
        f"{_short_label(scenario['label'])}\ngamma={float(scenario['gamma']):.2f}"
        for scenario, _ in loaded
    ]
    final_prices = [data["prices"][-5000:].mean() for _, data in loaded]
    final_rewards = [data["rewards"][-5000:].mean() for _, data in loaded]
    x = np.arange(len(labels))
    bars = ax.bar(x, final_prices, color=colors[: len(labels)], width=0.55)
    for (scenario, _), color in zip(loaded, colors):
        ax.axhline(float(scenario["nash_price"]), color=color, linestyle="--", linewidth=1, alpha=0.45)
        ax.axhline(float(scenario["collusive_price"]), color=color, linestyle=":", linewidth=1.4, alpha=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Final average price")
    ax.set_title("Final 5,000 Rounds: Price Level")
    ax.set_ylim(1.8, 2.9)
    for bar, value in zip(bars, final_prices):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.2f}", ha="center", va="bottom")
    for bar, (scenario, _), color in zip(bars, loaded, colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            1.86,
            _scenario_box(scenario),
            ha="center",
            va="bottom",
            fontsize=8.3,
            color=color,
            bbox={"boxstyle": "round,pad=0.32", "fc": "white", "ec": color, "alpha": 0.9},
        )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "final_price_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(x, final_rewards, color=colors[: len(labels)], width=0.55)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Average profit per agent")
    ax.set_title("Final 5,000 Rounds: Profit")
    ax.set_ylim(0, max(final_rewards) * 1.28)
    for bar, value in zip(bars, final_rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value:.1f}", ha="center", va="bottom")
    for bar, (scenario, _), color in zip(bars, loaded, colors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 0.54,
            _scenario_box(scenario),
            ha="center",
            va="center",
            fontsize=8.3,
            color=color,
            bbox={"boxstyle": "round,pad=0.32", "fc": "white", "ec": color, "alpha": 0.9},
        )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "final_profit_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    prices = np.concatenate([data["prices"][-5000:].ravel() for _, data in loaded])
    bin_width = 0.05
    bins = np.arange(prices.min() - bin_width, prices.max() + 2 * bin_width, bin_width)
    hist_peaks = []
    for (scenario, data), color in zip(loaded, colors):
        counts, _, _ = ax.hist(
            data["prices"][-5000:].ravel(),
            bins=bins,
            alpha=0.6,
            label=f"{_short_label(scenario['label'])} (gamma={float(scenario['gamma']):.2f})",
            color=color,
        )
        hist_peaks.append(float(counts.max()))
        ax.axvline(float(scenario["nash_price"]), color=color, linestyle="--", linewidth=1, alpha=0.45)
        ax.axvline(float(scenario["collusive_price"]), color=color, linestyle=":", linewidth=1.4, alpha=0.6)
    ax.set_title("Final 5,000 Rounds: Price Distribution")
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    peak_y = max(hist_peaks) if hist_peaks else 1.0
    for idx, ((scenario, data), color) in enumerate(zip(loaded, colors)):
        mean_price = float(data["prices"][-5000:].mean())
        y = peak_y * (0.72 if idx == 0 else 0.42)
        ax.annotate(
            _scenario_box(scenario),
            xy=(mean_price, y),
            xytext=(mean_price + (0.12 if idx == 1 else -0.52), y),
            fontsize=8.3,
            color=color,
            ha="left",
            va="center",
            bbox={"boxstyle": "round,pad=0.32", "fc": "white", "ec": color, "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
        )
    fig.tight_layout()
    fig.savefig(figure_dir / "final_price_distribution.png", dpi=180)
    plt.close(fig)
