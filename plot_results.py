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


def plot_experiment(result_dir: Path, figure_dir: Path, nash_price: float, collusive_price: float) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    private = np.load(result_dir / "private.npz")
    transparent = np.load(result_dir / "transparent.npz")

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, data, color in [
        ("Private observation", private, "#2b6cb0"),
        ("Transparent market", transparent, "#c2410c"),
    ]:
        logs = data["logs"]
        ax.plot(logs[:, 0], logs[:, 2], label=label, color=color, linewidth=2)
    ax.axhline(nash_price, color="#222222", linestyle="--", linewidth=1.6, label=f"Nash price = {nash_price:.2f}")
    ax.axhline(
        collusive_price,
        color="#6b21a8",
        linestyle=":",
        linewidth=2,
        label=f"Joint-profit price = {collusive_price:.2f}",
    )
    ax.set_title("Average Transaction Price During Learning")
    ax.set_xlabel("Training round")
    ax.set_ylabel("Average price")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "training_price_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Private\nobservation", "Transparent\nmarket"]
    final_prices = [private["prices"][-5000:].mean(), transparent["prices"][-5000:].mean()]
    final_rewards = [private["rewards"][-5000:].mean(), transparent["rewards"][-5000:].mean()]
    x = np.arange(len(labels))
    bars = ax.bar(x, final_prices, color=["#2b6cb0", "#c2410c"], width=0.55)
    ax.axhline(nash_price, color="#222222", linestyle="--", linewidth=1.6, label="Nash")
    ax.axhline(collusive_price, color="#6b21a8", linestyle=":", linewidth=2, label="Joint-profit")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Final average price")
    ax.set_title("Final 5,000 Rounds: Price Level")
    for bar, value in zip(bars, final_prices):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.2f}", ha="center", va="bottom")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "final_price_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, final_rewards, color=["#2b6cb0", "#c2410c"], width=0.55)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Average profit per agent")
    ax.set_title("Final 5,000 Rounds: Profit")
    for bar, value in zip(bars, final_rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value:.1f}", ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "final_profit_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0.95, 5.16, 0.1)
    ax.hist(private["prices"][-5000:].ravel(), bins=bins, alpha=0.65, label="Private observation", color="#2b6cb0")
    ax.hist(transparent["prices"][-5000:].ravel(), bins=bins, alpha=0.65, label="Transparent market", color="#c2410c")
    ax.axvline(nash_price, color="#222222", linestyle="--", linewidth=1.6)
    ax.axvline(collusive_price, color="#6b21a8", linestyle=":", linewidth=2)
    ax.set_title("Final 5,000 Rounds: Price Distribution")
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "final_price_distribution.png", dpi=180)
    plt.close(fig)
