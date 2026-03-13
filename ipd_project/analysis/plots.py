"""Plot utilities for offline IPD experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def load_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_core_experiments(results: dict, out_dir: Path) -> None:
    for exp_name, payload in results.get("experiments", {}).items():
        series = payload.get("series", [])
        if not series:
            continue
        df = pd.DataFrame(series)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(df["episode"], df["cooperation"], color="#2ec4b6", linewidth=1.6)
        axes[0].set_title(f"{exp_name}: Cooperation")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Cooperation Rate")

        axes[1].plot(df["episode"], df["joint_reward"], color="#f2b65c", linewidth=1.6)
        axes[1].set_title(f"{exp_name}: Joint Reward")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Average Joint Reward")

        fig.tight_layout()
        fig.savefig(out_dir / f"{exp_name.lower()}_timeseries.png", dpi=180)
        plt.close(fig)


def plot_tournament(results: dict, out_dir: Path) -> None:
    rows = results.get("tournament", {}).get("leaderboard", [])
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("mean_reward", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(df["strategy"], df["mean_reward"], color="#6c8bff")
    ax.set_title("Baseline Tournament: Mean Reward")
    ax.set_xlabel("Mean Reward")
    ax.set_ylabel("Strategy")
    fig.tight_layout()
    fig.savefig(out_dir / "tournament_leaderboard.png", dpi=180)
    plt.close(fig)


def plot_sweeps(results: dict, out_dir: Path) -> None:
    sweeps = results.get("sweeps", {})
    if not sweeps:
        return

    for key, rows in sweeps.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(df["value"], df["tail_cooperation"], marker="o", color="#55d187")
        axes[0].set_title(f"{key} sweep: Tail Cooperation")
        axes[0].set_xlabel(key)
        axes[0].set_ylabel("Tail Cooperation")

        axes[1].plot(df["value"], df["tail_reward"], marker="o", color="#ff6b6b")
        axes[1].set_title(f"{key} sweep: Tail Reward")
        axes[1].set_xlabel(key)
        axes[1].set_ylabel("Tail Reward")

        fig.tight_layout()
        fig.savefig(out_dir / f"sweep_{key}.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots from offline IPD results JSON.")
    parser.add_argument("--input", required=True, type=str, help="Path to results JSON")
    parser.add_argument("--out-dir", default="reports/figures", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(input_path)
    plot_core_experiments(results, out_dir)
    plot_tournament(results, out_dir)
    plot_sweeps(results, out_dir)

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
