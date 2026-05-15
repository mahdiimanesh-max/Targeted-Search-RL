#!/usr/bin/env python3
"""Plot toy PrefixIG RLVR learning curves from CSV."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/atgpo_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "reward_tpo": "Reward-TPO",
    "prefixig_grpo": "PrefixIG-GRPO",
    "prefixig_grpo_turn_norm": "PrefixIG-GRPO+TurnNorm",
    "prefixig_grpo_turn_norm_vr": "PrefixIG-GRPO+TurnNorm+VR",
    "prefixig_atgpo": "A-TGPO",
    "prefixig_tpo": "PrefixIG-TPO",
    "prefixig_tpo_rg_eff": "PrefixIG-TPO+RG-Eff",
}

COLORS = {
    "reward_tpo": "#4C78A8",
    "prefixig_grpo": "#9D755D",
    "prefixig_grpo_turn_norm": "#E45756",
    "prefixig_grpo_turn_norm_vr": "#72B7B2",
    "prefixig_atgpo": "#F58518",
    "prefixig_tpo": "#54A24B",
    "prefixig_tpo_rg_eff": "#B279A2",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate(rows: list[dict[str, str]], metric: str):
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["method"]][int(row["episode"])].append(float(row[metric]))

    out = {}
    for method, by_episode in grouped.items():
        episodes = np.array(sorted(by_episode), dtype=np.int32)
        means = np.array([np.mean(by_episode[int(ep)]) for ep in episodes], dtype=np.float32)
        stds = np.array([np.std(by_episode[int(ep)]) for ep in episodes], dtype=np.float32)
        out[method] = (episodes, means, stds)
    return out


def plot_metric(rows, metric: str, ylabel: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = aggregate(rows, metric)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    for method, (episodes, means, stds) in data.items():
        label = DISPLAY_NAMES.get(method, method)
        color = COLORS.get(method)
        ax.plot(episodes, means, label=label, color=color, linewidth=2.2)
        ax.fill_between(
            episodes,
            means - stds,
            means + stds,
            color=color,
            alpha=0.16,
            linewidth=0,
        )

    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.08 if metric == "useful_minus_redundant" else 0.0, 1.02)
    ax.grid(True, color="#D8DEE9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_combined(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("reward", "Final Reward"),
        ("useful_correct", "Useful Correct"),
        ("redundant_correct", "Redundant Correct"),
        ("useful_minus_redundant", "Useful - Redundant"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.3), dpi=180)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        data = aggregate(rows, metric)
        for method, (episodes, means, stds) in data.items():
            label = DISPLAY_NAMES.get(method, method)
            color = COLORS.get(method)
            ax.plot(episodes, means, label=label, color=color, linewidth=2.0)
            ax.fill_between(
                episodes,
                means - stds,
                means + stds,
                color=color,
                alpha=0.13,
                linewidth=0,
            )
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylim(-0.08 if metric == "useful_minus_redundant" else 0.0, 1.02)
        ax.grid(True, color="#D8DEE9", linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="outputs/toy_prefixig_rlvr_multiseed.csv")
    parser.add_argument("--output-dir", default="outputs/figures")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(Path(args.input_csv))
    out = Path(args.output_dir)

    plot_metric(rows, "reward", "Final reward", out / "toy_rlvr_reward.png")
    plot_metric(rows, "useful_correct", "Useful correct rate", out / "toy_rlvr_useful_correct.png")
    plot_metric(rows, "redundant_correct", "Redundant correct rate", out / "toy_rlvr_redundant_correct.png")
    plot_metric(
        rows,
        "useful_minus_redundant",
        "Useful correct - redundant correct",
        out / "toy_rlvr_useful_minus_redundant.png",
    )
    plot_combined(rows, out / "toy_rlvr_combined.png")

    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
