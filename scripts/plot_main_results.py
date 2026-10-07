"""Regenerate the README comparison chart from the paper's main results table."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = ["HotpotQA", "2WikiMultiHopQA", "MuSiQue"]
QSR = {"GPT-4o-mini": [73.65, 74.19, 55.35],
       "Qwen3-8B": [72.25, 71.88, 48.13]}
BEST_BASELINE = {"GPT-4o-mini": [70.47, 72.08, 51.02],
                 "Qwen3-8B": [66.98, 69.99, 42.24]}


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, model in zip(axes, QSR):
        x = np.arange(len(LABELS))
        ax.bar(x - .18, BEST_BASELINE[model], width=.36,
               color="#b9c7d8", label="Best baseline")
        ax.bar(x + .18, QSR[model], width=.36,
               color="#315cc6", label="QSR-RAG")
        for i, value in enumerate(QSR[model]):
            ax.text(i + .18, value + .8, f"{value:.2f}", ha="center",
                    fontsize=9, fontweight="bold", color="#214595")
        ax.set_xticks(x, LABELS)
        ax.set_title(model, fontsize=13, fontweight="bold", pad=16)
        ax.set_ylim(0, 85)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel("Token-level F1 (%)")
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Main results · 500 questions per dataset · seed 43",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    target = Path("assets/main_results_f1.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, bbox_inches="tight", facecolor="white")
    print(target)


if __name__ == "__main__":
    main()
