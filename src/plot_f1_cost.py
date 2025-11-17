"""
Generate a scatter plot comparing F1 score versus cost for different
interfaces and language models.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


DATA = [
    {"interface": "RAG", "model": "GPT-4.1", "f1": 0.77, "cost": 0.03},
    {"interface": "RAG", "model": "GPT-5", "f1": 0.87, "cost": 0.14},
    {"interface": "RAG", "model": "GPT-5-mini", "f1": 0.79, "cost": 0.01},
    {"interface": "API MCP", "model": "GPT-4.1", "f1": 0.68, "cost": 0.07},
    {"interface": "API MCP", "model": "GPT-5", "f1": 0.82, "cost": 0.22},
    {"interface": "API MCP", "model": "GPT-5-mini", "f1": 0.77, "cost": 0.04},
    {"interface": "NLWeb", "model": "GPT-4.1", "f1": 0.69, "cost": 0.07},
    {"interface": "NLWeb", "model": "GPT-5", "f1": 0.85, "cost": 0.12},
    {"interface": "NLWeb", "model": "GPT-5-mini", "f1": 0.76, "cost": 0.04},
]

INTERFACE_COLORS = {
    "RAG": "#2ca02c",  # green
    "API MCP": "#ff7f0e",  # orange
    "NLWeb": "#d62728",  # red
}

MODEL_MARKERS = {
    "GPT-4.1": "o",
    "GPT-5": "s",
    "GPT-5-mini": "^",
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for entry in DATA:
        ax.scatter(
            entry["cost"],
            entry["f1"],
            color=INTERFACE_COLORS[entry["interface"]],
            marker=MODEL_MARKERS[entry["model"]],
            s=140,
            edgecolor="black",
            linewidths=0.6,
            alpha=0.9,
        )

    ax.set_title("F1 Score vs Cost by Interface and Model")
    ax.set_xlabel("Cost ($)")
    ax.set_ylabel("F1 Score")
    ax.set_xlim(0, max(d["cost"] for d in DATA) * 1.2)
    ax.set_ylim(0.6, 0.9)
    ax.grid(True, linewidth=0.5, alpha=0.3)

    interface_handles = [
        Patch(color=color, label=name) for name, color in INTERFACE_COLORS.items()
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="black",
            markersize=9,
            linestyle="",
            label=name,
        )
        for name, marker in MODEL_MARKERS.items()
    ]

    ax.legend(
        handles=interface_handles + model_handles,
        title="Interface / Model",
        loc="lower right",
        ncol=2,
        frameon=True,
    )

    plt.tight_layout()

    output_path = Path("results") / "f1_score_vs_cost.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
