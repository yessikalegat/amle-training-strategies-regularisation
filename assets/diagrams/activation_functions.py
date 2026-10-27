"""Draw assets/activation_functions.png: sigmoid, tanh, ReLU and ELU with their slopes, for notebook 01.

Run from the repository root: uv run python assets/diagrams/activation_functions.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent.parent / "activation_functions.png"

INK, GREY, BLUE, AMBER = "#222222", "#6b6b6b", "#2f6fb0", "#c47612"
z = np.linspace(-4, 4, 801)
sigmoid = 1 / (1 + np.exp(-z))
tanh = np.tanh(z)
relu = np.maximum(0, z)
elu = np.where(z > 0, z, np.exp(z) - 1)

FUNCTIONS = [
    ("Sigmoid", sigmoid, sigmoid * (1 - sigmoid), "slope at most 0.25"),
    ("Tanh", tanh, 1 - tanh**2, "slope at most 1"),
    ("ReLU", relu, (z > 0).astype(float), "slope 1 when active, 0 otherwise"),
    ("ELU", elu, np.where(z > 0, 1.0, np.exp(z)), "slope 1 when positive, small when negative"),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), dpi=200, sharey=True)
fig.patch.set_facecolor("white")
for ax, (name, values, slope, note) in zip(axes, FUNCTIONS):
    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.axvline(0, color=GREY, linewidth=0.8)
    ax.plot(z, values, color=BLUE, linewidth=2.4, label="function")
    ax.plot(z, slope, color=AMBER, linewidth=2, linestyle="--", label="slope")
    ax.set_title(name, fontsize=14, color=INK)
    ax.set_xlabel("weighted sum $z$", color=GREY)
    ax.text(0.5, -0.32, note, transform=ax.transAxes, fontsize=10, color=GREY, ha="center", va="top")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.3, 2.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
axes[0].legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(OUT, facecolor="white", bbox_inches="tight")
