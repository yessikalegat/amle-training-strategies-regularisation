"""Draw assets/learning_curves.png and assets/learning_curves_quiz.png for notebook 02.

Both show six typical pairs of training and validation loss curves. The quiz version has letters only, so the
patterns can be discussed before the labelled version gives the diagnosis.

The curves are made up to show one pattern each; they are not results of a real training run.

Run from the repository root: uv run python assets/diagrams/learning_curves.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ASSETS = Path(__file__).parent.parent

INK, GREY, BLUE, AMBER = "#222222", "#6b6b6b", "#2f6fb0", "#c47612"
rng = np.random.default_rng(0)
epochs = np.arange(1, 101)


def decay(start, end, speed):
    """A loss that falls from start towards end."""
    return end + (start - end) * np.exp(-epochs / speed)


def noise(scale):
    return rng.normal(0, scale, size=epochs.size)


PANELS = [
    ("A", "Good fit", decay(1.0, 0.20, 12) + noise(0.004), decay(1.05, 0.24, 12) + noise(0.008),
     "both curves level off close together"),
    ("B", "Underfitting", decay(1.0, 0.62, 8) + noise(0.004), decay(1.02, 0.65, 8) + noise(0.008),
     "both curves level off at a high loss"),
    ("C", "Overfitting", decay(1.0, 0.05, 15) + noise(0.004),
     decay(1.05, 0.30, 10) + 0.0025 * np.maximum(0, epochs - 20) + noise(0.008),
     "training keeps falling, validation rises"),
    ("D", "Still improving", decay(1.0, 0.10, 90) + noise(0.004), decay(1.03, 0.13, 90) + noise(0.008),
     "both still falling when training stops"),
    ("E", "Learning rate too high", decay(1.0, 0.45, 4) + np.abs(noise(0.06)),
     decay(1.05, 0.50, 4) + np.abs(noise(0.14)), "large jumps from epoch to epoch"),
    ("F", "Unrepresentative validation set", decay(1.0, 0.25, 12) + noise(0.004),
     decay(0.9, 0.12, 12) + noise(0.03), "validation loss below training loss"),
]


def draw(labelled, out):
    """Draw the six panels, with titles and notes only when labelled is True."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=200, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, (letter, title, train, valid, note) in zip(axes.flat, PANELS):
        ax.plot(epochs, train, color=BLUE, linewidth=2.2, label="training loss")
        ax.plot(epochs, valid, color=AMBER, linewidth=2.2, label="validation loss")
        ax.set_title(f"{letter}. {title}" if labelled else letter, fontsize=13, color=INK, loc="left")
        if labelled:
            ax.text(0.97, 0.93, note, transform=ax.transAxes, fontsize=10, color=GREY, ha="right", va="top")
        ax.set_ylim(0, 1.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Epoch", color=GREY)
    for ax in axes[:, 0]:
        ax.set_ylabel("Loss", color=GREY)
    axes[0, 0].legend(loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)


draw(labelled=False, out=ASSETS / "learning_curves_quiz.png")
draw(labelled=True, out=ASSETS / "learning_curves.png")
