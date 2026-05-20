"""
Emotion score visualizations (radar / spider chart).
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

# FER-2013 label order (matches emotion_detector.emotion_labels)
EMOTION_ORDER = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def plot_emotion_radar(
    confidence_scores: Dict[str, float],
    highlight: Optional[str] = None,
) -> plt.Figure:
    """
    Build a polar radar chart of per-emotion confidence (%).

    Args:
        confidence_scores: e.g. {"happy": 72.5, "sad": 12.1, ...}
        highlight: Dominant emotion name (lowercase) for accent color on that spoke.
    """
    labels = EMOTION_ORDER
    values = [float(confidence_scores.get(name, 0.0)) for name in labels]
    n = len(labels)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    line_color = "#818cf8"
    fill_color = "#6366f1"
    if highlight and highlight in labels:
        line_color = "#34d399"
        fill_color = "#10b981"

    ax.plot(angles_closed, values_closed, "o-", linewidth=2, color=line_color, markersize=6)
    ax.fill(angles_closed, values_closed, alpha=0.35, color=fill_color)

    ax.set_xticks(angles)
    ax.set_xticklabels([name.capitalize() for name in labels], color="#e5e7eb", size=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], color="#9ca3af", size=8)
    ax.grid(color="#374151", linestyle="--", linewidth=0.6, alpha=0.8)
    ax.spines["polar"].set_color("#4b5563")
    ax.set_title("Emotion distribution (%)", color="#f3f4f6", pad=16, size=11)

    plt.tight_layout()
    return fig
