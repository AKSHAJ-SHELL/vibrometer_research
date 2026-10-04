"""House style for figures — fixed categorical order, no dual y-axis, json sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical order — never cycle
CLASS_ORDER = [
    "healthy",
    "imbalance",
    "misalignment",
    "bearing_IR",
    "bearing_OR",
    "bearing_ball",
]

# Colourblind-safe qualitative (max 3 series for scatter / small-multiples gate)
COLORS_3 = ["#0072B2", "#E69F00", "#009E73"]
COLORS_6 = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]

PROTOCOL_LABELS = {
    0: "P0 random (invalid)",
    1: "P1 leave-file",
    2: "P2 cross-load",
    3: "P3 fault-id",
    4: "P4 severity",
    5: "P5 cross-ds",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(
    fig: plt.Figure,
    out_dir: str | Path,
    name: str,
    data: dict[str, Any] | None = None,
    source: str = "vibedge synthetic / public datasets",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    # source footnote
    fig.text(0.01, 0.01, f"Source: {source}", fontsize=7, color="0.4")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(png, bbox_inches="tight")
    if data is not None:
        sidecar = out_dir / f"{name}.json"
        sidecar.write_text(json.dumps(data, indent=2, default=_json_default))
    plt.close(fig)
    return png


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj))
