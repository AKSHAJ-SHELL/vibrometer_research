"""House style for figures — fixed categorical order, no dual y-axis, json sidecars."""

from __future__ import annotations

import json
import os
import re
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
    # source footnote (the paper's captions carry provenance, so the paper build omits it)
    if not os.environ.get("VIBEDGE_PAPER"):
        fig.text(0.01, 0.01, f"Source: {source}", fontsize=7, color="0.4")
    _paper_fonts(fig, name)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    # VIBEDGE_PAPER=1 drops the internal figure codes ("C1 · ", "A3 · ") from titles: the paper's
    # captions carry the numbering
    if os.environ.get("VIBEDGE_PAPER") and fig._suptitle is not None:
        fig._suptitle.set_text(re.sub(r"^[A-Z]\d+[a-z]? · ", "", fig._suptitle.get_text()))
    for ax in fig.axes:
        if os.environ.get("VIBEDGE_PAPER"):
            for t in (ax.title, ax._left_title, ax._right_title):   # keep the size; change only the words
                t.set_text(re.sub(r"^[A-Z]\d+[a-z]? · ", "", t.get_text()))
    # VIBEDGE_FIG_DPI overrides the resolution (the paper build uses 300 dpi)
    dpi = int(os.environ.get("VIBEDGE_FIG_DPI", "0")) or None
    fig.savefig(png, bbox_inches="tight", dpi=dpi)
    if data is not None:
        sidecar = out_dir / f"{name}.json"
        sidecar.write_text(json.dumps(data, indent=2, default=_json_default))
    plt.close(fig)
    return png


# Printed width (inches) of each figure in the IEEE paper: 3.5 = one column, 7.16 = full text width.
PAPER_WIDTH_IN = {
    "C1_ladder_real": 7.16, "C1_ladder_real_gbdt": 7.16, "speed_ablation": 7.16,
    "loss_curves_gbdt": 5.7, "deployment_degradation_real": 6.1, "C10_transfer_real": 5.4,
}
PAPER_MIN_PT = 7.5


def _paper_fonts(fig: plt.Figure, name: str) -> None:
    """Paper build: enlarge text so it prints at >= PAPER_MIN_PT once scaled to its width in the paper."""
    if not os.environ.get("VIBEDGE_PAPER"):
        return
    import matplotlib.text as mtext

    scale = PAPER_WIDTH_IN.get(name, 3.5) / fig.get_size_inches()[0]
    if scale >= 1:
        return
    floor = PAPER_MIN_PT / scale
    for t in fig.findobj(mtext.Text):
        if t.get_text() and t.get_fontsize() < floor:
            t.set_fontsize(floor)
    for ax in fig.axes:   # tick labels are rebuilt at draw time, so set them through the axis
        ax.tick_params(axis="both", labelsize=floor)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj))
