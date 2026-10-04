"""Cut recordings into windows that remember which recording they came from.

Every window's SampleMeta keeps ``file_id`` = the source recording, so splits
can check that no recording lands on both sides (see splits.make_splits).
"""

from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np

from vibedge.taxonomy import SampleMeta


def window_starts(n: int, win: int, overlap: float = 0.0) -> list[int]:
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if win <= 0 or n < win:
        return []
    hop = max(1, int(round(win * (1.0 - overlap))))
    return list(range(0, n - win + 1, hop))


def make_windows(
    recordings: Sequence[tuple[np.ndarray, SampleMeta]],
    window_s: float,
    overlap: float = 0.0,
    max_windows_per_recording: int | None = None,
) -> list[tuple[np.ndarray, SampleMeta]]:
    """Split each (signal, meta) recording into windows of window_s seconds.

    Recordings shorter than one window are dropped. Window meta is a copy of
    the recording meta with extras["window_index"] / ["window_start"].
    """
    out: list[tuple[np.ndarray, SampleMeta]] = []
    for x, meta in recordings:
        win = int(round(window_s * meta.fs))
        starts = window_starts(len(x), win, overlap)
        if max_windows_per_recording is not None:
            starts = starts[:max_windows_per_recording]
        for k, s in enumerate(starts):
            m = copy.deepcopy(meta)
            extras: dict[str, Any] = dict(m.extras or {})
            extras.update({"window_index": k, "window_start": s, "recording_id": meta.file_id})
            m.extras = extras
            out.append((x[s : s + win], m))
    return out


def meta_columns(windows: Sequence[tuple[np.ndarray, SampleMeta]]) -> dict[str, list]:
    """Column view of window metadata, in the shape make_splits expects."""
    metas = [m for _, m in windows]
    return {
        "recording_ids": [m.file_id for m in metas],
        "fault_ids": [m.fault_id for m in metas],
        "labels": [m.label for m in metas],
        "loads": [m.load_hp for m in metas],
        "severities": [m.severity for m in metas],
        "datasets": [m.dataset for m in metas],
    }
