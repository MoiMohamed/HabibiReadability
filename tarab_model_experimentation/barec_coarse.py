from __future__ import annotations

import numpy as np


def level_19_from_cm_label(label: str | int) -> int:
    """Confusion-matrix indices in logs are 0–18; BAREC readability levels are 1–19."""
    return int(label) + 1


def collapse_19_to_7(level_19: int) -> int:
    if level_19 <= 4:
        return 1
    if level_19 <= 7:
        return 2
    if level_19 <= 9:
        return 3
    if level_19 <= 11:
        return 4
    if level_19 <= 13:
        return 5
    if level_19 <= 15:
        return 6
    return 7


def collapse_19_to_5(level_19: int) -> int:
    if level_19 <= 7:
        return 1
    if level_19 <= 11:
        return 2
    if level_19 <= 13:
        return 3
    if level_19 <= 15:
        return 4
    return 5


def collapse_19_to_3(level_19: int) -> int:
    if level_19 <= 11:
        return 1
    if level_19 <= 13:
        return 2
    return 3


_COLLAPSE_FN = {
    7: collapse_19_to_7,
    5: collapse_19_to_5,
    3: collapse_19_to_3,
}


def coarse_accuracy_from_confusion_matrix(cm_df, granularity: int) -> float | None:
    """Exact-match accuracy after collapsing true/pred 19-level labels (BAREC scheme)."""
    if cm_df is None or cm_df.empty:
        return None
    collapse = _COLLAPSE_FN.get(granularity)
    if collapse is None:
        raise ValueError(f"granularity must be 3, 5, or 7; got {granularity}")

    m = cm_df.to_numpy(dtype=np.int64)
    total = int(m.sum())
    if total == 0:
        return None

    correct = 0
    for i, true_label in enumerate(cm_df.index):
        true_19 = level_19_from_cm_label(true_label)
        true_coarse = collapse(true_19)
        for j, pred_label in enumerate(cm_df.columns):
            pred_19 = level_19_from_cm_label(pred_label)
            pred_coarse = collapse(pred_19)
            if true_coarse == pred_coarse:
                correct += int(m[i, j])
    return correct / total
