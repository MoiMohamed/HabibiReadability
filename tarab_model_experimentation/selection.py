from __future__ import annotations

import re
from dataclasses import dataclass

import streamlit as st

from tarab_model_experimentation.constants import LENGTH_MATCHED_DIR, LOGS_DIR, MIN_L_DIR

# Retired / hidden .out logs (not shown in dashboards).
_DEPRECATED_LOG_FILES: frozenset[str] = frozenset(
    {
        "habibi-readability-arabertv02-baseline-14943095.out",
        "barec_tarab_2X_55k_match_distribution_155k-14978447.out",
        "barec_tarab_2X_55k_match_distribution_155k_wo_pseudolabel19-15656421.out",
        "barec_tarab_2x_minL8-15943060.out",
    }
)

# When several .out files share a chart label, prefer this file.
_PREFERRED_LOG_FILE: dict[str, str] = {
    "baseline": "habibi-readability-arabertv02-baseline-15686675.out",
    "dist_155K": "barec_tarab_2X_55k_match_distribution_155k-15686677.out",
    "length_matched": "caps10_length_matching-15852637.out",
    "minL8": "barec_tarab_2x_minL8-15943060.out",
}

# Retired split CSVs (duplicate chart labels — not used on dashboards).
_DEPRECATED_SPLIT_CSV: frozenset[str] = frozenset(
    {
        "barec_tarab_2X_55k_match_distribution_155k_wo_class19.csv",
    }
)

# When several split CSVs share a chart label, prefer this file.
_PREFERRED_SPLIT_CSV: dict[str, str] = {
    "dist_155K": "barec_tarab_2X_55k_match_distribution_155k.csv",
}


@dataclass(frozen=True)
class ExperimentSelection:
    log_files: list[str]
    baseline_file: str | None
    selected_experiment: str
    selected_display: str


def log_display_name(fname: str) -> str:
    stem = fname[:-4] if fname.endswith(".out") else fname
    return re.sub(r"-\d+$", "", stem)


def experiment_chart_label(fname: str) -> str:
    """Short x-axis labels for performance charts."""
    stem = log_display_name(fname).lower()

    if "baseline" in stem:
        return "baseline"

    if "length_matching" in stem or "length_matched" in stem:
        return "length_matched"

    if "minl8" in stem or "min_l8" in stem:
        return "minL8"

    uniform_match = re.search(r"uniform_(\d+)k_per_class", stem)
    if uniform_match:
        return f"uni_{uniform_match.group(1)}K"

    dist_match = re.search(r"match_distribution_(\d+)k", stem)
    if dist_match:
        # if "wo_pseudolabel19" in stem or "wo_class19" in stem:
        #     return f"dist_{dist_match.group(1)}K_wo_19"
        return f"dist_{dist_match.group(1)}K"

    return log_display_name(fname)


def experiment_chart_sort_key(label: str) -> tuple[int, int | str]:
    if label == "baseline":
        return (-1, 0)

    # wo_dist_match = re.match(r"dist_(\d+)K_wo_19", label)
    # if wo_dist_match:
    #     return (0, int(wo_dist_match.group(1)), 1)

    dist_match = re.match(r"dist_(\d+)K", label)
    if dist_match:
        return (0, int(dist_match.group(1)))
    if label == "length_matched":
        return (0, 300)
    uni_match = re.match(r"uni_(\d+)K", label)
    if uni_match:
        return (1, int(uni_match.group(1)))
    return (2, label)


def list_experiment_log_files() -> list[str]:
    names: list[str] = []
    for directory in (LOGS_DIR, LENGTH_MATCHED_DIR, MIN_L_DIR):
        if not directory.exists():
            continue
        names.extend(
            p.name
            for p in directory.glob("*.out")
            if p.name not in _DEPRECATED_LOG_FILES
        )
    return sorted(dict.fromkeys(names))


_MAIN_TAB_HIDDEN_LABELS: frozenset[str] = frozenset({"length_matched"})


def list_model_experimentation_log_files() -> list[str]:
    """Logs for the Model experimentation tab (excludes length-matched runs)."""
    return [
        f
        for f in list_experiment_log_files()
        if experiment_chart_label(f) not in _MAIN_TAB_HIDDEN_LABELS
    ]


def list_performance_profile_log_files() -> list[str]:
    """All main-tab dist/uni/baseline runs plus length-matched for overview profile charts."""
    hidden = _MAIN_TAB_HIDDEN_LABELS - {"length_matched"}
    return [
        f
        for f in list_experiment_log_files()
        if experiment_chart_label(f) not in hidden
    ]


def resolve_training_log_path(log_filename: str):
    """Resolve a log file under ``data/logs``, ``data/length_matched``, or ``data/minL``."""
    for directory in (LOGS_DIR, LENGTH_MATCHED_DIR, MIN_L_DIR):
        path = directory / log_filename
        if path.exists():
            return path
    return LOGS_DIR / log_filename


def resolve_split_csv_for_chart_label(label: str, candidates: list[str]) -> str:
    """Pick one split CSV when several map to the same experiment chart label."""
    preferred = _PREFERRED_SPLIT_CSV.get(label)
    if preferred and preferred in candidates:
        return preferred
    active = [c for c in candidates if c not in _DEPRECATED_SPLIT_CSV]
    pool = active if active else candidates
    return sorted(pool)[0]


def preferred_log_for_chart_label(log_files: list[str], label: str) -> str | None:
    """Resolve chart label to a log file, preferring canonical runs when duplicated."""
    matches = [f for f in log_files if experiment_chart_label(f) == label]
    if not matches:
        return None
    preferred_file = _PREFERRED_LOG_FILE.get(label)
    if preferred_file and preferred_file in matches:
        return preferred_file
    return matches[0]


def render_experiment_selection() -> ExperimentSelection | None:
    log_files = list_experiment_log_files()
    if not log_files:
        st.warning("No `.out` log files found in `data/logs`.")
        return None

    baseline_file = preferred_log_for_chart_label(log_files, "baseline")
    if baseline_file is None:
        baseline_file = next((x for x in log_files if "baseline" in x.lower()), None)
    experiment_files = [x for x in log_files if x != baseline_file]
    options = experiment_files if experiment_files else log_files
    display_names = [log_display_name(f) for f in options]
    selected_display = st.selectbox(
        "Select experiment",
        display_names,
        index=0,
        key="model_exp_selected_log",
    )
    selected_experiment = options[display_names.index(selected_display)]
    return ExperimentSelection(
        log_files=log_files,
        baseline_file=baseline_file,
        selected_experiment=selected_experiment,
        selected_display=selected_display,
    )
