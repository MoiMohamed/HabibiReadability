from __future__ import annotations

import ast
import re

import streamlit as st

from tarab_model_experimentation.constants import LOGS_DIR


def parse_classification_report_block(lines: list[str], header_idx: int):
    import pandas as pd

    rows: list[dict[str, float | int | str | None]] = []
    i = header_idx + 2
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        m = re.match(
            r"^(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)$",
            s,
        )
        if m:
            rows.append(
                {
                    "label": m.group(1),
                    "precision": float(m.group(2)),
                    "recall": float(m.group(3)),
                    "f1-score": float(m.group(4)),
                    "support": int(m.group(5)),
                }
            )
            i += 1
            continue
        break

    for _ in range(5):
        if i >= len(lines):
            break
        s = lines[i].strip()
        m_acc = re.match(r"^accuracy\s+([0-9.]+)\s+(\d+)$", s)
        if m_acc:
            rows.append(
                {
                    "label": "accuracy",
                    "precision": None,
                    "recall": None,
                    "f1-score": float(m_acc.group(1)),
                    "support": int(m_acc.group(2)),
                }
            )
        m_avg = re.match(
            r"^(macro avg|weighted avg)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)$",
            s,
        )
        if m_avg:
            rows.append(
                {
                    "label": m_avg.group(1),
                    "precision": float(m_avg.group(2)),
                    "recall": float(m_avg.group(3)),
                    "f1-score": float(m_avg.group(4)),
                    "support": int(m_avg.group(5)),
                }
            )
        i += 1

    return pd.DataFrame(rows)


def parse_confusion_matrix_block(lines: list[str], matrix_start_idx: int):
    import numpy as np
    import pandas as pd

    block_parts: list[str] = []
    i = matrix_start_idx
    while i < len(lines):
        s = lines[i].strip()
        if s:
            block_parts.append(s)
        if "]]" in s:
            break
        i += 1

    block = " ".join(block_parts)
    row_chunks = re.findall(r"\[\s*([0-9\-\s]+?)\s*\]", block)
    rows = [[int(x) for x in re.findall(r"-?\d+", chunk)] for chunk in row_chunks if chunk.strip()]

    if not rows:
        raise ValueError("Could not parse confusion matrix rows from log block.")

    max_len = max(len(r) for r in rows)
    if any(len(r) != max_len for r in rows):
        rows = [r + [0] * (max_len - len(r)) for r in rows]

    matrix = np.array(rows, dtype=int)
    row_labels = [str(i) for i in range(matrix.shape[0])]
    col_labels = [str(i) for i in range(matrix.shape[1])]
    return pd.DataFrame(matrix, index=row_labels, columns=col_labels)


@st.cache_data(show_spinner=False)
def parse_training_log(log_filename: str):
    txt = (LOGS_DIR / log_filename).read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()

    model_match = re.search(r"model:\s*([^,]+),\s*levels:", txt)
    run_match = re.search(r"run='([^']+)'", txt)
    meta = {
        "model": model_match.group(1).strip() if model_match else "",
        "run_name": run_match.group(1).strip() if run_match else log_filename.replace(".out", ""),
    }

    epochs = []
    current_header_idx = None
    current_matrix_idx = None
    for idx, line in enumerate(lines):
        s = line.strip()
        if "precision" in s and "recall" in s and "f1-score" in s and "support" in s:
            current_header_idx = idx
        if s.startswith("[["):
            current_matrix_idx = idx
        if s.startswith("{") and "eval_Quadratic Weighted Kappa" in s and "'epoch':" in s:
            try:
                d = ast.literal_eval(s)
            except Exception:
                continue
            if current_header_idx is None or current_matrix_idx is None:
                continue
            report_df = parse_classification_report_block(lines, current_header_idx)
            cm_df = parse_confusion_matrix_block(lines, current_matrix_idx)
            epochs.append(
                {
                    "epoch": float(d.get("epoch", float("nan"))),
                    "qwk": float(d.get("eval_Quadratic Weighted Kappa", float("nan"))),
                    "metrics": d,
                    "report_df": report_df,
                    "cm_df": cm_df,
                }
            )

    best_epoch = None
    if epochs:
        best_epoch = max(epochs, key=lambda x: x["qwk"])

    test_metrics = {}
    capture_test = False
    for line in lines:
        s = line.strip()
        if s == "***** test metrics *****":
            capture_test = True
            continue
        if capture_test:
            m = re.match(r"^([a-zA-Z0-9_ ]+?)\s*=\s*([0-9.]+)$", s)
            if m:
                key = m.group(1).strip().replace(" ", "_")
                test_metrics[key] = float(m.group(2))
                continue
            if s.startswith("Accuracy:") or s.startswith("wandb"):
                continue
            if s == "":
                continue
            if not m and not s.startswith("test_"):
                if len(test_metrics) > 0:
                    break

    return {"meta": meta, "epochs": epochs, "best_epoch": best_epoch, "test_metrics": test_metrics}
