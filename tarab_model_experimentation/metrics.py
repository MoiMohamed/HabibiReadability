from __future__ import annotations


def compute_report_delta(exp_report_df, baseline_report_df):
    import pandas as pd

    metric_cols = ["precision", "recall", "f1-score"]
    e = exp_report_df.set_index("label")
    b = baseline_report_df.set_index("label")
    common = e.index.intersection(b.index)
    delta = e.loc[common, metric_cols] - b.loc[common, metric_cols]
    delta["support"] = e.loc[common, "support"]
    delta = delta.reset_index()
    return pd.DataFrame(delta)


def shift_readability_labels_to_one_based(df):
    out = df.copy()
    if "label" in out.columns:
        out["label"] = out["label"].apply(
            lambda x: str(int(x) + 1) if str(x).isdigit() else x
        )
    return out


def one_based_confusion_matrix(cm_df):
    out = cm_df.copy()
    out.index = [str(int(x) + 1) if str(x).isdigit() else str(x) for x in out.index]
    out.columns = [str(int(x) + 1) if str(x).isdigit() else str(x) for x in out.columns]
    return out


def predicted_class_distribution(cm_df):
    import pandas as pd

    m = cm_df.to_numpy(dtype=int)
    labels = [int(x) + 1 for x in cm_df.columns]
    return pd.Series(m.sum(axis=0), index=labels, dtype=int)


def true_class_distribution(cm_df):
    import pandas as pd

    m = cm_df.to_numpy(dtype=int)
    labels = [int(x) + 1 for x in cm_df.index]
    return pd.Series(m.sum(axis=1), index=labels, dtype=int)


def per_class_accuracy_series(cm_df):
    import numpy as np
    import pandas as pd

    m = cm_df.to_numpy(dtype=int)
    row_totals = m.sum(axis=1)
    diag = np.diag(m)
    labels = [int(x) + 1 for x in cm_df.index]
    acc = np.divide(
        diag.astype(float),
        row_totals.astype(float),
        out=np.zeros_like(diag, dtype=float),
        where=row_totals > 0,
    )
    return pd.Series(acc, index=labels)


def per_class_precision_series(cm_df):
    """Precision per level k: of predictions labeled k, fraction truly k."""
    import numpy as np
    import pandas as pd

    m = cm_df.to_numpy(dtype=int)
    col_totals = m.sum(axis=0)
    diag = np.diag(m)
    labels = [int(x) + 1 for x in cm_df.index]
    prec = np.divide(
        diag.astype(float),
        col_totals.astype(float),
        out=np.zeros_like(diag, dtype=float),
        where=col_totals > 0,
    )
    return pd.Series(prec, index=labels)


def signed_error_count_series(cm_df):
    import pandas as pd

    dev = compute_confusion_deviation_stats(cm_df)
    grouped = dev["cell_details_df"].groupby("pred_minus_true", as_index=True)["count"].sum()
    return grouped.astype(int)


def compute_confusion_deviation_stats(cm_df):
    import numpy as np
    import pandas as pd

    m = cm_df.to_numpy(dtype=np.int64)
    n_rows, n_cols = m.shape
    row_idx, col_idx = np.indices((n_rows, n_cols))
    diff = col_idx - row_idx
    abs_diff = np.abs(diff)
    total = int(m.sum())
    if total == 0:
        return {
            "summary_df": pd.DataFrame(
                [
                    {"metric": "total_samples", "value": 0},
                    {"metric": "exact_match_rate", "value": 0.0},
                    {"metric": "within_1_rate", "value": 0.0},
                    {"metric": "within_2_rate", "value": 0.0},
                    {"metric": "mae_levels", "value": 0.0},
                    {"metric": "rmse_levels", "value": 0.0},
                    {"metric": "signed_bias_levels", "value": 0.0},
                ]
            ),
            "distance_df": pd.DataFrame(),
            "cell_details_df": pd.DataFrame(),
            "per_true_summary_df": pd.DataFrame(),
        }

    exact = int(np.trace(m))
    within_1 = int(m[abs_diff <= 1].sum())
    within_2 = int(m[abs_diff <= 2].sum())
    mae = float((abs_diff * m).sum() / total)
    rmse = float(np.sqrt(((diff ** 2) * m).sum() / total))
    signed_bias = float((diff * m).sum() / total)

    max_d = int(abs_diff.max())
    dist_rows = []
    for d in range(max_d + 1):
        c = int(m[abs_diff == d].sum())
        dist_rows.append(
            {"|pred-true|": d, "count": c, "pct": (100.0 * c / total)}
        )

    cell_details_df = pd.DataFrame(
        {
            "true_level": (row_idx.ravel() + 1).astype(int),
            "pred_level": (col_idx.ravel() + 1).astype(int),
            "count": m.ravel().astype(int),
            "pred_minus_true": diff.ravel().astype(int),
            "abs_error": abs_diff.ravel().astype(int),
        }
    )
    row_totals = m.sum(axis=1).astype(np.int64)
    col_totals = m.sum(axis=0).astype(np.int64)
    cell_details_df["pct_within_true"] = cell_details_df.apply(
        lambda r: (100.0 * r["count"] / row_totals[int(r["true_level"]) - 1])
        if row_totals[int(r["true_level"]) - 1] > 0
        else 0.0,
        axis=1,
    )
    cell_details_df["pct_within_pred"] = cell_details_df.apply(
        lambda r: (100.0 * r["count"] / col_totals[int(r["pred_level"]) - 1])
        if col_totals[int(r["pred_level"]) - 1] > 0
        else 0.0,
        axis=1,
    )

    per_true_rows = []
    for r in range(n_rows):
        support = int(row_totals[r])
        if support == 0:
            per_true_rows.append(
                {
                    "true_level": r + 1,
                    "support": 0,
                    "exact_rate": 0.0,
                    "within_1_rate": 0.0,
                    "within_2_rate": 0.0,
                    "mean_abs_error": 0.0,
                    "mean_pred_minus_true": 0.0,
                    "under_rate": 0.0,
                    "over_rate": 0.0,
                }
            )
            continue
        row_counts = m[r]
        per_true_rows.append(
            {
                "true_level": r + 1,
                "support": support,
                "exact_rate": float(row_counts[r] / support),
                "within_1_rate": float(row_counts[abs_diff[r] <= 1].sum() / support),
                "within_2_rate": float(row_counts[abs_diff[r] <= 2].sum() / support),
                "mean_abs_error": float((abs_diff[r] * row_counts).sum() / support),
                "mean_pred_minus_true": float((diff[r] * row_counts).sum() / support),
                "under_rate": float(row_counts[diff[r] < 0].sum() / support),
                "over_rate": float(row_counts[diff[r] > 0].sum() / support),
            }
        )
    per_true_summary_df = pd.DataFrame(per_true_rows)

    summary_df = pd.DataFrame(
        [
            {"metric": "total_samples", "value": total},
            {"metric": "exact_match_rate", "value": exact / total},
            {"metric": "within_1_rate", "value": within_1 / total},
            {"metric": "within_2_rate", "value": within_2 / total},
            {"metric": "mae_levels", "value": mae},
            {"metric": "rmse_levels", "value": rmse},
            {"metric": "signed_bias_levels", "value": signed_bias},
        ]
    )
    return {
        "summary_df": summary_df,
        "distance_df": pd.DataFrame(dist_rows),
        "cell_details_df": cell_details_df,
        "per_true_summary_df": per_true_summary_df,
    }
