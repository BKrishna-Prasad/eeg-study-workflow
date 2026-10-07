"""Subject aggregation, paired comparisons and grand-average PSD curves."""

import numpy as np
import pandas as pd
from scipy.stats import sem, ttest_rel, wilcoxon


def load_features(paths):
    frames = [pd.read_csv(p, dtype={"subject_id": str, "condition": str, "metric": str}) for p in paths]
    if not frames:
        raise ValueError("Provide at least one feature table")
    data = pd.concat(frames, ignore_index=True)
    required = {"subject_id", "condition", "metric", "value", "units", "channels", "band_hz", "demonstration_only"}
    if not required.issubset(data):
        raise ValueError(f"Missing feature columns: {required - set(data)}")
    if data[list(required)].isna().any().any() or data.subject_id.astype(str).eq("").any():
        raise ValueError("Missing feature identities, metadata or values")
    if data.duplicated(["subject_id", "condition", "metric"]).any():
        raise ValueError("Duplicate subject/condition/metric rows")
    if not np.isfinite(data.value).all():
        raise ValueError("Feature values must be finite")
    if len(data.demonstration_only.unique()) != 1 or not data.demonstration_only.isin([True, False]).all():
        raise ValueError("Do not mix demonstration and research rows; use boolean demonstration status")
    for metric, subset in data.groupby("metric"):
        for column in ("units", "channels", "band_hz"):
            if subset[column].nunique() != 1:
                raise ValueError(f"Inconsistent {column} definition for {metric}")
    return data


def _test_pair(a, b):
    difference = a - b
    if len(a) < 2:
        return {"t": np.nan, "p_t_unadjusted": np.nan, "w": np.nan, "p_w_unadjusted": np.nan,
                "status": "fewer_than_two_pairs"}
    if np.all(difference == 0):
        return {"t": np.nan, "p_t_unadjusted": np.nan, "w": np.nan, "p_w_unadjusted": np.nan,
                "status": "all_differences_zero"}
    w, pw = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", method="auto")
    if np.ptp(difference) == 0:
        return {"t": np.nan, "p_t_unadjusted": np.nan, "w": w, "p_w_unadjusted": pw,
                "status": "constant_difference_t_undefined"}
    t, pt = ttest_rel(a, b, alternative="two-sided")
    return {"t": t, "p_t_unadjusted": pt, "w": w, "p_w_unadjusted": pw, "status": "ok"}


def paired_comparison(data, condition_a, condition_b, metrics=None):
    """Pair participants, never epochs; report dropped IDs and raw p-values."""
    if condition_a == condition_b:
        raise ValueError("Choose two different conditions")
    metrics = sorted(data.metric.unique()) if metrics is None else list(metrics)
    rows = []
    for metric in metrics:
        subset = data[data.metric == metric]
        if subset.duplicated(["subject_id", "condition"]).any():
            raise ValueError("Duplicate paired rows")
        wide = subset.pivot(index="subject_id", columns="condition", values="value")
        wide = wide.reindex(columns=[condition_a, condition_b])
        pairs = wide.dropna()
        missing = wide.index[wide.isna().any(axis=1)].astype(str).tolist()
        a, b = pairs.to_numpy().T
        row = {"metric": metric, "condition_a": condition_a, "condition_b": condition_b,
               "n_pairs": len(pairs), "paired_subject_ids": ";".join(pairs.index.astype(str)),
               "excluded_subject_ids": ";".join(missing), "difference_direction": "A minus B",
               "mean_difference": float((a - b).mean()) if len(a) else np.nan,
               "demonstration_only": bool(data.demonstration_only.iloc[0])}
        row.update(_test_pair(a, b)); rows.append(row)
    return pd.DataFrame(rows)


def hemispheric_comparison(data, conditions):
    """Compare right and left frontal log PSD within each selected condition."""
    rows = []
    for condition in conditions:
        subset = data[data.condition == condition].copy()
        bands = [m[len("Left_Frontal_"):] for m in subset.metric.unique()
                 if m.startswith("Left_Frontal_") and m.endswith("_ln_mean_psd")]
        for band in bands:
            pair = subset[subset.metric.isin(["Left_Frontal_" + band, "Right_Frontal_" + band])].copy()
            pair["condition"] = pair.metric.str.split("_", n=1).str[0]
            pair["metric"] = band
            result = paired_comparison(pair, "Right", "Left", [band])
            result["source_condition"] = condition
            rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def grand_average(curves):
    """Average subject dB curves, with SEM and equal participant weighting."""
    keys = ["subject_id", "condition", "roi", "frequency_hz"]
    if curves.empty or curves.duplicated(keys).any() or not np.isfinite(curves.psd_db).all():
        raise ValueError("Require unique, finite subject PSD curves")
    if curves.demonstration_only.nunique() != 1:
        raise ValueError("Do not mix synthetic and research curves")
    rows = []
    for (condition, roi), subset in curves.groupby(["condition", "roi"]):
        if subset.channels.nunique() != 1:
            raise ValueError("ROI channels differ across subjects")
        wide = subset.pivot(index="subject_id", columns="frequency_hz", values="psd_db")
        if wide.isna().any().any():
            raise ValueError("Subject frequency grids differ; no implicit interpolation")
        mean = wide.to_numpy().mean(axis=0)
        error = sem(wide.to_numpy(), axis=0) if len(wide) > 1 else np.full(len(mean), np.nan)
        for frequency, value, uncertainty in zip(wide.columns, mean, error):
            rows.append({"condition": condition, "roi": roi, "frequency_hz": frequency,
                         "mean_subject_db": value, "sem_subject_db": uncertainty, "n_subjects": len(wide),
                         "demonstration_only": bool(subset.demonstration_only.iloc[0])})
    return pd.DataFrame(rows)
