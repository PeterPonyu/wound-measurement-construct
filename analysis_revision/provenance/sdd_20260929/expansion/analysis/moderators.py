"""Application conditions: meta-regression of task-level log ratios on pre-target features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def meta_regression(lr: pd.DataFrame, features) -> dict:
    import statsmodels.api as sm

    features = list(dict.fromkeys(features))
    data = lr.replace([np.inf, -np.inf], np.nan).dropna(subset=["dataset_id", "log_ratio", *features]).copy()
    clusters = int(data["dataset_id"].nunique())
    out = {"model": "OLS with dataset-clustered errors", "n": int(len(data)), "datasets": clusters,
           "standardized_features": True, "descriptive_only": True, "notes": []}
    if data.empty:
        return {**out, "reason": "no complete finite rows"}
    varying = [name for name in features if data[name].nunique() > 1]
    out["dropped_constant_features"] = sorted(set(features) - set(varying))
    for name in varying:
        data[name] = (data[name] - data[name].mean()) / data[name].std(ddof=1)
    design = sm.add_constant(data[varying], has_constant="add").rename(columns={"const": "Intercept"})
    if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1] or len(data) <= design.shape[1]:
        return {**out, "reason": "rank-deficient design or insufficient residual degrees of freedom"}
    fitted = sm.OLS(data["log_ratio"], design).fit()
    out["params"] = {str(k): float(v) for k, v in fitted.params.items()}
    if clusters < 10 or clusters <= design.shape[1]:
        return {**out, "reason": "fewer than 10 datasets or insufficient clusters for design; no inferential p-values/CI"}
    groups = pd.factorize(data["dataset_id"])[0]
    fitted = sm.OLS(data["log_ratio"], design).fit(
        cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True, "df_correction": True}, use_t=True)
    interval = fitted.conf_int()
    if not np.all(np.isfinite(interval.to_numpy())):
        return {**out, "reason": "nonfinite cluster-robust interval"}
    return {**out, "descriptive_only": False, "inference_df": clusters - 1,
            "pvalues": {str(k): float(v) for k, v in fitted.pvalues.items()},
            "ci": {str(k): [float(interval.loc[k, 0]), float(interval.loc[k, 1])] for k in fitted.params.index}}
