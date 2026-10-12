"""Patients x cells design calculator for a patient-level fraction readout.

Variance components (method of moments):
  sigma_c2_per_cell  binomial variance per cell, p(1-p), for a fraction readout
  sigma_s2           between-specimen variance within a patient, from repeated specimens,
                     after removing expected cell-sampling variance
  sigma_b2           between-patient variance within an outcome group, after removing
                     expected specimen and cell-sampling variance
Power is simulated for a Welch test on patient means.  More cells shrink only the
last term; outcome power is governed by the number of patients.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def estimate_components(specimens: pd.DataFrame, *, value: str = "high_state_fraction", count: str = "n_fib",
                        patient: str = "patient", group: str = "arm", groups=("DFU-healer", "DFU-nonhealer")) -> dict:
    data = specimens[specimens[group].isin(groups)].copy()
    mean = float(data[value].mean())
    per_cell = mean * (1.0 - mean)
    pair_terms = []
    for _, block in data.groupby(patient):
        if len(block) < 2:
            continue
        values, cells = block[value].to_numpy(dtype=float), block[count].to_numpy(dtype=float)
        observed = float(np.var(values, ddof=1))
        expected_cells = float(np.mean(per_cell / cells))
        pair_terms.append(observed - expected_cells)
    sigma_s2 = max(float(np.mean(pair_terms)), 0.0) if pair_terms else 0.0
    patient_means = data.groupby([group, patient]).agg(value=(value, "mean"), specimens=(value, "size"),
                                                       cells=(count, "mean")).reset_index()
    pooled, degrees = 0.0, 0
    for _, block in patient_means.groupby(group):
        if len(block) > 1:
            pooled += float(np.var(block["value"], ddof=1)) * (len(block) - 1)
            degrees += len(block) - 1
    observed_between = pooled / degrees if degrees else float("nan")
    expected_noise = float(np.mean(sigma_s2 / patient_means["specimens"]
                                   + per_cell / (patient_means["specimens"] * patient_means["cells"])))
    return {"value": value, "mean": mean, "sigma_c2_per_cell": per_cell, "sigma_s2": sigma_s2,
            "sigma_b2": max(observed_between - expected_noise, 0.0), "observed_between_patient_variance": observed_between,
            "repeated_patients": int(len(pair_terms)), "patients": int(len(patient_means)),
            "group_sizes": {str(k): int(v) for k, v in patient_means.groupby(group).size().items()}}


def reliability(components: dict, cells: int, specimens: int) -> float:
    noise = components["sigma_s2"] / specimens + components["sigma_c2_per_cell"] / (specimens * cells)
    total = components["sigma_b2"] + noise
    return float(components["sigma_b2"] / total) if total > 0 else float("nan")


def simulate_power(delta: float, n_per_group: int, cells: int, specimens: int, components: dict, *,
                   alpha: float = 0.05, n_sim: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    means = []
    for shift in (0.0, float(delta)):
        truth = rng.normal(components["mean"] + shift, np.sqrt(components["sigma_b2"]), size=(n_sim, n_per_group, 1))
        specimen = truth + rng.normal(0.0, np.sqrt(components["sigma_s2"]), size=(n_sim, n_per_group, specimens))
        observed = rng.binomial(int(cells), np.clip(specimen, 1e-6, 1 - 1e-6)) / float(cells)
        means.append(observed.mean(axis=2))
    first, second = means
    result = stats.ttest_ind(second, first, axis=1, equal_var=False)
    return float(np.mean(result.pvalue < alpha))


def design_grid(components: dict, *, patients, cells, specimens, deltas, n_sim: int = 1000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for delta in deltas:
        for n in patients:
            for c in cells:
                for k in specimens:
                    rows.append({"delta": float(delta), "patients_per_group": int(n), "cells_per_specimen": int(c),
                                 "specimens_per_patient": int(k),
                                 "power": simulate_power(delta, n, c, k, components, n_sim=n_sim, seed=seed),
                                 "reliability": reliability(components, c, k)})
    return pd.DataFrame(rows)
