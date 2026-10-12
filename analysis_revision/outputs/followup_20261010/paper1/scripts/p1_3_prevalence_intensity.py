#!/usr/bin/env python3
"""P1-3 prevalence vs intensity against explicit generative nulls.

Data: topic-0 loadings of the 19,410 discovery fibroblasts in 25 specimens
(cells.parquet), frozen cut c = 0.1842958.

Descriptive estimands (Eq. 6-7): A_s = (1-w_s) m_L + w_s m_H,
R_s = Ybar_s - A_s.  Variance share Var(A)/Var(Ybar), Var(R)/Var(Ybar),
2Cov(A,R)/Var(Ybar) and R^2 = corr(A, Ybar)^2, across 25 specimens and across
20 patients (fibroblasts pooled within patient).

Fig-2F statistics: Spearman rho and OLS slope of the low-bin mean and of the
high-bin mean against w_s (high bin: specimens with >=1 and, as sensitivity,
>=10 above-cut cells); mean-fraction Spearman rho_MF.

Generative nulls fitted per specimen on z = logit(clip(Y, 1e-7, 1-1e-7)).
No exact zeros/ones occur (checked and reported), so the zero-mass term is empty.
  C1g  continuous location shift: z ~ N(mu_s, sigma^2) with pooled sigma
  C2g  continuous location-scale: z ~ N(mu_s, sigma_s^2)
  C1e  continuous location shift of a shared empirical centered shape
       (pooled within-specimen residuals z - mean_s z), as in the paper's C1
  M1g  fixed-component mixture: two shared logit-Gaussian components (GMM on
       all pooled z, outcome blind), specimen weight w_s by fixed-component EM
  H1e  fixed empirical components: pooled cells below / above the cut,
       specimen weight = observed fraction above the cut
Parametric bootstrap (B=1000 per null, each specimen keeps its cell count) of
every statistic; reported as null median, 2.5/97.5 percentiles and the
two-sided tail probability of the observed value.
LR test of a specimen-specific high-component location (M2g) against M1g with
a parametric bootstrap under M1g (B=200).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_common import RES, FROZEN_CUT, dump_json, load_cells, write_manifest  # noqa: E402

EPS = 1e-7
rng = np.random.default_rng(20261010)


def logit(y):
    y = np.clip(y, EPS, 1 - EPS)
    return np.log(y) - np.log1p(-y)


def expit(z):
    return 1 / (1 + np.exp(-z))


def stats(Ys: list[np.ndarray], patient_of: list[str], c: float = FROZEN_CUT) -> dict:
    allY = np.concatenate(Ys)
    lo, hi = allY[allY <= c], allY[allY > c]
    mL, mH = lo.mean(), (hi.mean() if len(hi) else np.nan)
    ybar = np.array([y.mean() for y in Ys]); w = np.array([(y > c).mean() for y in Ys])
    nh = np.array([(y > c).sum() for y in Ys])
    yL = np.array([y[y <= c].mean() if (y <= c).any() else np.nan for y in Ys])
    yH = np.array([y[y > c].mean() if (y > c).any() else np.nan for y in Ys])
    A = (1 - w) * mL + w * mH
    R = ybar - A
    out = {}
    vY = ybar.var(ddof=1)
    out["var_share_A"] = A.var(ddof=1) / vY
    out["var_share_R"] = R.var(ddof=1) / vY
    out["cov_share_2AR"] = 2 * np.cov(A, R)[0, 1] / vY
    out["R2_A"] = np.corrcoef(A, ybar)[0, 1] ** 2
    out["rho_MF"] = spearmanr(ybar, w)[0]
    out["rms_R"] = np.sqrt(np.mean(R ** 2))
    okL = ~np.isnan(yL)
    out["rho_low_vs_w"] = spearmanr(w[okL], yL[okL])[0]
    out["slope_low_vs_w"] = np.polyfit(w[okL], yL[okL], 1)[0]
    for thr in (1, 10):
        ok = nh >= thr
        if ok.sum() >= 3:
            out[f"rho_high_vs_w_min{thr}"] = spearmanr(w[ok], yH[ok])[0]
            out[f"slope_high_vs_w_min{thr}"] = np.polyfit(w[ok], yH[ok], 1)[0]
            out[f"n_high_specimens_min{thr}"] = int(ok.sum())
        else:
            out[f"rho_high_vs_w_min{thr}"] = np.nan
            out[f"slope_high_vs_w_min{thr}"] = np.nan
            out[f"n_high_specimens_min{thr}"] = int(ok.sum())
    out["n_empty_high_bin"] = int((nh == 0).sum())
    # patient level (pool fibroblasts within patient)
    pats = sorted(set(patient_of))
    Yp = [np.concatenate([y for y, p in zip(Ys, patient_of) if p == q]) for q in pats]
    yb = np.array([y.mean() for y in Yp]); wp = np.array([(y > c).mean() for y in Yp])
    Ap = (1 - wp) * mL + wp * mH; Rp = yb - Ap
    out["patient_var_share_A"] = Ap.var(ddof=1) / yb.var(ddof=1)
    out["patient_var_share_R"] = Rp.var(ddof=1) / yb.var(ddof=1)
    out["patient_R2_A"] = np.corrcoef(Ap, yb)[0, 1] ** 2
    out["patient_rho_MF"] = spearmanr(yb, wp)[0]
    return {k: float(v) for k, v in out.items()}


def em_weight(z, mu, sd, pi0=0.5, iters=200):
    """EM for the high-component weight with fixed Gaussian components (vectorised over cells)."""
    from scipy.stats import norm
    f0 = norm.pdf(z, mu[0], sd[0]); f1 = norm.pdf(z, mu[1], sd[1])
    w = pi0
    for _ in range(iters):
        r = w * f1 / (w * f1 + (1 - w) * f0 + 1e-300)
        nw = r.mean()
        if abs(nw - w) < 1e-10:
            w = nw; break
        w = nw
    ll = np.log(w * f1 + (1 - w) * f0 + 1e-300).sum()
    return w, ll


def em_m2(z, mu, sd, iters=300):
    """Weight and high-component mean free; shared low component and both SDs fixed."""
    from scipy.stats import norm
    w, m1 = 0.5, mu[1]
    f0 = norm.pdf(z, mu[0], sd[0])
    for _ in range(iters):
        f1 = norm.pdf(z, m1, sd[1])
        r = w * f1 / (w * f1 + (1 - w) * f0 + 1e-300)
        nw = r.mean(); nm = (r * z).sum() / max(r.sum(), 1e-12)
        if abs(nw - w) < 1e-10 and abs(nm - m1) < 1e-10:
            w, m1 = nw, nm; break
        w, m1 = nw, nm
    f1 = norm.pdf(z, m1, sd[1])
    return w, m1, np.log(w * f1 + (1 - w) * f0 + 1e-300).sum()


def main() -> int:
    cells = load_cells()
    fib = cells[cells.celltype == "fibroblast"]
    specs = sorted(fib.gsm.unique())
    Ys = [fib.loc[fib.gsm == s, "topic_0"].to_numpy(np.float64) for s in specs]
    pat = [fib.loc[fib.gsm == s, "patient_id"].iloc[0] for s in specs]
    ns = [len(y) for y in Ys]
    allY = np.concatenate(Ys)
    zero_info = {"exact_zero": int((allY == 0).sum()), "exact_one": int((allY == 1).sum()),
                 "below_1e-7": int((allY < EPS).sum()), "min": float(allY.min()), "max": float(allY.max())}
    obs = stats(Ys, pat)

    Zs = [logit(y) for y in Ys]
    mus = np.array([z.mean() for z in Zs]); sds = np.array([z.std() for z in Zs])
    pooled_sd = np.sqrt(sum(((z - z.mean()) ** 2).sum() for z in Zs) / sum(ns))
    resid = np.concatenate([z - z.mean() for z in Zs])
    gm = GaussianMixture(2, n_init=10, random_state=0).fit(np.concatenate(Zs).reshape(-1, 1))
    order = np.argsort(gm.means_.ravel())
    cmu = gm.means_.ravel()[order]; csd = np.sqrt(gm.covariances_.ravel()[order])
    m1 = [em_weight(z, cmu, csd) for z in Zs]
    w_m1 = np.array([a for a, _ in m1]); ll_m1 = np.array([b for _, b in m1])
    m2 = [em_m2(z, cmu, csd) for z in Zs]
    ll_m2 = np.array([c for _, _, c in m2])
    from scipy.stats import norm
    ll_c1 = np.array([norm.logpdf(z, z.mean(), pooled_sd).sum() for z in Zs])
    ll_c2 = np.array([norm.logpdf(z, z.mean(), z.std()).sum() for z in Zs])
    N = sum(ns)
    fit = {"logit_gaussian_components": {"means": cmu.tolist(), "sds": csd.tolist(),
                                         "expit_means": expit(cmu).tolist()},
           "pooled_within_sd": float(pooled_sd),
           "total_loglik": {"C1g": float(ll_c1.sum()), "C2g": float(ll_c2.sum()),
                            "M1g": float(ll_m1.sum()), "M2g": float(ll_m2.sum())},
           "n_params": {"C1g": 25 + 1, "C2g": 50, "M1g": 25 + 4, "M2g": 50 + 4}}
    fit["BIC"] = {k: float(-2 * fit["total_loglik"][k] + fit["n_params"][k] * np.log(N)) for k in fit["n_params"]}
    fit["specimens_preferring_M1g_over_C2g_loglik"] = int((ll_m1 > ll_c2).sum())
    fit["specimens_preferring_M1g_over_C1g_loglik"] = int((ll_m1 > ll_c1).sum())
    lr_obs = 2 * (ll_m2 - ll_m1).sum()
    lo_pool = allY[allY <= FROZEN_CUT]; hi_pool = allY[allY > FROZEN_CUT]
    w_obs = np.array([(y > FROZEN_CUT).mean() for y in Ys])

    def sim(null: str) -> list[np.ndarray]:
        out = []
        for i, n in enumerate(ns):
            if null == "C1g":
                z = rng.normal(mus[i], pooled_sd, n)
            elif null == "C2g":
                z = rng.normal(mus[i], sds[i], n)
            elif null == "C1e":
                z = mus[i] + rng.choice(resid, n, replace=True)
            elif null == "M1g":
                h = rng.random(n) < w_m1[i]
                z = np.where(h, rng.normal(cmu[1], csd[1], n), rng.normal(cmu[0], csd[0], n))
            elif null == "H1e":
                h = rng.random(n) < w_obs[i]
                z = None
                y = np.where(h, rng.choice(hi_pool, n), rng.choice(lo_pool, n))
                out.append(y); continue
            out.append(expit(z))
        return out

    B = 1000
    nulls = {}
    rows = []
    for null in ["C1g", "C2g", "C1e", "M1g", "H1e"]:
        draws = pd.DataFrame([stats(sim(null), pat) for _ in range(B)])
        draws.insert(0, "null", null)
        rows.append(draws)
        summ = {}
        for k, v in obs.items():
            d = draws[k].dropna().to_numpy()
            if len(d) == 0 or np.isnan(v):
                continue
            summ[k] = {"observed": v, "null_median": float(np.median(d)),
                       "null_2.5": float(np.quantile(d, 0.025)), "null_97.5": float(np.quantile(d, 0.975)),
                       "p_two_sided": float(min(1.0, 2 * min(np.mean(d >= v), np.mean(d <= v)))),
                       "n_valid_draws": int(len(d))}
        nulls[null] = summ
    pd.concat(rows).to_csv(RES / "p1_3_bootstrap_draws.csv.gz", index=False)

    # LR bootstrap under M1g
    lr_null = []
    for _ in range(200):
        Ysim = sim("M1g")
        Zsim = [logit(y) for y in Ysim]
        l1 = sum(em_weight(z, cmu, csd)[1] for z in Zsim)
        l2 = sum(em_m2(z, cmu, csd)[2] for z in Zsim)
        lr_null.append(2 * (l2 - l1))
    lr_null = np.array(lr_null)
    fit["LR_M2g_vs_M1g"] = {"observed": float(lr_obs), "null_median": float(np.median(lr_null)),
                            "null_95": float(np.quantile(lr_null, 0.95)),
                            "p_parametric_bootstrap": float((1 + np.sum(lr_null >= lr_obs)) / (1 + len(lr_null))),
                            "B": 200}
    spec = pd.DataFrame({"gsm": specs, "patient_id": pat, "n_fib": ns, "mean": [y.mean() for y in Ys],
                         "w_cut": w_obs, "w_M1g": w_m1, "logit_mean": mus, "logit_sd": sds,
                         "high_mean_M2g_expit": expit(np.array([b for _, b, _ in m2])),
                         "w_M2g": [a for a, _, _ in m2], "ll_C1g": ll_c1, "ll_C2g": ll_c2,
                         "ll_M1g": ll_m1, "ll_M2g": ll_m2})
    spec.to_csv(RES / "p1_3_specimen_fits.csv", index=False)
    res = {"zero_mass_check": zero_info, "observed": obs, "model_fit": fit, "bootstrap_B": B, "nulls": nulls}
    dump_json(res, RES / "p1_3_prevalence_intensity.json")
    write_manifest("p1_3_prevalence_intensity")
    import json
    print(json.dumps({"zero": zero_info, "observed": obs, "fit": fit}, indent=1))
    keys = ["var_share_A", "R2_A", "rho_MF", "rho_low_vs_w", "slope_low_vs_w", "rho_high_vs_w_min1",
            "slope_high_vs_w_min1", "rho_high_vs_w_min10", "slope_high_vs_w_min10", "patient_var_share_A", "n_empty_high_bin"]
    for null, s in nulls.items():
        print(null, {k: (round(s[k]["observed"], 3), round(s[k]["null_median"], 3), round(s[k]["null_2.5"], 3),
                         round(s[k]["null_97.5"], 3), round(s[k]["p_two_sided"], 3)) for k in keys if k in s})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
