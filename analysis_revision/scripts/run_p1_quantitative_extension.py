#!/usr/bin/env python3
"""Patient covariate balance and a full-array bulk wound contrast.

The covariate table uses the 11 mapped DFU patients already scored for topic 0.
The bulk contrast uses GSE80178 sample-level HuGene signals and a frozen
decoder, after mapping transcript clusters through Ensembl. It does not use the
submitters' differentially expressed gene list as the primary universe.
"""
from __future__ import annotations

import gzip
import itertools
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/quantitative-extension-20261011"
PATIENTS = ROOT / "outputs/followup_20261010/paper1/results/p1_5_patient_table.csv"
MAP = ROOT / "metadata_intake/GSE165816_ncomms_subject_sample_map.csv"
SERIES = ROOT / "data/raw/GSE80178/GSE80178_series_matrix.txt.gz"
PROCESSED = ROOT / "data/raw/GSE80178/GSE80178_further_processed_data.txt.gz"
PANEL = ROOT / "outputs/direction1_dfu_mainline/panel.npy"
BETA = ROOT / "outputs/direction1_dfu_mainline/beta.npy"
MAPPING = OUT / "hugene20_hgnc.tsv"
TOL = 1e-12
COVARIATES = ("age", "bmi", "hba1c", "wound_area_cm2", "dm_duration_years")
READOUTS = ("S", "M", "W", "F")


def dump(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(payload, indent=2) + "\n")


def probe_ids() -> list[str]:
    with gzip.open(SERIES, "rt") as handle:
        lines = handle.read().splitlines()
    start = lines.index("!series_matrix_table_begin")
    end = lines.index("!series_matrix_table_end")
    return [line.split("\t", 1)[0].strip('"') for line in lines[start + 2:end]]


def download_mapping() -> pd.DataFrame:
    if MAPPING.is_file() and MAPPING.stat().st_size > 1000:
        frame = pd.read_csv(MAPPING, sep="\t")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        probes = probe_ids()
        chunks = [probes[offset:offset + 400] for offset in range(0, len(probes), 400)]

        def fetch(chunk: list[str]) -> str:
            query = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="0" datasetConfigVersion="0.6">
  <Dataset name="hsapiens_gene_ensembl" interface="default">
    <Filter name="affy_hugene_2_0_st_v1" value="{",".join(chunk)}"/>
    <Attribute name="affy_hugene_2_0_st_v1"/>
    <Attribute name="hgnc_symbol"/>
  </Dataset>
</Query>"""
            url = "https://www.ensembl.org/biomart/martservice?query=" + urllib.parse.quote(query)
            with urllib.request.urlopen(url, timeout=90) as response:
                return response.read().decode()

        rows = ["probe\tsymbol"]
        done = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            for text in pool.map(fetch, chunks):
                rows.extend(line for line in text.splitlines() if line and not line.startswith("Query ERROR"))
                done += 1
                print(f"mapped chunks {done}/{len(chunks)}", flush=True)
        MAPPING.write_text("\n".join(rows) + "\n")
        frame = pd.read_csv(MAPPING, sep="\t")
    frame.columns = ["probe", "symbol"]
    frame = frame.dropna(subset=["probe"])
    frame["probe"] = frame["probe"].map(lambda value: str(int(float(value))))
    frame["symbol"] = frame["symbol"].fillna("").astype(str).str.strip()
    frame = frame[frame["symbol"].ne("") & frame["symbol"].ne("nan")]
    # A probe with more than one symbol is not assigned.
    counts = frame.groupby("probe")["symbol"].nunique()
    frame = frame[frame["probe"].isin(counts[counts == 1].index)]
    return frame.drop_duplicates()


def exact_rank_p(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    pool = np.concatenate([a, b])
    ranks = pd.Series(pool).rank(method="average").to_numpy()
    n1 = len(a)
    observed = float(ranks[:n1].sum())
    null = np.array([
        ranks[list(choice)].sum()
        for choice in itertools.combinations(range(len(pool)), n1)
    ])
    center = float(null.mean())
    probability = float(np.mean(np.abs(null - center) >= abs(observed - center) - TOL))
    return probability, int(len(null))


def hodges_lehmann(a: np.ndarray, b: np.ndarray) -> float:
    differences = (a[:, None] - b[None, :]).ravel()
    return float(np.median(differences))


def spearman(x: np.ndarray, y: np.ndarray, draws: int = 20000, seed: int = 20261011) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return {"n": int(len(x)), "rho": None, "p_permutation": None}
    rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    rng = np.random.default_rng(seed)
    null = np.empty(draws)
    for i in range(draws):
        null[i] = pd.Series(x).corr(pd.Series(rng.permutation(y)), method="spearman")
    probability = float(np.mean(np.abs(null) >= abs(rho) - TOL))
    return {"n": int(len(x)), "rho": rho, "p_permutation": probability, "draws": draws}


def iqr(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 25)), float(np.percentile(values, 75))]


def covariate_report() -> dict:
    patients = pd.read_csv(PATIENTS)
    if len(patients) != 11 or set(patients["healed"]) != {True, False}:
        raise ValueError("patient table is not the 11 DFU outcome patients")
    clinical = pd.read_csv(MAP)
    foot = clinical[(clinical["tissue_site"] == "foot") & clinical["group"].isin(["DFU-H", "DFU-NH"])]
    foot = foot.drop_duplicates("patient_id")
    if set(foot["patient_id"]) != set(patients["patient_id"]):
        raise ValueError("foot clinical rows do not match the scored patients")
    merged = patients.merge(
        foot[["patient_id", "healing_status", *COVARIATES, "sex"]],
        on="patient_id", suffixes=("", "_map"))
    for column in (*COVARIATES, "sex"):
        if not merged[column].astype(str).equals(merged[column + "_map"].astype(str)):
            # NaN equals fails; compare numerically where possible.
            if column == "sex":
                if not merged[column].equals(merged[column + "_map"]):
                    raise ValueError(f"sex disagrees with the clinical map for {column}")
            else:
                left = pd.to_numeric(merged[column], errors="coerce")
                right = pd.to_numeric(merged[column + "_map"], errors="coerce")
                if not np.allclose(left.fillna(-999), right.fillna(-999), equal_nan=True):
                    raise ValueError(f"{column} disagrees with the clinical map")
    healed = merged["healed"].to_numpy(bool)
    rows = []
    for column in COVARIATES:
        values = pd.to_numeric(merged[column], errors="coerce").to_numpy(float)
        a, b = values[healed], values[~healed]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        probability, n_alloc = exact_rank_p(a, b)
        rows.append({
            "covariate": column,
            "healed_n": int(len(a)),
            "healed_median": float(np.median(a)),
            "healed_iqr": iqr(a),
            "nonhealed_n": int(len(b)),
            "nonhealed_median": float(np.median(b)),
            "nonhealed_iqr": iqr(b),
            "mean_difference": float(a.mean() - b.mean()),
            "hodges_lehmann": hodges_lehmann(a, b),
            "p_exact_rank": probability,
            "n_allocations": n_alloc,
        })
    sex = merged["sex"].to_numpy()
    # Healed male, healed female, nonhealed male, nonhealed female.
    table = [
        [int(((sex == "M") & healed).sum()), int(((sex == "F") & healed).sum())],
        [int(((sex == "M") & ~healed).sum()), int(((sex == "F") & ~healed).sum())],
    ]
    # Fisher exact over the same 2x2, two-sided, using enumeration of the
    # hypergeometric probability rather than an imported p-value helper.
    a_male, a_female = table[0]
    b_male, b_female = table[1]
    n_male = a_male + b_male
    n_healed = a_male + a_female
    n = n_male + a_female + b_female
    def hyper(k):
        return (comb(n_male, k) * comb(n - n_male, n_healed - k)) / comb(n, n_healed)
    lo = max(0, n_healed - (n - n_male))
    hi = min(n_male, n_healed)
    observed = hyper(a_male)
    fisher_p = float(sum(hyper(k) for k in range(lo, hi + 1) if hyper(k) <= observed + 1e-15))
    associations = []
    for readout in READOUTS:
        y = merged[readout].to_numpy(float)
        for column in COVARIATES:
            x = pd.to_numeric(merged[column], errors="coerce").to_numpy(float)
            associations.append({"readout": readout, "covariate": column, **spearman(x, y)})
    return {
        "n_patients": 11,
        "n_healed": int(healed.sum()),
        "n_nonhealed": int((~healed).sum()),
        "balance": rows,
        "sex": {"table_healed_then_nonhealed_male_female": table, "p_fisher": fisher_p},
        "associations": associations,
    }


def comb(n, k):
    if k < 0 or k > n:
        return 0
    value = 1
    for i in range(k):
        value = value * (n - i) // (i + 1)
    return value


def series_matrix():
    with gzip.open(SERIES, "rt") as handle:
        lines = handle.read().splitlines()
    titles = [item.strip('"') for item in lines[[i for i, line in enumerate(lines) if line.startswith("!Sample_title")][0]].split("\t")[1:]]
    start = lines.index("!series_matrix_table_begin")
    end = lines.index("!series_matrix_table_end")
    frame = pd.read_csv(StringIO("\n".join(lines[start + 1:end])), sep="\t", index_col=0)
    frame.index = frame.index.astype(str)
    return frame, titles


def gene_matrix(expr: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    mapped = expr.join(mapping.set_index("probe"), how="inner")
    grouped = mapped.groupby("symbol").mean(numeric_only=True)
    centered = grouped.sub(grouped.mean(axis=1), axis=0)
    scale = grouped.std(axis=1, ddof=1).replace(0, np.nan)
    return centered.div(scale, axis=0).dropna()


def exact_group(score: np.ndarray, group_a: np.ndarray, group_b: np.ndarray) -> dict:
    a = score[group_a]
    b = score[group_b]
    observed = float(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    null = np.array([
        pool[list(choice)].mean() - np.delete(pool, list(choice)).mean()
        for choice in itertools.combinations(range(len(pool)), len(a))
    ])
    return {
        "difference": observed,
        "p_exact": float(np.mean(np.abs(null) >= abs(observed) - TOL)),
        "n_relabellings": int(len(null)),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


def random_fraction(z: pd.DataFrame, genes: list[str], group_a, group_b, draws: int = 4000) -> dict:
    matrix = z.to_numpy(float)
    positions = {gene: i for i, gene in enumerate(z.index)}
    observed_score = matrix[[positions[gene] for gene in genes]].mean(axis=0)
    observed = float(observed_score[group_a].mean() - observed_score[group_b].mean())
    rng = np.random.default_rng(20261011)
    null = np.empty(draws)
    width = len(genes)
    for i in range(draws):
        score = matrix[rng.choice(matrix.shape[0], size=width, replace=False)].mean(axis=0)
        null[i] = score[group_a].mean() - score[group_b].mean()
    return {
        "random_draws": draws,
        "fraction_at_least_as_large": float(np.mean(null >= observed - TOL)),
        "fraction_at_least_as_extreme": float(np.mean(np.abs(null) >= abs(observed) - TOL)),
        "random_median": float(np.median(null)),
    }


def score_universe(z: pd.DataFrame, beta: np.ndarray, panel: np.ndarray, groups: dict, name: str) -> dict:
    b0 = pd.Series(beta[0], index=panel)
    lift = pd.Series(beta[0] / np.maximum(beta.mean(axis=0), 1e-12), index=panel)
    overlap = z.index.intersection(panel)
    top = list(b0.loc[overlap].sort_values(ascending=False).index[:50])
    high = b0[b0 > b0.quantile(0.9)].index.intersection(overlap)
    lifted = list(lift.loc[high].sort_values(ascending=False).index[:50])
    weights = b0.loc[overlap]
    weights = weights / weights.sum()
    weighted = z.loc[overlap].mul(weights, axis=0).sum(axis=0)
    signatures = {
        "top50_decoder": top,
        "lift50_decoder": lifted,
    }
    out = {"universe": name, "n_genes": int(z.shape[0]), "n_panel_overlap": int(len(overlap)), "signatures": {}}
    for label, genes in signatures.items():
        score = z.loc[genes].mean(axis=0).to_numpy()
        record = {"n_genes": len(genes), "genes": genes}
        for contrast, a, b in groups:
            record[contrast] = exact_group(score, a, b)
            record[contrast].update(random_fraction(z, genes, a, b))
        out["signatures"][label] = record
    score = weighted.to_numpy()
    record = {"n_genes": int(len(overlap))}
    for contrast, a, b in groups:
        record[contrast] = exact_group(score, a, b)
    out["signatures"]["beta_weighted_overlap"] = record
    return out


def bulk_report(mapping: pd.DataFrame) -> dict:
    expr, titles = series_matrix()
    if len(titles) != 12 or expr.shape[1] != 12:
        raise ValueError("GSE80178 does not have 12 samples")
    groups_named = np.array(["DFU" if "Ulcer" in title else ("NDM" if title.startswith("non-") else "DM") for title in titles])
    if sorted(zip(*np.unique(groups_named, return_counts=True))) != [("DFU", 6), ("DM", 3), ("NDM", 3)]:
        raise ValueError(f"unexpected group counts: {groups_named}")
    z_full = gene_matrix(expr, mapping)
    processed = pd.read_csv(PROCESSED, sep="\t")
    if not ((processed["FDR p-value (DFU vs. FS)"] < 0.05).all() and (processed["Fold Change (linear) (DFU vs. FS)"].abs() >= 2).all()):
        raise ValueError("processed file is not the expected DE-filtered table")
    de_symbols = set(processed["Gene Symbol"].dropna().astype(str).str.split(" /// ").str[0].str.strip())
    z_de = z_full.loc[z_full.index.isin(de_symbols)]
    panel = np.load(PANEL, allow_pickle=True).astype(str)
    beta = np.load(BETA)
    dfu, dm, ndm = groups_named == "DFU", groups_named == "DM", groups_named == "NDM"
    contrasts = [
        ("DFU_vs_intact", dfu, dm | ndm),
        ("DFU_vs_diabetic_foot", dfu, dm),
        ("diabetic_vs_nondiabetic_foot", dm, ndm),
    ]
    return {
        "n_probes": int(expr.shape[0]),
        "n_mapped_genes_full": int(z_full.shape[0]),
        "n_de_filtered_genes_present": int(z_de.shape[0]),
        "de_filter": "submitter table keeps only genes with FDR < 0.05 and absolute linear fold change at least 2",
        "samples": dict(zip(titles, groups_named.tolist())),
        "full_array": score_universe(z_full, beta, panel, contrasts, "full_mapped_array"),
        "de_filtered_sensitivity": score_universe(z_de, beta, panel, contrasts, "submitter_DE_genes"),
    }


def main() -> None:
    mapping = download_mapping()
    payload = {
        "covariates": covariate_report(),
        "bulk": bulk_report(mapping),
        "limits": [
            "Eleven outcome patients are the same discovery patients.",
            "Balance tests are not covariate adjustment and do not exclude confounding.",
            "GSE80178 has no healing labels. It tests a frozen gene score in bulk wound versus intact skin.",
            "The DE-filtered universe is a sensitivity, not the primary contrast.",
            "Random signatures and sample relabellings do not add patients.",
        ],
    }
    dump(payload)
    if payload["bulk"]["n_mapped_genes_full"] < 1000:
        raise ValueError("gene mapping did not cover the array")
    primary = payload["bulk"]["full_array"]["signatures"]["top50_decoder"]["DFU_vs_intact"]
    print(json.dumps({
        "age_p": payload["covariates"]["balance"][0]["p_exact_rank"],
        "age_hl": payload["covariates"]["balance"][0]["hodges_lehmann"],
        "full_top50_diff": primary["difference"],
        "full_top50_p": primary["p_exact"],
        "full_genes": payload["bulk"]["n_mapped_genes_full"],
    }, indent=2))


if __name__ == "__main__":
    main()
