# Paper 1 follow-up analyses (reviewer items P1-1 to P1-6), 2026-10-10

Everything in this report was computed from local files only. Nothing outside `outputs/followup_20261010/paper1/` was modified. The frozen model and gate were not refitted.

- **Scripts:** `scripts/`
- **Results:** `results/`
- **Logs:** `logs/`
- **SHA-256 manifest:** `MANIFEST_SHA256.json`. It covers 51 read-only repository inputs, the cited provenance documents and every file in this folder.
- **Python:** `/home/zeyufu/miniconda3/envs/dl/bin/python` (torch 2.12.1+cu130, scanpy 1.11.4).

**Common definitions**
- Unit: the 11 author-mapped DFU patients (7 healed, 4 non-healed). Cells are pooled within each patient.
- Readouts:
  - **S**: all-cell mean topic-0 loading (Eq. 8).
  - **M**: fibroblast mean loading.
  - **W**: fraction of fibroblasts above the frozen cut 0.1842958.
  - **F**: fibroblast composition (fibroblasts / all cells).
- Healing tests use exact enumeration of all 330 allocations, two-sided on |Δ|, with no add-one term unless stated.

**Reproduction checks (all pass)**
- Baseline healing contrasts reproduce the manuscript exactly:
  - S 0.0293, p = 90/330 = 0.2727
  - M 0.0892, p = 0.3121
  - W 0.1514, p = 0.3697
  - F 0.0708, p = 0.4697
- Re-projecting the 25 discovery specimens from `GSE165816_RAW.tar` through the frozen whole pipeline matches `cells.parquet`:
  - All 76,461 cells reproduced, with maximum |Δθ| = 4.8×10⁻⁷.
  - Lineage labels agree 100%, including all 19,410 fibroblasts.
  - Pipeline used: RT06 functions in `scripts/55_full_pipeline_depth_audit.py`, imported read-only. QC is ≥200 genes and mito ≤0.20; the gate is the frozen 76,461-cell marker reference; the score is softmax(μ).
- Refitting the per-topic two-Gaussian midpoint for topic 0 with the original recipe (`scripts/13`: n_init = 10, random_state = 0) gives 0.184297, versus the frozen 0.184296.
- Nine-fit saved equal-specimen patient means are reproduced with |Δ| = 0.

---

## P1-1 Selection-aware healing inference (`p1_1_selection.py`)

### How topic 0 and the 23 anchor genes were chosen (repository record)

**Timeline of the fit and the topic-0 naming**
- The frozen fit ran on 2026-09-17 from 04:47 to 04:52 (`logs/direction1_mainline.log`).
- Line 37 of that log prints `mean theta by arm` for all 15 topics. Topic 0 has the largest healer/non-healer ratio there: 0.0753/0.0267 = 2.82, against 2.01 for the next topic (topic 2).
- `docs/05_DATASET_AUDIT.md` §5, lines 134–143, labels topic 0 the **"阳性对照" (positive control)**. It cites the "2.8×" healer/non-healer ratio and says CHI3L1/TIMP1 are the healing-associated HE-Fibro markers reported by the source papers. The section's own caveat says these figures are cell-pooled arm means from the pre-repair stochastic projection.
- `scripts/11_validate_gse231643.py` (file time 05:29, about 37 minutes after the fit) opens with "The discovery run … found topic 0 = TIMP1/CHI3L1/COL3A1/FN1 loaded ~2.8x higher in DFU-healer cells" (lines 5–8).
- No file predating the fit names topic 0 or a TIMP1/CHI3L1 programme as the endpoint.
- Topic indices are arbitrary. In an earlier run without technical-gene exclusion, topic 0 was a mitochondrial topic (`docs/05_DATASET_AUDIT.md` lines 88 and 127).

**Conclusion on topic choice.** The record is consistent with the reviewer's reading. Topic 0 was named the programme of interest after the arm-wise outcome means were printed, and it was the maximum-ratio topic. Its gene content also matched the source paper's healer-associated markers. Both facts were used.

**The 23 anchor genes**
- They are hard-coded in `scripts/10_run_dfu_mainline.py` lines 41–49 as `PINNED_REGULATORS`, with the comment "Regulators the wound-healing literature puts on the arrest axis". They are pinned into the panel only; loss weights are unchanged.
- They were in place at fit time: the log line 8 reads "pinned present 23/23", and they are listed in `report.json` `target_line.pinned_present` (lines 35–58).
- Partial pre-fit sources are in git-tracked configs:
  - `config/pipeline_config.yaml` (commit 0310ad63): CDKN1A, FOS, JUNB, KRT14, KRT5, TNF, IL1B, VEGFA.
  - `config/datasets_matrix.json` (commit d9ca82d4): PRG4/THY1 (synovial), IL1B, MMP3, VEGFA.
- No tracked or pre-fit document lists CHI3L1, TIMP1, MMP1, MMP13, COL7A1, NRG1, ASPN, TLR4, NLRP3, IFNG, CD68 or CD163. `scripts/10` is untracked and its file time is 21:01, after the fit.
- docs/05 calls CHI3L1/TIMP1 source-paper healer markers. The anchor list therefore contains literature healer-associated genes, chosen without using this dataset's labels.
- Four anchors are among topic 0's top 16 decoder genes: TIMP1 #1, CHI3L1 #8, MMP1 #9 and MMP3 #16.
- Not tested here: which anchors would have entered the panel by dispersion alone. That needs the reviewer's item-5 refit without anchors.

**Side answer to the reviewer's open question.** `scripts/10` has defaults `--batch-size 512` and `--steps 20000`. The log records 40,000 steps, so the frozen fit was run with a non-default step count at the default batch size of 512.

### Selection-adjusted results (11 patients, 330 allocations)

For each allocation the maximum over all 15 topics is taken. Selection-adjusted p is the share of allocations with a maximum at least as large as the observed one.

| Readout | Topic 0 ratio (rank /15) | Topic 0 marginal p (two-sided Δ; one-sided ratio) | Observed max ratio (topic) | **Selection-adjusted p, max ratio** | Selection-adjusted p, max \|log ratio\| / max \|Δ\| / max \|t\| |
|---|---|---|---|---|---|
| S all-cell mean | 1.957 (1) | 0.273; 0.118 | 1.957 (0) | **0.785** | 0.633 / 0.827 / 0.894 |
| M fibroblast mean | 2.066 (1) | 0.312; 0.164 | 2.066 (0) | **0.609** | 0.394 / 0.645 / 0.861 |
| W fraction > per-topic cut | 2.012 (3) | 0.370; 0.185 | 3.291 (14) | **0.885** | 0.655 / 0.685 / 0.800 |
| C cell-pooled arm mean (the statistic that was originally inspected) | 3.089 (1) | 0.276; 0.085 | 3.089 (0) | **0.245** | 0.279 / 0.530 / – |

- Joint maximum |log ratio| over S, M and W: selection-adjusted p = 0.655.
- Under the null, the median maximum ratio is 2.28 for S and 2.60 for C. The observed topic-0 ratios (1.96 and 3.09) are ordinary values for "the best of 15".
- With the per-topic midpoint, topic 0 is not even the top W topic: topic 14 has ratio 3.29 and topic 5 has 2.47.

**Selection loops** (the topic is chosen inside each fold by maximum ratio; the whole loop is rerun under all 330 allocations):

| Readout | LOPO AUC (one-sided exact p) | Held-pair AUC, 28 pairs (one-sided exact p) | Topic chosen in folds |
|---|---|---|---|
| S | 0.500 (0.518) | 0.393 (0.682) | LOPO: topic 0 in 8/11, topic 2 in 3/11; pairs: topic 0 17, topic 2 10, topic 1 1 |
| M | 0.571 (0.439) | 0.500 (0.521) | LOPO: topic 0 in 10/11; pairs: topic 0 19, topic 2 5, topic 14 4 |
| W | 0.393 (0.691) | 0.393 (0.700) | LOPO: topic 14 in 8/11, topic 0 in 1/11 |

With topic choice inside the loop, out-of-sample discrimination is at chance. The fixed topic-0 held-pair value of 0.714 (20/28, manuscript Eq. 14) is conditional on a topic picked with all 11 patients' labels.

**Implications**
- The claim that point estimates are positive in every sensitivity check and that the direction "warrants prospective evaluation" is **overturned as evidence**. Topic 0's positive direction is what selecting the best of 15 topics produces. Selection-adjusted p is 0.61–0.89 on the patient readouts, and 0.245 on the statistic that was actually inspected.
- The claim that healing association is unvalidated is **supported**, and more strongly than the manuscript states.
- The topic-0 provenance should be disclosed in Methods, citing the files above.

---

## P1-2 Diabetes / wound / site decomposition (`p1_2_project_excluded.py`, `p1_2_decomposition.py`)

**What was projected.** Eight diabetic non-ulcer foot specimens (G1A, G2A, G3, G3A, G5, G38, G41, G46; patients S12–S17) and all 11 forearm specimens went through the same frozen QC → gate → encoder.

**Coverage caveat.** The "A" specimens G1A, G2A and G3A have low panel coverage: 2,998–3,258 of 7,002 genes present in the released matrix, against 4,937–6,512 for the others. G38 and G46 have 1,197 and 1,171 QC cells. All are retained.

### Patient-level fibroblast fraction W (topic 0 > 0.1843)

- **Non-diabetic intact foot (H, 9 patients):** 0, 0, 0, 0, 0, 0, 0.001, 0.019 and **0.841 (S19)**.
- **Diabetic non-ulcer foot (DM, 6 patients):** 0, 0, 0.004, 0.005, 0.014 and **0.330 (S16; G1A 0.322 and G3A 0.347)**.
- **DFU, healed (7):** 0.023, 0.048, 0.082, 0.296, 0.372, 0.586, 0.699.
- **DFU, non-healed (4):** 0, 0.003, 0.084, 0.511.
- **Forearm (11 specimens, including 5 DFU-patient forearms):** every value ≤ 0.0014.

### Exact patient-level contrasts

All relabellings of the two compared groups are enumerated. Each cell gives Δ, then the permutation p for the mean, then the exact Mann–Whitney p.

| Contrast | W | M | S | Hurdle W > 0.02 (rate; exact p) |
|---|---|---|---|---|
| DM vs H (diabetes) | −0.037; 0.947; 0.337 | −0.020; 0.968; 0.145 | −0.007; 0.975; 0.224 | 1/6 vs 1/9; 1.00 |
| DFU vs DM (wound within diabetes) | +0.187; 0.110; 0.061 | +0.094; 0.113; 0.098 | +0.021; 0.294; 0.149 | 9/11 vs 1/6; **0.018** |
| DFU vs H | +0.150; 0.229; **0.011** | +0.074; 0.262; **0.031** | +0.013; 0.593; **0.038** | 9/11 vs 1/9; **0.0055** |
| DFU vs all intact foot (DM + H, 15 patients) | +0.165; 0.099; **0.0055** | +0.082; 0.113; **0.018** | +0.016; 0.433; **0.027** | 9/11 vs 2/15; **0.00095** |

- The three non-diabetic rows reproduce the reviewer's rank-test figure of p ≈ 0.013 (here 0.011).
- Mean-difference permutation tests stay non-significant because the two intact-skin outliers (S19, S16) dominate them.
- F does not differ in any contrast: DFU vs intact Δ = −0.002, p = 0.97.

### Site (paired foot minus forearm, exact sign flip)

| Pair set | n | ΔW (p) | ΔM (p) | ΔS (p) |
|---|---|---|---|---|
| All 10 pairs (the manuscript's "anatomical positive control") | 10 | 0.268 (0.016) | 0.139 (0.014) | 0.0476 (0.0098) |
| DFU patients only (S01–S05) | 5 | 0.364 (0.125) | 0.182 (0.125) | 0.054 (0.0625) |
| Intact skin only (S12, S13, S19, S20, S21) | 5 | 0.172 (0.25) | 0.096 (0.0625) | 0.041 (0.31) |
| Intact skin without S19 | 4 | 0.0055 (0.50) | 0.0068 (0.125) | 0.0011 (0.625) |

- Per-patient ΔW: S01 0.51, S02 0.59, S04 0.70, S19 0.84. All others are between 0 and 0.023, including DFU patients S03 and S05.
- With five intact pairs, the smallest attainable p is 0.0625.
- The interaction test (DFU pair difference minus intact pair difference) has ΔW = 0.19, exact p = 0.44.
- The frozen-gate values agree with the manuscript's anatomical-control file: |Δ| ≤ 7.5×10⁻⁹ all-cell and ≤ 0.003 fibroblast mean. That control re-standardised the gate on foot plus forearm cells.

**Implications**
- **"Responds to a known tissue difference" is overturned as an anatomy claim.**
  - The 10-pair effect is carried by three DFU ulcer–forearm pairs and the single intact outlier S19.
  - In intact skin without S19, the foot–forearm difference is ≤ 0.007 on every readout.
  - It should be relabelled a wound(+outlier) contrast.
- **High-loading fibroblasts are not seen in forearm skin.** In all 11 forearms, including those of DFU patients, W ≤ 0.0014.
- **The healthy-skin "counterexample" framing is weakened.** The better description:
  - The above-cut state is common in DFU (9/11 with W > 0.02).
  - It is rare in intact foot skin (2/15: S19 non-diabetic and S16 diabetic).
  - It is absent in forearm.
  - There is no diabetes effect in intact skin (DM vs H: p = 0.95 for the mean, 0.34 for the rank test).
- **The positive construct result is only partial.** The reviewer's criterion "DFU > DM ≈ H" holds on rank and hurdle statistics. It does not hold on mean-difference permutation tests, and two intact-skin patients have DFU-level fractions.
- **This does not test healing.** Healed DFU vs DM gives rank p = 0.014 and hurdle p = 0.0047. Non-healed DFU vs DM gives rank p = 0.75. These are post hoc and add nothing beyond P1-1.

---

## P1-3 Prevalence vs intensity against generative nulls (`p1_3_prevalence_intensity.py`)

The data are 19,410 fibroblasts in 25 specimens. There are no exact zeros or ones (minimum 0.0040), so the zero-mass term is empty.

### Descriptive variance split (Eq. 7)

| | Specimens (25) | Patients (20) |
|---|---|---|
| Var(A)/Var(Ȳ) | **0.884** | **0.891** |
| Var(R)/Var(Ȳ) | 0.012 | 0.010 |
| 2Cov(A,R)/Var(Ȳ) | 0.104 | – |
| R² of Ȳ on A | 0.992 | 0.992 |

- At fixed pooled bin means, about 88% of between-specimen variance in the fibroblast mean comes from the prevalence term.
- About 1% comes from within-bin deviation alone.
- About 10% comes from the positive co-movement of the two: bin means rise with prevalence.
- The reviewer's "98.8%" corresponds to R², not the variance share. Their own Var(A)/Var(total) = 0.0191/0.0216 = 0.884.

### Fig. 2F statistics and model comparison

- Observed: low-bin mean vs w has ρ = 0.903 and slope 0.083. High-bin mean vs w has ρ = 0.824 and slope 0.312 (16 specimens with ≥1 above-cut cell), or ρ = 0.734 and slope 0.266 (12 specimens with ≥10). Nine specimens have an empty high bin.
- **Fixed-component mixture nulls cannot produce the rising high-bin mean.** Each null below is a parametric bootstrap with B = 1,000 and the specimens' cell counts. "Outside all 1,000 draws" means p < 0.002.
  - M1g (two shared logit-Gaussian components, specimen weight by EM): high-bin slope null median 0.003, 95% range −0.175 to 0.143.
  - H1e (empirical below/above-cut components): high-bin slope null median 0.002, 95% range −0.127 to 0.110.
  - The observed slope of 0.312 lies outside all 1,000 draws under both. Restricting to specimens with ≥10 above-cut cells, the observed 0.266 is still outside: M1g p = 0.0, H1e p = 0.0.
- **Continuous location-shift nulls reproduce rising bin means of the right order:**
  - C1g (shared-SD logit-normal): high slope 0.258 (0.225–0.285); the ≥10 version has p = 0.37.
  - C2g (specimen SD): 0.376.
  - C1e (shared empirical residual shape, as in the paper's C1): 0.143.
- **No null fits everything.**
  - Variance share: observed 0.884. Continuous nulls give 0.68–0.84 (all p < 0.002); M1g gives 0.82 and H1e 1.00.
  - Empty high bins: observed 9. C1g gives 1, C1e 0, C2g 13, M1g 6 (p = 0.028), H1e 10 (p = 0.80).
  - ρ_MF does not discriminate: observed 0.933, null medians 0.92 (C2g) to 0.99 (M1g). It is reproduced by the continuous nulls (C1g p = 0.38, C1e p = 0.18).
- **Likelihood ratio for a specimen-specific high-component location (M2g vs M1g):** 2LR = 2,037, against a parametric-bootstrap null median of 19 (95th percentile 29). p = 1/201, the smallest attainable with 200 replicates.
- **BIC ranking:** M2g 43,404 < C2g 45,080 < M1g 45,194 < C1g 60,258.

**Implications**
- **The construct question can now be answered quantitatively.** Between-sample variation in the fibroblast mean is prevalence-dominated (Var(A)/Var(Ȳ) = 0.88), but not purely prevalence. The high-score component's location moves with prevalence: bin means rise, and the M2 vs M1 likelihood ratio is decisive.
- **The pure-prevalence reading is overturned.** A model with fixed component shapes and varying weight is rejected.
- **A pure continuous shift is also rejected**, by BIC and by the empty-bin and variance-share mismatches.
- **The abstract's "combines high-score prevalence and within-bin variation" is supported and should be quantified** with the 0.88 / 0.01 / 0.10 split and these null comparisons.
- **The reviewer's predicted "success" pattern did not occur.** That pattern was: the slopes fall outside the shift null and inside the mixture null. The opposite held, and this should be reported as such.
- **ρ_MF = 0.93 is uninformative about construct identity.** All five nulls produce 0.92–0.99, as the reviewer said.

---

## P1-4 Seed/capacity propagation to the healing contrast (`p1_4_fit_propagation.py`)

Each fit uses its saved Hungarian-assigned topic and saved cut unchanged. The readouts are the manuscript's cell-pooled 11-patient estimands with exact p.

| Fit set | ΔS range (p range) | ΔM range (p range) | ΔW range (p range) |
|---|---|---|---|
| Frozen | 0.0293 (0.273) | 0.0892 (0.312) | 0.1514 (0.370) |
| 9 grid fits (K = 10/15/20 × 3 seeds, batch 256) | 0.0153–0.0358 (0.085–0.427) | 0.0223–0.0948 (0.267–0.324) | 0.057–0.202 (0.294–0.600) |
| 6 diagnostic fits (K = 15, batch 128) | 0.0017–0.0556 (0.109–0.894) | 0.0019–0.1413 (0.203–0.642) | 0.046–0.215 (0.224–0.700) |

- All 15 × 3 differences are positive, and none has p < 0.05.
- The fits with the weakest decoder match have the smallest contrasts:
  - K20 seed 2 (r = 0.60): ΔM 0.022, ΔW 0.057.
  - K15 seed 2 (r = 0.75): ΔM 0.029, ΔW 0.057.
  - The two diagnostic fits with r ≈ 0.50–0.54: ΔM about 0.002.
- **Re-choosing the topic per fit by maximum healer/non-healer ratio**:
  - The chosen topic equals the Hungarian match to frozen topic 0 in 9/9 grid fits under the cell-pooled ratio and 8/9 under the patient-S ratio. It does so in 3/6 diagnostic fits.
  - Selection-adjusted p (maximum over K topics) is 0.21–0.58 for the cell-pooled ratio and 0.54–0.92 for patient S across the grid.
  - Marginal one-sided ratio p values for the matched topic go as low as 0.030 in some fits. These are unadjusted and one-sided.
  - In one diagnostic fit (inherited 20k, seed 1), the cell-pooled selection-adjusted p is 0.027. This is one of 30 fit × statistic combinations and is not reproduced by any grid fit.

**Implications**
- **The positive sign is stable across fits; the size is not.** It ranges from about 0 to 1.6 times the frozen value and tracks decoder match quality. No fit makes the contrast distinguishable from the 330-allocation null, before or after selection adjustment.
- **"Consistent positive direction" is a selection-plus-dependence phenomenon, not independent replication.** In 9/9 grid re-fits (cell-pooled ratio; 8/9 by patient S; 3/6 diagnostic fits), the topic matched to frozen topic 0 is also the topic one would have selected by ratio.

---

## P1-5 Covariate balance and rank/hurdle statistics (`p1_5_covariates_rank.py`)

**Covariate balance**

| Covariate | Healed, median (n) | Non-healed, median (n) | Exact p (DFU) | HL shift, healed − non-healed [exact CI] |
|---|---|---|---|---|
| Age | 63 (7) | 52.5 (4) | 0.29 (MW); AUC 0.71 | +9 [−17, 29] |
| BMI | 34.9 (7) | 36.3 (4) | 0.79 | −1.2 [−36.4, 8.9] |
| HbA1c | 8.5 (4; 3 missing) | 7.6 (4) | 0.40 | +0.55 [−2.2, 7.9] |
| Wound area (cm²) | 1.00 (7) | 1.15 (4) | 0.96 | −0.10 [−29, 3.7] |
| Diabetes duration (y) | 13 (6; 1 missing) | 12.5 (4) | 1.00 | −0.5 [−37, 12] |
| Male | 2/7 | 2/4 | 0.58 (Fisher) | – |

- The exact Lehmann CI coverage is 0.958–0.971.
- The DM and H columns are in `results/p1_5_covariate_balance.csv`. The H column includes S18, who has a forearm specimen only.
- HbA1c is higher in DM (median 9.0) and DFU (8.5 / 7.6) than in H (5.5).

**Readout rank statistics** (7 vs 4, exact over 330 allocations; HL = Hodges–Lehmann shift):

| Readout | AUC | Exact p | HL shift [95.8% exact CI] |
|---|---|---|---|
| S | 0.714 | 0.315 | 0.023 [−0.018, 0.100] |
| M | 0.750 | 0.230 | 0.078 [−0.085, 0.329] |
| W | 0.714 | 0.315 | 0.081 [−0.215, 0.586] |
| F | 0.607 | 0.648 | 0.046 [−0.110, 0.286] |

- Hurdle W > 0.05: 5/7 vs 2/4, Fisher p = 0.58.
- Hurdle W > 0.02: 7/7 vs 2/4, Fisher p = 0.11.
- Conditional intensity (mean loading of above-cut fibroblasts, in the 9 patients with ≥10 such cells): AUC 0.64, p = 0.67.

**Within-DFU Spearman correlation with covariates (n = 11; 20,000 permutations)**
- W vs age: ρ = 0.26 (p = 0.45). W vs wound area: ρ = −0.16 (p = 0.63). Both match the reviewer.
- No readout–covariate |ρ| reaches p < 0.05. The largest is F vs HbA1c: ρ = 0.65, p = 0.09, n = 8.

**Implications**
- No measured covariate is clearly imbalanced, though healed patients are older by a median of 9 years. With n = 11, balance tests cannot rule out confounding.
- Rank and hurdle statistics agree with the mean-based ones: no association. The table should be added to the manuscript.

---

## P1-6 Lineage specificity, doublets and bulk construct test

### a. Lineage and doublets (`p1_6_lineage_doublets.py`, `p1_6c_allcell_without_melanocytes.py`)

**Topic-0 loading by lineage**

| Lineage | Mean topic 0 | Fraction > cut | Share of topic-0 mass |
|---|---|---|---|
| Melanocyte | 0.153 | 0.342 | 6.0% |
| Fibroblast | 0.116 | 0.188 | 66.3% |
| Pericyte/SMC | 0.022 | 0.015 | 12.2% |

The reviewer's figures are confirmed.

**Why melanocytes load on topic 0.** The topic-0 decoder itself carries the melanocyte programme:
- Rank within topic 0: DCT 15, TYRP1 20, PMEL 34, MLANA 42, TYR 281.
- For each of these genes, 99.7–100% of the decoder mass summed across the 15 topics sits in topic 0.
- None of the 15 topics is a melanocyte topic, so the model folded melanocyte genes into topic 0.
- Above-cut "melanocytes" (n = 455) are bona fide melanocytes, not fibroblast doublets:
  - About 92% express PMEL, MLANA, TYRP1 and DCT.
  - They express COL1A1 and DCN less often than below-cut melanocytes (27% vs 45% and 16% vs 38%).
  - Their predicted-doublet rate is 2.0%.
- Topic-0 top-gene content for all four melanocyte gate markers also means the manuscript's "lineage markers share no genes with the topic list" holds only for the 10–12-gene description list.
- COL1A1, a fibroblast gate marker, is decoder rank 18.

**Scrublet** (`scanpy.pp.scrublet`, per specimen, defaults, random_state = 0)
- 669 / 76,461 cells (0.9%) are called doublets.
- The doublet rate is lower above the cut than below it (0.38% vs 0.91%).
- Above-cut pericyte/SMC cells (n = 285) have twice the mean doublet score (0.066 vs 0.036). Within pericytes, ρ(doublet score, topic 0) = 0.45. They also show larger libraries and more frequent CHI3L1 (16%) and MMP1 (24%). Whether these are fibroblast–pericyte doublets or transitional cells is unresolved, and they make up only 1.5% of pericytes.

**Readouts after doublet removal**
- Healing contrasts are essentially unchanged: S 0.0293 (p 0.273), M 0.0894 (p 0.309), W 0.1518 (p 0.370). Specimen ρ_MF is 0.938.
- Removing the top 10% of doublet scores within each specimen gives W 0.151 (p 0.358) and ρ_MF 0.907.
- All-cell S without melanocytes: Δ 0.0353, p 0.182. Without melanocytes and pericytes: Δ 0.0459, p 0.206. Non-healers carry relatively more melanocyte topic-0 mass.

**Implications**
- "Fibroblast-associated" is **weakened for the all-cell score S**. Topic 0 is a composite of an inflammatory-fibroblast programme and the whole melanocyte programme. S is therefore not a fibroblast measurement, and it is better replaced by the fibroblast-gated readouts.
- Doublets do not explain the melanocyte loading or the fibroblast readouts.

### b. Bulk GSE80178 (`p1_6b_bulk_gse80178.py`)

**Hard data limitation.** No GPL16686 annotation is available offline. Gene symbols exist only in the submitters' processed file, which keeps only DFU-vs-foot-skin DE clusters (all FDR < 0.05 and |FC| ≥ 2; 74% are down in DFU). Any DFU-vs-intact test inside this universe is therefore selection-biased:
- 13 of 15 topics' signatures give p ≤ 0.0065.
- Random signatures from the universe are DFU-down (median Δ −0.85), so "random-signature percentile = 0" is uninformative.

**Results** (958 genes overlap the panel)
- **LIFT50 signature** (top 50 decoder-lift genes among topic 0's top-decile genes):
  - DFU − intact Δ = +0.69 z, exact p = 4/924 = 0.0043.
  - DFU − DM foot: +0.67, p = 0.036.
  - DM foot − non-DM foot: +0.04, p = 0.80 (n = 3 vs 3).
  - Excluding melanocyte genes: +0.91, p = 0.0022.
- **TOP50 by decoder probability** (dominated by DCN/IER3/SPARC and melanocyte genes): +0.17, p = 0.26.
- **Decoder-weighted score over all 958 genes:** −0.17, p = 0.33.
- **Topic-specificity control** (same LIFT50 construction for all 15 topics): topic 0 has the largest DFU-minus-intact difference (rank 1/15; next is topic 5, keratinocyte, +0.66).

**Implication.** This is weak, non-independent support that the CHI3L1/MMP-type part of topic 0 is wound-associated and not diabetes-associated. It is consistent with P1-2. Its magnitude cannot be interpreted until the full-array annotation is available.

---

## Claim-by-claim summary

| Manuscript claim | Verdict | Key evidence |
|---|---|---|
| Healing association is unvalidated | **Supported, strengthened** | Selection-adjusted p 0.61–0.89 (patient), 0.245 (original statistic). Selection loops at chance (LOPO AUC 0.39–0.57). No fit gives p < 0.05. |
| Positive point estimates across sensitivities / "warrants prospective evaluation" | **Overturned as evidence** | Topic 0 was the maximum-ratio topic when named (log line 37; docs/05 §5). The best-of-15 null reproduces the observed ratio. In 9/9 grid re-fits the matched topic is also the max-ratio topic (cell-pooled ratio; 8/9 by patient S). |
| Topic 0 chosen without outcome information (implied) | **Overturned / must be disclosed** | Repository record above. The anchor list includes literature healer markers. |
| Mean combines prevalence and within-bin variation | **Supported, now quantified** | Var(A)/Var(Ȳ) = 0.88, Var(R) 0.01, 2Cov 0.10. Fixed-component mixture rejected (high-bin slope 0.31 vs null ≈ 0). M2 vs M1 2LR = 2,037, p = 1/201. |
| ρ_MF = 0.93 as construct evidence / robustness across fits | **Weakened** | Every continuous and mixture null gives 0.92–0.99. |
| Healthy-skin specimen is a counterexample to healing specificity | **Weakened, needs reframing** | 13/15 intact-foot patients have W ≤ 0.019 (S19 0.84 and DM S16 0.33 are the exceptions). DFU 9/11 > 0.02. Forearm 0/11. DFU vs intact rank p = 0.0055; hurdle p = 0.00095. DM vs H shows no difference. |
| Anatomical positive control ("responds to a known tissue difference") | **Overturned** | Effect comes from three DFU ulcer pairs plus S19. Intact skin without S19 shows Δ ≤ 0.007. |
| "Fibroblast-associated" topic | **Weakened for all-cell S** | Melanocytes have the highest mean loading because the decoder holds the melanocyte programme. 34% of topic-0 mass is outside fibroblasts. The fibroblast-gated readouts are unaffected by doublet removal. |
| Seed/capacity: persistent association, changing identity | **Supported, and extended to outcome** | Sign stable, size varies (ΔW 0.046–0.215) and tracks match quality. All p ≥ 0.085. |
| Covariate confounding not shown | **Now addressed (descriptively)** | No significant imbalance. Healed patients are 9 years older (HL; p = 0.29). n = 11 cannot exclude confounding. |

## Caveats

- **Exchangeability.** All permutation p values assume exchangeability of observational labels. Contrasts in P1-2 that are not the manuscript's prespecified readouts are exploratory.
- **Multiplicity in P1-2.** Many contrasts × readouts were computed and none is adjusted. Rank and hurdle statistics were chosen with knowledge of the reviewer's computation.
- **P1-2 data quality.** DM specimens G1A, G2A and G3A have low gene coverage. Two forearm specimens have very few fibroblasts (G17: 35; G11: 62). Site tests on intact skin have only 5 pairs, so p ≥ 0.0625 by construction.
- **P1-3 null specification.**
  - Parametric nulls use logit-Gaussian forms or empirical shapes. Conclusions depend on these specifications: high-bin slopes discriminate, ρ_MF does not.
  - The bootstrap conditions on fitted specimen parameters; "patient-level" variation enters only through those specimen parameters.
  - Scores were clipped to [1e-7, 1−1e-7], which affects nothing because no score lies outside that range.
- **P1-4.** The per-fit cuts are the fits' own descriptive midpoints (training-only for the diagnostic fits). The frozen configuration (batch 512) was not refitted; reviewer item 5 is not done here.
- **P1-6.** Scrublet's automatic threshold flags few cells. The top-10% sensitivity covers this. Doublet calling cannot detect homotypic doublets.
- **Anchor genes.** The influence of the anchor genes on the topic itself is untested (needs refits without anchors).
- **GSE80178.** Bulk results are confined to a DE-filtered gene universe (see P1-6b).

## File index (`results/`)

- **P1-1:** `p1_1_selection.json`, `p1_1_topic_contrasts.csv`, `p1_1_patient_readouts_all_topics.csv`
- **P1-2:**
  - `p1_2_projected_cells.parquet` (44 specimens, frozen pipeline)
  - `p1_2_projection_specimens.csv`
  - `p1_2_reproduction_check.json`
  - `p1_2_patient_readouts.csv`, `p1_2_specimen_readouts.csv`
  - `p1_2_group_contrasts.csv`, `p1_2_paired_site.csv`
  - `p1_2_decomposition.json`
- **P1-3:** `p1_3_prevalence_intensity.json`, `p1_3_specimen_fits.csv`, `p1_3_bootstrap_draws.csv.gz`
- **P1-4:** `p1_4_fit_contrasts.csv`, `p1_4_fit_reselection.csv`, `p1_4_fit_patient_readouts.csv`, `p1_4_fit_propagation.json`
- **P1-5:** `p1_5_covariate_balance.csv`, `p1_5_patient_table.csv`, `p1_5_readout_covariate_spearman.csv`, `p1_5_covariates_rank.json`
- **P1-6:**
  - `p1_6_lineage_doublets.json`
  - `p1_6_lineage_table_before.csv`, `p1_6_lineage_table_after_doublets.csv`
  - `p1_6_coexpression_by_lineage_bin.csv`, `p1_6_doublet_scores.parquet`
  - `p1_6b_bulk_gse80178.json`
  - `p1_6c_allcell_without_melanocytes.json`
- **Manifests:** `*_input_manifest.json` (per script) and the top-level `MANIFEST_SHA256.json`.
