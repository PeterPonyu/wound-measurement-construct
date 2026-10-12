"""Measurement job executors: semantics tests, stability retraining and the design calculator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..common.io import atomic_write_json, sanitize, sha256_file
from ..common.provenance import code_digest
from . import design, programs, semantics, stability

SPECIMENS = "outputs/biological_expansion/measurement/specimen_readouts.csv"
PAPER_CUT = 0.1842958


def _lineage_semantics(values: np.ndarray, frame: pd.DataFrame, lineages, config: dict, *, scale: str, cut=None) -> dict:
    out = {}
    for lineage in lineages:
        mask = (frame["lineage"] == lineage).to_numpy()
        if mask.sum() < config.get("min_cells", 50) * 3:
            out[str(lineage)] = {'available': False, 'reason': 'fewer than three specimens worth of cells',
                                 'n_cells': int(mask.sum())}
            continue
        if not np.isfinite(values[mask]).all():
            out[str(lineage)] = {'available': False, 'reason': 'program scores contain nonfinite values',
                                 'n_cells': int(mask.sum())}
            continue
        _, summary = semantics.semantics_test(values[mask], frame.loc[mask, "specimen"], frame.loc[mask, "unit"],
                                              scale=scale, cut=cut, n_splits=config.get("n_splits", 10),
                                              seed=config.get("seed", 0), min_cells=config.get("min_cells", 50),
                                              max_cells=config.get('max_cells', 4000),
                                              min_positive=config.get('min_positive', 20))
        out[str(lineage)] = summary
    return out


def run_semantics_job(job, *, protocol: dict, root) -> dict:
    """One cohort x program family: every program in every lineage with enough cells."""
    root = Path(root)
    config = protocol["semantics"]
    family = job["params"]["family"]
    scale = config.get("scale_by_family", {}).get(family, "logit")
    if job["params"]["source"] == "discovery":
        cells = pd.read_parquet(root / stability.CELLS)
        frame = pd.DataFrame({"specimen": cells["gsm"], "unit": cells["patient_id"], "lineage": cells["celltype"]})
        if family == "topic":
            scores = {f"topic_{k}": cells[f"topic_{k}"].to_numpy(dtype=float) for k in range(15)}
        else:
            proportions, _ = stability.load_discovery_panel_counts(root)
            panel = np.load(root / stability.DISCOVERY / "panel.npy", allow_pickle=True)
            scores = _family_scores(family, proportions, panel, root, config)
    else:
        from ..data.standard import load_standard

        adata = load_standard(job["params"]["dataset_file"])
        frame = pd.DataFrame({"specimen": adata.obs["sample"].to_numpy(),
                              "unit": adata.obs["individual"].to_numpy(),
                              "lineage": programs.gate_lineages(adata.X, adata.var_names.to_numpy(dtype=object))})
        scores = _family_scores(family, adata.X, adata.var_names.to_numpy(dtype=object), root, config)
    lineages = config.get("lineages") or sorted(frame["lineage"].unique())
    results = {}
    for name, values in scores.items():
        cut = PAPER_CUT if (family == "topic" and name == "topic_0") else None
        results[name] = _lineage_semantics(np.asarray(values, dtype=float), frame, lineages, config, scale=scale, cut=cut)
    return {"cohort": job["dataset_id"], "family": family, "scale": scale, "programs": results,
            'validation_scope': 'conditional_on_supplied_scores_and_lineage_labels',
            'end_to_end_cross_fitted': False,
            'score_construction': {'topic': 'frozen_discovery_encoder',
                                   'module': 'frozen_gene_sets_with_whole_cohort_control_bins',
                                   'nmf': 'HVG_and_NMF_fitted_on_whole_cohort'}[family],
            'lineage_construction': ('supplied_discovery_labels' if job['params']['source'] == 'discovery'
                                     else 'whole_cohort_calibrated_marker_gate'),
            'inference_policy': 'descriptive_only_not_discrete_state_or_clinical_validation'}


def _family_scores(family: str, counts, genes, root: Path, config: dict) -> dict:
    discovery = root / stability.DISCOVERY
    if family == "topic":
        theta, covered = programs.topic_theta(counts, genes, discovery)
        return {f"topic_{k}": theta[:, k] for k in range(theta.shape[1])}
    if family == "module":
        beta = np.load(discovery / "beta.npy")
        panel = np.load(discovery / "panel.npy", allow_pickle=True)
        sets = {f"module_{k}": members for k, members in
                programs.top_genes_per_topic(beta, panel, n=config.get("module_genes", 30)).items()}
        table = programs.module_scores(counts, genes, sets, n_control=config.get("module_controls", 50))
        return {column: table[column].to_numpy() for column in table.columns}
    if family == "nmf":
        fitted = programs.fit_nmf(counts, genes, n_components=config.get("nmf_components", 15),
                                  n_hvg=config.get("nmf_hvg", 2000), seed=config.get("seed", 0))
        weights = programs.nmf_proportions(fitted, counts, genes)
        return {f"nmf_{k}": weights[:, k] for k in range(weights.shape[1])}
    raise ValueError(f"unknown program family {family!r}")


def run_stability_job(job, *, protocol: dict, root, device: str = "cpu", out_dir=None) -> dict:
    root = Path(root)
    config = protocol["stability"]
    K, seed = int(job["params"]["K"]), int(job["seed"])
    panel_fit = None
    if config.get('panel_selection') is not None:
        if config.get('patient_partition') is None:
            raise ValueError('training-only panel requires a frozen patient partition')
        proportions, cells, panel, panel_fit = stability.load_training_panel_counts(
            root, config['patient_partition'], config['panel_selection'])
    else:
        proportions, cells = stability.load_discovery_panel_counts(root)
        panel = None
    identity = None
    if out_dir is not None:
        identity = next((name for name in ('reconstructed_raw_cell_id', 'cell_id') if name in cells), None)
        if identity is None:
            raise ValueError('cell identity column is required to persist ordered projections')
        stability._identifiers(cells[identity].to_numpy(), 'cell_ids')
        if panel is None:
            panel = np.load(root / stability.DISCOVERY / 'panel.npy', allow_pickle=True)
        stability._identifiers(panel, 'panel')
    partition_config = config.get('patient_partition')
    training_values, extra, partition = proportions, {}, None
    if partition_config is not None:
        training_mask, eval_rows, partition = stability.patient_partition(cells, partition_config)
        training_values = proportions[training_mask]
        extra = {'diagnostic_inputs': {name: {'proportions': proportions[rows],
                  'units': cells.iloc[rows].patient_id.to_numpy(dtype=str)} for name, rows in eval_rows.items()},
                 'diagnostic_every': config['diagnostic_every']}
    steps = job['params'].get('steps', config.get('steps'))
    if config.get('step_budgets') is not None and steps not in config['step_budgets']:
        raise ValueError('job optimizer budget is outside the frozen protocol')
    trained = stability.train_topic_model(training_values, K=K, seed=seed, steps=steps,
                                          batch_size=config["batch_size"], lr=config["lr"],
                                          kl_weight=config["kl_weight"], device=device, **extra)
    if partition is not None:
        if panel_fit is not None:
            partition['limitations'] = ['inherited_discovery_raw_gene_universe_and_cell_QC',
                                       'supplied_patient_and_lineage_labels']
            partition['panel_fit'] = panel_fit
        trained['training_config'].update(patient_partition=partition,
                                          optimization_proportions_sha256=stability._matrix_hash(training_values))
    # Always use the same deterministic CPU path for the saved/reloaded projection.
    theta = stability.project_theta(trained["state"], proportions, K, device='cpu')
    checkpoint = None
    if out_dir is not None:
        inputs = [stability.COUNTS, stability.GENES, stability.CELLS,
                  stability.DISCOVERY + '/panel.npy', stability.DISCOVERY + '/beta.npy']
        checkpoint = stability.save_inference_checkpoint(out_dir, trained, proportions, theta, panel=panel,
            cell_ids=cells[identity].to_numpy(), provenance={'code_digest': code_digest(),
                'input_sha256': {name: sha256_file(root / name) for name in inputs},
                'cell_identity_column': identity, 'protocol': protocol,
                'identity_scope': 'binds_provided_row_order_not_independent_raw_join_verification'})
        if config.get('retain_undefined_reference_match'):
            atomic_write_json(Path(out_dir) / 'fit_trace.json', sanitize({key: trained[key] for key in
                ('training_config', 'training_trace', 'evaluation_trace')}))
    # Save valid numeric fits before descriptive matching; a degenerate decoder
    # is evidence. Explicit scoped protocols retain undefined matches, no rescue.
    reference = np.load(root / stability.DISCOVERY / "beta.npy")
    try:
        if panel_fit is not None:
            reference_panel = np.load(root / stability.DISCOVERY / 'panel.npy', allow_pickle=True)
            matched = stability.match_panel_topics(reference, reference_panel, trained['beta'], panel)
        else:
            matched = stability.match_topics(reference, trained["beta"])
    except ValueError as error:
        if (not config.get('retain_undefined_reference_match') or not any(reason in str(error) for reason in
                ('constant topic rows', 'too few shared genes'))):
            raise
        matched = {'assignment': {}, 'assigned_correlation': {}, 'best_match': {0: None},
            'best_correlation': {0: None}, 'correlation': None, 'status': 'undefined_no_rescue',
            'reason': str(error), 'primary_rule': 'hungarian_one_to_one',
            'unmatched_reference': list(range(reference.shape[0])), 'unmatched_new': list(range(K))}
    topic = matched['assignment'].get(0)
    result = {"K": K, "seed": seed, "losses": trained["losses"], "match_to_reference": matched,
              'primary_match_status': 'unmatched' if topic is None else 'assigned',
              'primary_match_rule': 'hungarian_one_to_one', 'fibroblast_topic': topic,
              'fibroblast_topic_correlation': matched['assigned_correlation'].get(0),
              'rowwise_best_sensitivity': {'topic': matched['best_match'][0],
                                          'correlation': matched['best_correlation'][0], 'used_for_primary': False},
              'cut': None, 'cut_scope': 'full_sample_descriptive_only_not_used_for_held_unit_density',
              'mean_fraction_spearman': None, 'patient_readouts': [], 'specimen_readouts': [],
              'inference_checkpoint': checkpoint,
              'scope': 'computational_retraining_not_additional_biological_patients'}
    if panel_fit is not None:
        result['panel_fit'] = panel_fit
    if partition is not None:
        result.update(steps=steps, patient_partition=partition,
                      diagnostic_every=config['diagnostic_every'],
                      training_trace=trained['training_trace'], evaluation_trace=trained['evaluation_trace'],
                      scope='patient_partition_optimization_diagnostic_not_end_to_end_or_clinical_validation',
                      cut_scope='training_patient_fibroblasts_only_descriptive',
                      semantics={'available': False, 'reason': 'not_requested_in_partition_convergence_lane'})
        # Keep a failed/unassigned primary readout explicit; never rescue with
        # a row-wise alternative or choose a cut using held-patient cells.
        if topic is not None:
            fibroblast = (cells.celltype == 'fibroblast').to_numpy()
            values = theta[:, topic]
            training_fib = training_mask & fibroblast
            if training_fib.sum() >= 4 and np.ptp(values[training_fib]) > 1e-12:
                cut = stability.gmm_midpoint(values[training_fib])
                frame = pd.DataFrame({'specimen': cells.loc[fibroblast, 'gsm'].to_numpy(),
                    'unit': cells.loc[fibroblast, 'patient_id'].to_numpy(), 'value': values[fibroblast]})
                table = frame.groupby(['unit', 'specimen'])['value'].agg(mean='mean', n_cells='size',
                                           fraction=lambda v: float(np.mean(v > cut)))
                patients = table.groupby('unit').agg(mean=('mean', 'mean'), fraction=('fraction', 'mean'),
                                         n_cells=('n_cells', 'sum'), n_specimens=('mean', 'size'))
                patients['partition'] = ['held' if p in partition['held_patients'] else 'train' for p in patients.index]
                result.update(cut=cut, patient_readouts=patients.reset_index().to_dict(orient='records'),
                    specimen_readouts=table.reset_index().to_dict(orient='records'),
                    patient_aggregation='equal_specimen_mean_not_cell_pooled', primary_readout_status='available')
            else:
                result['primary_readout_status'] = 'insufficient_or_constant_training_fibroblast_loadings'
        else:
            result['primary_readout_status'] = 'reference_topic_0_unmatched_no_primary_readout'
        if out_dir is not None:
            atomic_write_json(Path(out_dir) / 'convergence.json', sanitize({key: result[key] for key in
                ('steps', 'patient_partition', 'diagnostic_every', 'training_trace', 'evaluation_trace', 'scope')}))
            atomic_write_json(Path(out_dir) / 'readouts.json', sanitize({'specimens': result['specimen_readouts'],
                'patients': result['patient_readouts'], 'cut': result['cut'], 'cut_scope': result['cut_scope']}))
        return result
    if topic is None:
        result['semantics'] = {'available': False, 'reason': 'reference_topic_0_unmatched_no_primary_readout'}
        return result
    fibroblast = (cells["celltype"] == "fibroblast").to_numpy()
    loading = theta[fibroblast, topic]
    if len(loading) < 4 or np.ptp(loading) <= 1e-12:
        result['semantics'] = {'available': False, 'reason': 'insufficient or constant fibroblast loadings'}
        return result
    cut = stability.gmm_midpoint(loading)
    frame = pd.DataFrame({'specimen': cells.loc[fibroblast, 'gsm'].to_numpy(), 'value': loading,
                          'unit': cells.loc[fibroblast, 'patient_id'].to_numpy()})
    per_specimen = frame.groupby(['unit', 'specimen'])['value'].agg(mean='mean', n_cells='size',
                                                                                fraction=lambda v: float(np.mean(v > cut)))
    rho = (float(spearmanr(per_specimen['mean'], per_specimen['fraction']).statistic)
           if per_specimen['mean'].nunique() > 1 and per_specimen['fraction'].nunique() > 1 else None)
    patients = per_specimen.groupby('unit').agg(mean=('mean', 'mean'), fraction=('fraction', 'mean'),
                                               n_cells=('n_cells', 'sum'), n_specimens=('mean', 'size'))
    # cut=None: each leave-patient-out fold estimates its own mixture midpoint, so H1 never sees the held unit.
    semantic_config = protocol['semantics']
    table, summary = semantics.semantics_test(loading, cells.loc[fibroblast, 'gsm'], cells.loc[fibroblast, 'patient_id'],
        scale='logit', cut=None, n_splits=semantic_config.get('n_splits', 10), seed=seed,
        min_cells=semantic_config.get('min_cells', 50), max_cells=semantic_config.get('max_cells', 4000),
        min_positive=semantic_config.get('min_positive', 20))
    result.update(cut=cut, mean_fraction_spearman=rho, semantics=summary,
                  specimen_readouts=per_specimen.reset_index().to_dict(orient='records'),
                  patient_readouts=patients.reset_index().to_dict(orient='records'),
                  patient_aggregation='equal_specimen_mean_not_cell_pooled')
    if out_dir is not None:
        atomic_write_json(Path(out_dir) / 'readouts.json', sanitize({'specimens': result['specimen_readouts'],
            'patients': result['patient_readouts'], 'semantics_rows': table.to_dict(orient='records'),
            'cut_scope': result['cut_scope']}))
    return result


def run_design_job(job, *, protocol: dict, root) -> dict:
    config = protocol["design"]
    specimens = pd.read_csv(Path(root) / SPECIMENS)
    results = {}
    for readout in config.get("readouts", ["high_state_fraction"]):
        components = design.estimate_components(specimens, value=readout)
        sd = float(np.sqrt(components["sigma_b2"] + components["sigma_s2"]))
        deltas = [round(multiple * sd, 6) for multiple in config.get("delta_multiples_of_sd", [0.5, 0.8, 1.0])]
        grid = design.design_grid(components, patients=config["patients_per_group"], cells=config["cells_per_specimen"],
                                  specimens=config["specimens_per_patient"], deltas=deltas,
                                  n_sim=config.get("n_sim", 2000), seed=config.get("seed", 0))
        results[readout] = {"components": components, "deltas": deltas, "grid": grid.to_dict(orient="records")}
    return results
